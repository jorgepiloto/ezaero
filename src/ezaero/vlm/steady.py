"""
The :mod:`ezaero.vlm.steady` module includes a Vortex Lattice Method
implementation for lifting surfaces.

References
----------
.. [1] Katz, J. et al., *Low-Speed Aerodynamics*, 2nd ed, Cambridge University
   Press, 2001: Chapter 12
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from ezaero.vlm.meshing import mesh_surface
from ezaero.vlm.plotting import (
    plot_cl_distribution_on_wing,
    plot_control_points,
    plot_panels,
)
from ezaero.vlm.utils import biot_savart


class VLM_solver:
    def __init__(
        self, part_list, mesh_conf=None, alpha=np.pi / 36, Uinf=10.0, rho=1.225
    ):
        """ Initializes the simulation """

        # Generate a private variable for checking if simulation was executed
        self._executed = False

        # Count the number of parts
        self.N_parts = len(part_list)

        # Check if mesh_list available and shares part_list dimensions
        if not mesh_conf:
            mesh_conf = (np.array([4, 12]) * np.ones((self.N_parts, 2))).astype(int)
        else:
            assert len(mesh_conf) == self.N_parts

        # Store simulation conditions
        self.alpha, self.Uinf, self.rho = alpha, Uinf, rho

        # Allocate results variable
        self.all_parts = part_list
        self.all_meshconf = [mn for mn in mesh_conf]
        self.all_panels = []
        self.all_cpoints = []
        self.all_vortex_panels = []
        self.all_normals = []
        self.all_surfaces = []
        self.all_wakes = []
        self.all_results = []

    def plot_model(self, fig=None, color="lightgray"):
        """ Returns a graphical representation of model's geometry """

        # Check if figure available
        if not fig:
            fig = go.Figure()
            fig.update_layout(scene_aspectmode="data")

        # Add each part to the figure
        for part in self.all_parts:
            fig = part.plot(fig=fig, color=color, label=part.name)

        return fig

    def plot_mesh(self, fig=None, color="blue"):

        # Check if figure available
        if not fig:
            fig = go.Figure()
            fig.update_layout(scene_aspectmode="data")

        if not self._executed:
            self._build_panels()

        for panels, cpoints in zip(self.all_panels, self.all_cpoints):
            fig = plot_panels(panels, fig=fig)
            fig = plot_control_points(cpoints, fig=fig)

        return fig

    def plot_wake(self, fig=None, color="red"):

        if not fig:
            fig = go.Figure()
            fig.update_layout(scene_aspectmode="data")

        for part_wakes in self.all_wakes:
            for r_wake in part_wakes:
                wake_trace = go.Scatter3d(
                    x=list(r_wake[:, 0]),
                    y=list(r_wake[:, 1]),
                    z=list(r_wake[:, 2]),
                    showlegend=False,
                    marker=dict(color=color, size=5),
                )
                fig.add_trace(wake_trace)

        return fig

    def _build_panels(self):
        """
        Build panels and collocation points.

        Creates
        -------
        self.wing_panels : np.ndarray, shape (m, n, 4, 3)
            Array containing the (x,y,z) coordinates of all wing panel vertices.
        self.cpoints : np.ndarray, shape (m, n, 3)
            Array containing the (x,y,z) coordinates of all collocation points.
        """

        for part, MN in zip(self.all_parts, self.all_meshconf):

            # Solve panels and cpoints for model part
            panels, cpoints = mesh_surface(part, *MN)

            # Append to global data dictionary
            self.all_panels.append(panels)
            self.all_cpoints.append(cpoints)

    def _build_vortex_panels(self):
        """
        Creates
        -------
        aic : np.ndarray, shape (m, n, 4, 3)
            Array containing the (x,y,z) coordinates of all vortex panel vertices.
        """

        for part, (m, n), panels in zip(
            self.all_parts, self.all_meshconf, self.all_panels
        ):

            # Solve panels coordinates positions
            X, Y, Z = [panels[:, :, :, i] for i in range(3)]

            dxv = (X[:, :, [3, 2, 2, 3]] - X[:, :, [0, 1, 1, 0]]) / 4
            XV = X + dxv

            YV = Y

            ZV = np.empty((m, n, 4))
            Z01 = Z[:, :, [0, 1]]
            dzv = Z[:, :, [3, 2]] - Z01
            ZV[:, :, [0, 1]] = Z01 + 1 / 4 * dzv
            ZV[:, :, [3, 2]] = Z01 + 5 / 4 * dzv

            vortex_panels = np.stack([XV, YV, ZV], axis=3)

            self.all_vortex_panels.append(vortex_panels)

    def _calculate_panel_normal_vectors(self):
        """
        Calculate the normal vector for each wing panel, approximated
        by the direction of the cross product of the panel diagonals.

        Creates
        -------
        normals : np.ndarray, shape (m, n, 3)
            Array containing the normal vectors to all wing panels.
        """

        for panels in self.all_panels:

            d1 = panels[:, :, 2] - panels[:, :, 0]
            d2 = panels[:, :, 1] - panels[:, :, 3]
            nv = np.cross(d1, d2)

            normals = nv / np.linalg.norm(nv, ord=2, axis=2, keepdims=True)

            self.all_normals.append(normals)

    def _calculate_wing_planform_surface(self):
        """
        Calculate the planform projected surface of all wing panels.

        Creates
        -------
        panel_surfaces : np.ndarray, shape (m, n)
            Array containing the planform (projected) surface of each panel.
        """

        for panels in self.all_panels:

            x, y = [panels[:, :, :, i] for i in range(2)]

            # shoelace formula to calculate flat polygon area (XY projection)
            einsum_str = "ijk,ijk->ij"
            d1 = np.einsum(einsum_str, x, np.roll(y, 1, axis=2))
            d2 = np.einsum(einsum_str, y, np.roll(x, 1, axis=2))
            panel_surfaces = 0.5 * np.abs(d1 - d2)

            self.all_surfaces.append(panel_surfaces)

    def _build_wake(self, offset=300):
        """
        Build the steady wake vortex panels.

        offset : int
            Downstream distance at which the steady wake is truncated
            (expressed in multiples of the wingspan)

        Creates
        -------
        wake : np.ndarray, shape (n, 4, 3)
            Array containing the (x,y,z) coordinates of the panel vertices that
            form the steady wake.
        """

        for part, panels, (m, n), vortex_panels in zip(
            self.all_parts, self.all_panels, self.all_meshconf, self.all_vortex_panels
        ):

            wake = np.empty((n, 4, 3))
            wake[:, [0, 1]] = vortex_panels[m - 1][:, [3, 2]]
            delta = (
                offset
                * part.planform_wingspan
                * np.array(
                    [
                        np.cos(self.alpha),
                        0,
                        np.sin(self.alpha),
                    ]
                )
            )
            wake[:, [3, 2]] = wake[:, [0, 1]] + delta

            self.all_wakes.append(wake)

    def _calculate_wing_influence_matrix(self):
        """
        Calculate influence matrix (wing contribution).

        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Wing contribution to the influence matrix.
        """

        # Number of panels
        self.N_panels = np.sum(
            [panels.shape[0] * panels.shape[1] for panels in self.all_panels]
        )

        # Convert to array
        cpoints = np.asarray(self.all_cpoints)
        vortex_panels = np.asarray(self.all_vortex_panels)
        normals = np.asarray(self.all_normals)

        r = vortex_panels.reshape((self.N_panels, 1, 4, 3)) - cpoints.reshape(
            (1, self.N_panels, 1, 3)
        )

        vel = biot_savart(r)
        nv = normals.reshape((self.N_panels, 3))
        self.all_aic_wing = np.einsum("ijk,jk->ji", vel, nv)

    def _calculate_wake_wing_influence_matrix(self):
        """
        Calculate influence matrix (steady wake contribution).

        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Wake contribution to the influence matrix.
        """

        # Allocate final wake wing influence matix
        self.all_aic_wake = np.zeros((self.N_panels, self.N_panels))

        # Solve wake in each geometric part
        for i, part in enumerate(self.all_parts):

            # Convert to array
            cpoints = np.asarray(self.all_cpoints)[i]
            wake = self.all_wakes[i]
            normals = np.asarray(self.all_normals)[i]
            m, n = np.asarray(self.all_meshconf)[i, :]
            mn = m * n

            r = wake[:, np.newaxis, :, :] - cpoints.reshape((1, mn, 1, 3))
            vel = biot_savart(r)
            nv = normals.reshape((mn, 3))

            aic_wake = np.zeros((mn, mn))

            aic_wake[:, -n:] = np.einsum("ijk,jk->ji", vel, nv)
            lim, _ = aic_wake.shape

            self.all_aic_wake[
                i * lim : (i + 1) * lim, i * lim : (i + 1) * lim
            ] = aic_wake

    def _calculate_influence_matrix(self):
        """
        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Influence matrix, including wing and wake contributions.
        """
        self._calculate_wing_influence_matrix()
        self._calculate_wake_wing_influence_matrix()
        self.all_aic = self.all_aic_wing + self.all_aic_wake

    def _calculate_rhs(self):
        """
        Returns
        -------
        rhs : np.ndarray, shape (m * n, )
            RHS vector.
        """

        normals = np.asarray(self.all_normals)

        u = self.Uinf * np.array([np.cos(self.alpha), 0, np.sin(self.alpha)])
        self.rhs = -np.dot(normals.reshape(self.N_panels, -1), u)

    def _solve_net_panel_circulation_distribution(self):
        """
        Calculate panel net circulation by solving the linear equation:
        AIC * circulation = RHS

        Creates
        -------
        net_circulation : np.ndarray, shape (m, n)
            Array containing net circulation for each panel.
        """

        # Solve whole solution
        g = np.linalg.solve(self.all_aic, self.rhs)

        # Allocate panels gamma solutions
        self.all_circulations = []

        # Deconstruct matrix solution
        for (m, n) in self.all_meshconf:

            # Total number of panels for the part
            mn = m * n

            # Get the first elements of circulation solution and remove them
            panel_gamma = g[:mn].reshape(m, n)
            self.all_circulations.append(panel_gamma)
            g = g[mn:]

    def _calculate_aero_distributions_from_circulation(self):
        """ Solves for all aerodynamic properties """

        # Allocate variables
        self.all_dL = []
        self.all_dp = []
        self.all_cl = []
        self.all_cl_wing = []

        for part, panels, (m, n), surfaces, circulation in zip(
            self.all_parts,
            self.all_panels,
            self.all_meshconf,
            self.all_surfaces,
            self.all_circulations,
        ):

            rho, Uinf = self.rho, self.Uinf
            bp = part.planform_wingspan
            dL = np.zeros((m,n))

            for i in range(m):
                for j in range(n):
                    if i == 1:
                        ddL = circulation[i,j] * rho * Uinf * bp / n
                    else:
                        ddL = (circulation[i,j] - circulation[i-1, j]) * rho * Uinf * bp / n
                    dL[i,j] = ddL

            dp = dL / surfaces
            cl = dp / (0.5 * rho * Uinf ** 2)
            cl_wing = dL.sum() / (0.5 * rho * Uinf ** 2 * surfaces.sum())

            self.all_dL.append(dL)
            self.all_dp.append(dp)
            self.all_cl.append(cl)
            self.all_cl_wing.append(cl_wing)

    def run(self, force=False):
        """ Returns simulation results """

        # Check if simualtion was previously run or forced condition
        if self._executed and force is False:
            return self.results

        else:

            # Generate panels over each part of the model
            self._build_panels()
            self._build_vortex_panels()
            self._calculate_panel_normal_vectors()
            self._calculate_wing_planform_surface()
            self._build_wake()
            self._calculate_wing_influence_matrix()
            self._calculate_wake_wing_influence_matrix()
            self._calculate_influence_matrix()
            self._calculate_rhs()
            self._solve_net_panel_circulation_distribution()
            self._calculate_aero_distributions_from_circulation()

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


class VLM_solver:
    def __init__(
        self, part_list, mesh_conf=None, alpha=np.pi / 36, Uinf=10.0, rho=1.225
    ):
        """ Initializes the simulation """

        # Count the number of parts
        self.N_parts = len(part_list)

        # Check if mesh_list available and shares part_list dimensions
        if not mesh_conf:
            mesh_conf = (np.array([4, 12]) * np.ones((self.N_parts,2))).astype(int)
        else:
            assert len(mesh_conf) == self.N_parts

        # Assign part and mesh configuration collections
        self.part_list, self.mesh_conf = part_list, mesh_conf

        # Build a dictionary for model data: {"part", [MN, panels, cpoints]}
        self.parts_data = {part: [NM] for (part, NM) in zip(part_list, mesh_conf)}

        # Store simulation conditions
        self.alpha, self.Uinf, self.rho = alpha, Uinf, rho

        # Generate a private variable for checking if simulation was executed
        self._executed = False

        # Allocate results variable
        self.results = None

    def plot_model(self, fig=None, color="lightgray"):
        """ Returns a graphical representation of model's geometry """

        # Check if figure available
        if not fig:
            fig = go.Figure()
            fig.update_layout(scene_aspectmode="data")

        # Add each part to the figure
        for part in self.part_list:
            fig = part.plot(fig=fig, color=color, label=part.name)

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


        for part, MN in zip(self.part_list, self.mesh_conf):

            # Solve panels and cpoints for model part
            panels, cpoints = mesh_surface(part, *MN)

            # Append to global data dictionary
            self.parts_data.update({part: [MN, panels, cpoints]})

    def _build_vortex_panels(self):
        """
        Creates
        -------
        aic : np.ndarray, shape (m, n, 4, 3)
            Array containing the (x,y,z) coordinates of all vortex panel vertices.
        """

        for part, data in self.parts_data.items():

            # Unpack useful data
            MN, panels, _ = data
            m, n = MN

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

            self.parts_data[part].append(vortex_panels)

    def _calculate_panel_normal_vectors(self):
        """
        Calculate the normal vector for each wing panel, approximated
        by the direction of the cross product of the panel diagonals.

        Creates
        -------
        normals : np.ndarray, shape (m, n, 3)
            Array containing the normal vectors to all wing panels.
        """

        for part, data in self.parts_data.items():

            # Unpack variables
            panels = data[1]

            d1 = panels[:, :, 2] - panels[:, :, 0]
            d2 = panels[:, :, 1] - panels[:, :, 3]
            nv = np.cross(d1, d2)

            normals = nv / np.linalg.norm(nv, ord=2, axis=2, keepdims=True)

            self.parts_data[part].append(normals)

    def _calculate_wing_planform_surface(self):
        """
        Calculate the planform projected surface of all wing panels.

        Creates
        -------
        panel_surfaces : np.ndarray, shape (m, n)
            Array containing the planform (projected) surface of each panel.
        """

        for part, data in self.parts_data.items():

            # Unpack data
            panels = data[1]

            x, y = [panels[:, :, :, i] for i in range(2)]

            # shoelace formula to calculate flat polygon area (XY projection)
            einsum_str = "ijk,ijk->ij"
            d1 = np.einsum(einsum_str, x, np.roll(y, 1, axis=2))
            d2 = np.einsum(einsum_str, y, np.roll(x, 1, axis=2))
            panel_surfaces = 0.5 * np.abs(d1 - d2)

            self.parts_data[part].append(panel_surfaces)

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

        for part, data in self.parts_data.items():

            # Unpack data
            (m, n), panels, _, vortex_panels, *_ = data

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

            self.parts_data[part].append(wake)


    def _calculate_wing_influence_matrix(self):
        """
        Calculate influence matrix (wing contribution).

        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Wing contribution to the influence matrix.
        """
        r = self.vortex_panels.reshape(
            (self.mesh.m * self.mesh.n, 1, 4, 3)
        ) - self.cpoints.reshape((1, self.mesh.m * self.mesh.n, 1, 3))

        vel = biot_savart(r)
        nv = self.normals.reshape((self.mesh.m * self.mesh.n, 3))
        self.aic_wing = np.einsum("ijk,jk->ji", vel, nv)

    def _calculate_wake_wing_influence_matrix(self):
        """
        Calculate influence matrix (steady wake contribution).

        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Wake contribution to the influence matrix.
        """
        mn = self.mesh.m * self.mesh.n
        self.aic_wake = np.zeros((mn, mn))
        r = self.wake[:, np.newaxis, :, :] - self.cpoints.reshape((1, mn, 1, 3))
        vel = biot_savart(r)
        nv = self.normals.reshape((mn, 3))
        self.aic_wake[:, -self.mesh.n :] = np.einsum("ijk,jk->ji", vel, nv)

    def _calculate_influence_matrix(self):
        """
        Creates
        -------
        aic : np.ndarray, shape (m * n, m * n)
            Influence matrix, including wing and wake contributions.
        """
        self._calculate_wing_influence_matrix()
        self._calculate_wake_wing_influence_matrix()
        self.aic = self.aic_wing + self.aic_wake

    def _calculate_rhs(self):
        """
        Returns
        -------
        rhs : np.ndarray, shape (m * n, )
            RHS vector.
        """
        u = self.flight_conditions.ui * np.array(
            [np.cos(self.flight_conditions.aoa), 0, np.sin(self.flight_conditions.aoa)]
        )
        self.rhs = -np.dot(self.normals.reshape(self.mesh.m * self.mesh.n, -1), u)

    def _solve_net_panel_circulation_distribution(self):
        """
        Calculate panel net circulation by solving the linear equation:
        AIC * circulation = RHS

        Creates
        -------
        net_circulation : np.ndarray, shape (m, n)
            Array containing net circulation for each panel.
        """
        g = np.linalg.solve(self.aic, self.rhs).reshape(self.mesh.m, self.mesh.n)

        self.net_circulation = np.empty_like(g)
        self.net_circulation[0, :] = g[0, :]
        self.net_circulation[1:, :] = g[1:, :] - g[:-1, :]

    def _calculate_aero_distributions_from_circulation(self):
        m, n = self.mesh.m, self.mesh.n
        rho, ui = (
            self.flight_conditions.rho,
            self.flight_conditions.ui,
        )
        bp = self.wing.planform_wingspan
        dL = self.net_circulation * rho * ui * bp / n
        dp = dL / self.panel_surfaces
        cl = dp / (0.5 * rho * ui ** 2)
        cl_wing = dL.sum() / (0.5 * rho * ui ** 2 * self.panel_surfaces.sum())
        cl_span = cl.sum(axis=0) / m
        return SimulationResults(dL=dL, dp=dp, cl=cl, cl_wing=cl_wing, cl_span=cl_span)

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

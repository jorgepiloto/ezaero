""" Holds LiftingSurface class definition """

import numpy as np
import plotly.graph_objects as go


class LiftingSurface:
    """ Simulates a trapezoidal lifting surface """

    def __init__(
        self,
        r_offset: list = [0, 0, 0],
        root_chord: float = 1,
        tip_chord: float = 1,
        planform_wingspan: float = 4,
        sweep_angle: float = 0,
        dihedral_angle: float = 0,
        name="wing",
    ):

        """Initializes a LiftingSurface instance

        Attributes
        ----------
        r_offset: float
            Position of the root's leading edge.
        root_chord : float
            Chord at root of the wing.
        tip_chord : float
            Chord at tip of the wing.
        planform_wingspan : float
            Wingspan of the planform.
        sweep_angle : float
            Sweep angle of the 1/4 chord line, expressed in radians.
        dihedral_angle : float
            Dihedral angle, expressed in radians.
        """

        # Assign attributes
        self.r_offset = r_offset
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.planform_wingspan = planform_wingspan
        self.sweep_angle = sweep_angle
        self.dihedral_angle = dihedral_angle
        self.name = name

    def __repr__(self):
        """ Returns a human readable representation of the object """
        return f"LiftingSurface = {self.name}"

    @property
    def LE_root(self):
        """ Returns position for the root leading edge """
        return self.limits[0, :]

    @property
    def LE_tip(self):
        """ Returns position for the tip leading edge """
        return self.limits[1, :]

    @property
    def TE_root(self):
        """ Returns position for the root trailing edge """
        return self.limits[3, :]

    @property
    def TE_tip(self):
        """ Returns position for the tip trailing edge """
        return self.limits[2, :]

    @property
    def limits(self):
        """ Returns surface vertex positions """

        # Compute surface vertex
        LE_root = self.r_offset
        TE_root = LE_root + np.array([self.root_chord, 0, 0])
        LE_tip = LE_root + np.array(
            [
                self.planform_wingspan / 2 * np.tan(self.sweep_angle),
                self.planform_wingspan / 2,
                self.planform_wingspan / 2 * np.tan(self.dihedral_angle),
            ]
        )
        TE_tip = LE_tip + np.array([self.tip_chord, 0, 0])

        return np.stack((LE_root, LE_tip, TE_tip, TE_root))

    def plot(self, fig=None, color="lightgray", label=None):
        """Returns a graphical representation for the object

        Parameters
        ----------
        fig: ~plotly.graph_objects.figure
            A canvas for drawing the surface
        color: string
            The color for the surface
        label: string
            The name to be placed in figure legend

        """

        # Check if figure available
        if not fig:
            fig = go.Figure()
            fig.update_layout(scene_aspectmode="data")

        # Compute limits coordinates
        X, Y, Z = [self.limits[:, i] for i in range(3)]

        # Build panels for a better configuration

        # Draw both sides of the wing
        for k in [-1, 1]:
            surface_trace = go.Mesh3d(
                x=X,
                y=k * Y,
                z=Z,
                color=color,
                name=label,
                showlegend=(True if (label and k < 0) else False),
            )

            surface_frame = go.Scatter3d(
                x=X,
                y=k * Y,
                z=Z,
                line=dict(color="black", width=2),
                marker=dict(color="black", size=2),
                showlegend=False,
            )

            # Add traces to figure
            fig.add_trace(surface_frame)
            fig.add_trace(surface_trace)

        return fig

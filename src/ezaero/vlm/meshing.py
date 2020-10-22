""" Holds lifting surface meshing routines """

import numpy as np

from ezaero.vlm.utils import get_chord_at_section, get_quarter_chord_x


def _build_panel(lifting_surface, i, j, m, n):
    """Returns panel boundaries and collocation point

    Parameters
    ----------
    lifting_surface: LiftingSurface
        The surface to be meshed
    m : int
        Number of chordwise panels.
    n : int
        Number of spanwise panels.
    i : int
        Index of chordwise panels.
    j : int
        Index of spanwise panels.

    Returns
    -------
    panel: np.array
        A 4x3 matrix hosting panel boundaries coordinates
    cp: np.array
        A 1x3 matrix hosting panel collocation coordinates

    """

    dy = lifting_surface.planform_wingspan / n
    y_A = -lifting_surface.planform_wingspan / 2 + j * dy
    y_B = y_A + dy
    y_C, y_D = y_A, y_B
    y_pc = y_A + dy / 2

    # chord law evaluation
    c_AC, c_BD, c_pc = [
        get_chord_at_section(
            y,
            root_chord=lifting_surface.root_chord,
            tip_chord=lifting_surface.tip_chord,
            span=lifting_surface.planform_wingspan,
        )
        for y in (y_A, y_B, y_pc)
    ]

    # division of the chord in m equal panels
    dx_AC, dx_BD, dx_pc = [c / m for c in (c_AC, c_BD, c_pc)]

    # r,s,q are the X coordinates of the quarter chord line at spanwise
    # locations: y_A, y_B and y_pc respectively
    r, s, q = [
        get_quarter_chord_x(
            y, cr=lifting_surface.root_chord, sweep=lifting_surface.sweep_angle
        )
        for y in (y_A, y_B, y_pc)
    ]

    # TODO: Original code introduces a shift in x-direction
    """
    x_A = (r - c_AC / 4) + i * dx_AC
    x_B = (s - c_BD / 4) + i * dx_BD
    x_C = x_A + dx_AC
    x_D = x_B + dx_BD
    x_pc = (q - c_pc / 4) + (i + 3 / 4) * dx_pc
    """
    x_A = (r) + i * dx_AC
    x_B = (s) + i * dx_BD
    x_C = x_A + dx_AC
    x_D = x_B + dx_BD
    x_pc = (q) + (i + 3 / 4) * dx_pc

    x = np.array([x_A, x_B, x_D, x_C]) - lifting_surface.root_chord / 4
    y = np.array([y_A, y_B, y_D, y_C])
    z = np.tan(lifting_surface.dihedral_angle) * np.abs(y)
    panel = np.stack((x, y, z), axis=-1)

    z_pc = np.tan(lifting_surface.dihedral_angle) * np.abs(y_pc)
    pc = np.array([x_pc, y_pc, z_pc])

    return panel, pc


def mesh_surface(lifting_surface, m=4, n=16):
    """Returns a collection of panels and their control points

    Parameters
    ----------
    lifting_surface: LiftingSurface
        The surface to be meshed
    m : int
        Number of chordwise panels.
    n : int
        Number of spanwise panels.

    Returns
    -------
    self.wing_panels : np.ndarray, shape (m, n, 4, 3)
        Array containing the (x,y,z) coordinates of all wing panel vertices.
    self.cpoints : np.ndarray, shape (m, n, 3)
        Array containing the (x,y,z) coordinates of all collocation points.

    """

    # Allocate output varibales
    wing_panels = np.empty((m, n, 4, 3))
    cpoints = np.empty((m, n, 3))

    # Compute boundaries for each panel
    for i in range(m):
        for j in range(n):

            # Append new panel location to output variables
            wing_panels[i, j], cpoints[i, j] = _build_panel(lifting_surface, i, j, m, n)

    return wing_panels, cpoints

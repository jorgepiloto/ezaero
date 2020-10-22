""" Holds auxiliary functions and routines """

import numpy as np

def get_quarter_chord_x(y, cr, sweep):
    """ Solves for slope of the quarter chord line

    Parameters
    ----------
    y: float
        Spanwise distance
    cr: float
        Chord at the root
    sweep: float
        Sweep angle

    Returns
    -------
    p: float
        Slope of the quarter chord line

    """

    p = np.tan(sweep)
    return cr / 4 + p * abs(y)


def get_chord_at_section(y, root_chord, tip_chord, span):
    """ Solves for chord at particular spanwise distance
    
    Parameters
    ----------
    y: float
        Spanwise distance
    root_chord: float
        Chord value at the root
    tip_chord: float
        Chord value at the tip
    span: float
        Span of the wing

    Returns
    -------
    c: float
        Chord at evaluated section

    """

    c = root_chord + (tip_chord - root_chord) * abs(2 * y / span)
    return c

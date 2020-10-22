""" Test example """

import numpy as np
import plotly.graph_objects as go

from ezaero.vlm.lifting_surface import LiftingSurface
from ezaero.vlm.steady import VLM_solver

# Parts definition
wing_root = LiftingSurface(
    r_offset=np.array([0, 0, 0]),
    root_chord=1,
    tip_chord=0.25,
    planform_wingspan=4,
    sweep_angle=30 * np.pi / 180,
    dihedral_angle=10 * np.pi / 180,
    name="Inner wing",
)

wing_tip = LiftingSurface(
    r_offset=wing_root.LE_tip,
    root_chord=0.25,
    tip_chord=0.05,
    planform_wingspan=1,
    sweep_angle=30 * np.pi / 180,
    dihedral_angle=45 * np.pi / 180,
    name="Outer wing",
)

model = [wing_root, wing_tip]


sim = VLM_solver(model)
sim.run()
print([len(sim.parts_data[part]) for part in sim.part_list])

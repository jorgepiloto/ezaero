""" Test example """

import numpy as np
import plotly.graph_objects as go

from ezaero.vlm.lifting_surface import LiftingSurface
from ezaero.vlm.steady import VLM_solver

# Parts definition
wing_root = LiftingSurface(
    r_offset=np.array([0, 0, 0]),
    root_chord=1,
    tip_chord=1 - np.tan(30 * np.pi / 180) * (1.6 / 2),
    planform_wingspan=1.6,
    sweep_angle=30 * np.pi / 180,
    dihedral_angle=0 * np.pi / 180,
    name="Inner wing",
)

wing_tip = LiftingSurface(
    r_offset=wing_root.LE_tip,
    root_chord=1 - np.tan(30 * np.pi / 180) * (1.6 / 2),
    tip_chord=0.25,
    planform_wingspan=2.15,
    sweep_angle=30 * np.pi / 180,
    dihedral_angle=0 * np.pi / 180,
    name="Outer wing",
)

model = [wing_root, wing_tip]

cl_obtained = []

aoa = np.linspace(0, 12.0, 13)
for a in aoa:
    sim = VLM_solver(model, alpha=a * np.pi / 180, Uinf=7.0)
    sim.run()
    cl_obtained.append(sum(sim.all_cl_wing))

fig = go.Figure(go.Scatter(x=aoa, y=cl_obtained))
fig.show()



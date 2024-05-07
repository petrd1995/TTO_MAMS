import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import Materials
import Plots
import Truss
import Optimization
from HeuristicTSP import custom_tsp

import random
import numpy as np
import os
from time import time

np.random.default_rng(0)
random.seed(0)

time1 = time()
rect = Truss.Truss()
plotter = Plots.TrussPlotter(rect)
rect.create_rectangular_grid(15, 15, 0.100, 0.100)
rect.create_bars(bar_area=1, 
bar_material=Materials.Inconel718mm, 
bar_max_length=0.013)
rect.assign_temperatures(r"source1_square.txt")

selection = 25
rect.select_hottest_nodes(selection)

def set_forces():
    node.bc_force_label = True
    node.forces = np.array([0,-100,0])

def set_bcs():
    node.bc_clamp_label = True
    node.boundary_conditions = np.array([1,1,1])

for node in rect.nodes:
    node.boundary_conditions = np.array([0,0,1])
    if node.x <= 0.100/15:
        set_bcs()
    if node.x == 0.100 and node.y == 0:
        set_forces()
    if node.x == 0.100 and node.y == 0.100/14:
        set_forces()
    if node.x == 13*0.100/14 and node.y == 0:
        set_forces()

rect.InitiateGraph(selected_nodes=rect.hottest_nodes)

start_node = 0
end_node = 210
required_nodes_list = [node.num for node in rect.hottest_nodes]+[end_node]

rect.node_num_path = custom_tsp(rect.G_ground_structure, required_nodes_list, start_node, end_node)


cool = [node for node in rect.node_num_path]
rect.bar_path_from_node_num_path(node_num_path=rect.node_num_path)

time2 = time()
r_inner = 0.005
w = 0.001
A_inner = np.pi*((r_inner + w)**2 - r_inner**2)

total_time = time2 - time1
# print time in seconds
print("Time in seconds:", total_time)


plotter.plot_truss(
    plot_nodes=True,
    node_edgecolors="k",
    plot_bars=True,
    view="2D",
    node_size=10,
    cooling_path=cool,
)

# opt = Optimization.Optimization(rect, min_radius=0.1e-3, max_radius=5e-3,vol_frac=0.1)
# res = opt.sdp(volume_fraction=0.1, path=rect.bar_num_path, A_inner=A_inner)

# plotter.plot_truss(plot_nodes=True, node_edgecolors='k',plot_bars=True, view='2D',node_size=10)

# save the result so that it can be loaded for truss topology optimization later
rect.save(rf"test_2D_Heu")

pass

import sys
import os
import time
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import Materials
import Plots
import Truss
from HeuristicTSP import custom_tsp

import numpy as np
import os

cube = Truss.Truss()

cube.create_cuboid_grid(10, 10, 10, 0.100, 0.100, 0.100)
cube.create_bars(bar_area=0.1, bar_material=Materials.Inconel718mm, bar_max_length=18e-3)
cube.assign_temperatures(r"cube_meters.txt") 

plotter = Plots.TrussPlotter(cube)

selection = 5
vol_frac = 0.1
path_ratio = 0.1

cube.select_hottest_nodes(selection)
hot5 = cube.hottest_nodes

cube.InitiateGraph(selected_nodes=cube.hottest_nodes)

start_node = 0
end_node = 10
required_nodes_list = [node.num for node in cube.hottest_nodes]+[end_node]

start_time = time.time()
cube.node_num_path = custom_tsp(cube.G_ground_structure, required_nodes_list, start_node, end_node)
end_time = time.time()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"The function took {int(hours)}h {int(minutes)}m {seconds:.2f}s to execute.")

cube.bar_path_from_node_path()

for node in cube.nodes:
    if node.temperature > 340:
        node.bc_force_label = True
        node.forces = np.array([0,-100,0])
    elif node.temperature  <= 294:
        node.bc_clamp_label = True
        node.boundary_conditions = np.array([1,1,1])

plotter.plot_truss(
    plot_bars=True,
    plot_nodes=False,
    plot_bc_clamp_nodes=True,
    plot_bc_force_nodes=True,
    plot_bc_heat_nodes=False,
    node_edgecolors='k',
    node_size=80,
    figsize=(5,5),
    dpi=300,
    plot_cooling_path_res=True,
    optimized_areas_color='#000000',
    cooling_path_color='#d70929'
)

plotter.plot_truss(plot_nodes=False, plot_bars=True, view='3D')

cube.save(fr"test_3D_Heu")

pass

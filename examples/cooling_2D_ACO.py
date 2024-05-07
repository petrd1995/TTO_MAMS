import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import Materials
import Plots
import Truss

import random
import numpy as np

# np.random.default_rng(0)
# random.seed(0)

rect = Truss.Truss()

rect.create_rectangular_grid(15, 15, 0.100, 0.100)
rect.create_bars(bar_area=1, bar_material=Materials.Inconel718mm, bar_max_length=0.013)
rect.assign_temperatures(r"source1_square.txt")
rect.select_hottest_nodes(10)
for node in rect.nodes:
    if node.x < 1:
        node.bc_clamp_label = True
    if node.x == 150 and node.y == 0:
        node.bc_force_label = True
    if node.x == 150 and node.y == 100/9:
        node.bc_force_label = True
    if node.x == 13*150/14 and node.y == 0:
        node.bc_force_label = True
rect.InitiateGraph(selected_nodes=rect.hottest_nodes)
plotter = Plots.TrussPlotter(rect)

best_path = rect.tsp_aco(graph=rect.G_selected,
        main_graph=rect.G_ground_structure, 
        n_ants=50, 
        n_iterations=50, # 200
        alpha=1, 
        beta=2, 
        prescribed_start = 0,
        prescribed_end = 210,
        evaporation_rate=0.1,
        loop=False,
        plot_progress=False)

rect.colony.restructure_by_rank()

plotter.plot_truss( 
plot_bars=True, 
cooling_path=rect.colony.combined_paths['combined_path_1']['path'], 
plot_node_numbers=True, 
alpha_bars=0.3,
view="2D",
selected_nodes=rect.hottest_nodes, 
plot_temperatures=True, 
plot_nodes=True, 
plot_other_nodes=True)

plotter.plot_truss(
plot_bars=False,
cooling_path=rect.best_aco_path,
plot_node_numbers=False,
alpha_bars=0.3,
view="2D",
selected_nodes=rect.hottest_nodes,
plot_temperatures=False,
plot_nodes=False,
plot_other_nodes=False)

# save the result so that it can be loaded for truss topology optimization later
rect.save(rf"test_2D_ACO")

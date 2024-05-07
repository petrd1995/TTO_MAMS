import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import Materials
import Plots
import Truss

import os

cube = Truss.Truss()

cube.create_cuboid_grid(10, 10, 10, 0.100, 0.100, 0.100)
cube.create_bars(bar_area=0.1, bar_material=Materials.Inconel718, bar_max_length=18e-3)

cube.assign_temperatures(r"cube_meters.txt")

plotter = Plots.TrussPlotter(cube)

selection = 5
cube.select_hottest_nodes(selection)
cube.InitiateGraph(selected_nodes=cube.hottest_nodes)

prescribed_start = 0
prescribed_end = 10

best_path = cube.tsp_aco(graph=cube.G_selected,
        main_graph=cube.G_ground_structure, 
        n_ants=40, 
        n_iterations=50, 
        alpha=1, 
        beta=2, 
        prescribed_start=prescribed_start, # 0
        prescribed_end=prescribed_end, # 10
        evaporation_rate=0.1,
        loop=False,
        plot_progress=False)

cube.colony.restructure_by_rank()

plotter.plot_truss(
    plot_bars=True,
    cooling_path=cube.colony.combined_paths["combined_path_1"]["path"],
    plot_node_numbers=True,
    alpha_bars=0.3,
    view="3D",
    selected_nodes=cube.hottest_nodes,
    plot_temperatures=True,
    plot_nodes=True,
    plot_other_nodes=True,
)

plotter.plot_truss(
    plot_bars=False,
    cooling_path=cube.best_aco_path,
    plot_node_numbers=False,
    alpha_bars=0.3,
    view="3D",
    selected_nodes=cube.hottest_nodes,
    plot_temperatures=False,
    plot_nodes=False,
    plot_other_nodes=False,
)

cube.save(filename=rf'test_3D_ACO')


pass

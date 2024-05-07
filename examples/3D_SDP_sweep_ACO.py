import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


import Materials
import Plots
import Truss
import Optimization

import numpy as np
from itertools import product

cube = Truss.Truss()

selections = [0, 1, 2]
sels = [5, 10, 15]
volume_fractions = [0.1,0.2,0.3,0.4,0.5]

paths = ["test_3D_ACO", "path/to/15%path", "path/to/20%path", "path/to/25%path"]


for en,(selection,volume_fraction) in enumerate(product(selections,volume_fractions)):

    print(f"selection: {sels[selection]}, volume fraction: {volume_fraction}")
    cube = Truss.Truss()
    cube.load(fr'{paths[selection]}')
    plotter = Plots.TrussPlotter(cube)
    for bar in cube.bars:
        bar.area=1
        bar.material=Materials.Inconel718mm()

    cube.hottest_nodes_num = [node.num for node in cube.hottest_nodes]

    for node in cube.nodes:
        if node.num in cube.hottest_nodes_num and node.y<0.06:
            node.bc_force_label = True
            node.forces = np.array([0,-100,0])
        elif node.num in [494,495,594,595]:
            node.bc_force_label = True
            node.forces = np.array([0,-100,0])
        elif node.temperature < 300:
            node.bc_clamp_label = True
            node.boundary_conditions = np.array([1,1,1])

    for node in cube.nodes:
        node.x = node.x
        node.y = node.y
        node.z = node.z
    for bar in cube.bars:
        bar.length = bar.length

    cool = [node for node in cube.best_aco_path]
    cube.node_path_from_node_nums(path = cube.best_aco_path)
    find_invalid_segments = cube.find_invalid_segments(node_path=cube.best_aco_path)
    cube.best_aco_path = cube.find_and_insert_shortest_paths(cube.best_aco_path, find_invalid_segments)
    cube.bar_path_from_node_num_path(cube.best_aco_path)

    r_inner = 0.005
    r_outer = 0.006
    w = r_outer - r_inner

    A_inner = np.pi*((r_inner + w)**2 - r_inner**2)
    A_max =np.pi*r_outer**2

    opt = Optimization.Optimization(cube, min_radius=0.1e-3, max_radius=5e-3,vol_frac=0.1)
    res = opt.sdp(volume_fraction=volume_fraction, path=cube.bar_num_path, A_inner=A_inner, A_max=A_max)

    plotter.plot_truss(
        plot_bars=True,
        # plot_nodes=True,
        # plot_bc_clamp_nodes=True,
        # plot_bc_force_nodes=True,
        # plot_bc_heat_nodes=False,
        # node_edgecolors='k',
        # node_size=80,
        figsize=(5,5),
        dpi=300,
        view="3D",
    )    
    plotter.plot_truss(
        plot_bars=True,
        # plot_nodes=True,
        # plot_bc_clamp_nodes=True,
        # plot_bc_force_nodes=True,
        # plot_bc_heat_nodes=False,
        # node_edgecolors='k',
        # node_size=80,
        figsize=(5,5),
        dpi=300,
        view="3D",
        plot_cooling_path_res=True,
        optimized_areas_color='#000000',
        cooling_path_color='#d70929'
    )
    plotter.plot_truss(
        plot_bars=True,
        # plot_nodes=True,
        # plot_bc_clamp_nodes=True,
        # plot_bc_force_nodes=True,
        # plot_bc_heat_nodes=False,
        # node_edgecolors='k',
        # node_size=80,
        figsize=(5,5),
        dpi=300,
        view="3D",
        plot_cooling_path_res=True,
        optimized_areas_color='#000000',
        cooling_path_color='#0072bb'
    )
    plotter.plot_truss(
        plot_bars=False, 
        cooling_path=cube.best_aco_path, 
        plot_node_numbers=False, 
        alpha_bars=0.3,
        node_edgecolors='k',
        node_size=80,
        figsize=(5,5),
        dpi=300,
        view="3D",
        selected_nodes=cube.hottest_nodes, 
        plot_temperatures=False, 
        plot_nodes=False, 
        plot_other_nodes=False,
        )

    h = cube.cooling(r_inner,w,0.01,293)
    cube.Qdot_sum = h[0]
    cube.temp_along_path = h[1]
    cube.Qdots = h[2]
    cube.opt = opt
    # save the results
    cube.save(rf"save/results/here")

pass

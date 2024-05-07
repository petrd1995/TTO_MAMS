import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import Materials
import Plots
import Truss
import Optimization

import random
import numpy as np

np.random.default_rng(0)
random.seed(0)

selection = 10

rect = Truss.Truss()
plotter = Plots.TrussPlotter(rect)
rect.create_rectangular_grid(15, 15, 0.100, 0.100)
rect.create_bars(bar_area=1, 
bar_material=Materials.Inconel718mm, 
bar_max_length=0.013)
rect.assign_temperatures(r"C:\Users\david\OneDrive - České vysoké učení technické v Praze\0 PhD\0 Články\3 Aktivní chlazení\4 Výpočty (zaloha)\source1_square.txt")

selections = [10,15,20,25]
volume_fractions = [0.1,0.2,0.3,0.4,0.5]

# paths = [
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 40, 41, 42, 57, 56, 72, 73, 58, 43, 28, 13, 14, 29, 44, 59, 74, 89, 88, 87, 86, 85, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210],
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 24, 39, 40, 55, 56, 57, 42, 41, 26, 27, 12, 13, 28, 43, 58, 59, 74, 73, 72, 71, 70, 86, 87, 88, 103, 104, 118, 117, 116, 115, 114, 113, 127, 141, 140, 154, 168, 182, 196, 210],
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 23, 38, 39, 54, 55, 40, 25, 10, 11, 12, 27, 26, 41, 42, 57, 56, 71, 72, 73, 58, 43, 28, 13, 14, 29, 44, 59, 74, 89, 104, 103, 88, 87, 86, 101, 102, 117, 118, 119, 133, 132, 116, 100, 85, 70, 69, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210],
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 23, 38, 53, 54, 39, 24, 9, 10, 25, 40, 55, 56, 57, 42, 41, 26, 27, 12, 13, 28, 43, 58, 59, 74, 73, 72, 71, 70, 69, 68, 84, 85, 100, 101, 102, 87, 88, 103, 104, 119, 118, 117, 116, 115, 131, 132, 133, 134, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 154, 168, 182, 196, 210]
# ]

paths_to_cooling_paths = [
    'test_2D_Heu',
    # 'second_path',
    # 'third_path',
    # 'fourth_path'
]


for volume_fraction in volume_fractions:

    for en, example in enumerate(selections):
        rect = Truss.Truss()
        plotter = Plots.TrussPlotter(rect)

        rect.load(f"{paths_to_cooling_paths[en]}")

        rect.select_hottest_nodes(selections[en])
        rect.InitiateGraph(selected_nodes=rect.hottest_nodes)

        start_node = 0
        end_node = 210
        required_nodes_list = [node.num for node in rect.hottest_nodes]+[end_node]

        rect.bar_path_from_node_num_path(node_num_path=rect.node_num_path)

        for bar in rect.bars:
            if not bar.num in rect.bar_num_path:
                bar.pipe = False

        r_inner = 0.005
        r_outer = 0.006
        w = r_outer - r_inner

        A_inner = np.pi*((r_inner + w)**2 - r_inner**2)
        A_max =np.pi*r_outer**2

        opt = Optimization.Optimization(rect, min_radius=0.1e-3, max_radius=5e-3,vol_frac=0.1)
        res = opt.sdp(volume_fraction=volume_fraction, path=rect.bar_num_path, A_inner=A_inner, A_max=A_max)

        plotter.plot_truss(
            plot_bars=True,
            plot_nodes=True,
            plot_bc_clamp_nodes=True,
            plot_bc_force_nodes=True,
            plot_bc_heat_nodes=False,
            node_edgecolors='k',
            node_size=80,
            figsize=(5,5),
            dpi=300,
            view="2D",
        )    
        plotter.plot_truss(
            plot_bars=True,
            plot_nodes=True,
            plot_bc_clamp_nodes=True,
            plot_bc_force_nodes=True,
            plot_bc_heat_nodes=False,
            node_edgecolors='k',
            node_size=80,
            figsize=(5,5),
            dpi=300,
            view="2D",
            plot_cooling_path_res=True,
            optimized_areas_color='#000000',
            cooling_path_color='#d70929'
        )
        plotter.plot_truss(
            plot_bars=True,
            plot_nodes=True,
            plot_bc_clamp_nodes=True,
            plot_bc_force_nodes=True,
            plot_bc_heat_nodes=False,
            node_edgecolors='k',
            node_size=80,
            figsize=(5,5),
            dpi=300,
            view="2D",
            plot_cooling_path_res=True,
            optimized_areas_color='#000000',
            cooling_path_color='#0072bb'
        )
        plotter.plot_truss(
            plot_bars=False, 
            cooling_path=rect.node_num_path, 
            plot_node_numbers=False, 
            alpha_bars=0.3,
            node_edgecolors='k',
            node_size=80,
            figsize=(5,5),
            dpi=300,
            view="2D",
            selected_nodes=rect.hottest_nodes, 
            plot_temperatures=False, 
            plot_nodes=False, 
            plot_other_nodes=False,
            )

        h = rect.cooling(r_inner,w,0.01,293)
        rect.Qdot_sum = h[0]
        rect.temp_along_path = h[1]
        rect.Qdots = h[2]
        rect.opt = opt
        # save the results
        rect.save(rf"save/results/here")
        print(h[0])


pass

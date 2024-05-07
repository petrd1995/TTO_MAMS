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

selection = [10,15,20,25]

paths = [
'test_2D_ACO',
'path/to/15%path',
'path/to/20%path',
'path/to/25%path'
]

def set_forces():
    node.bc_force_label = True
    node.forces = np.array([0,-100,0])

def set_bcs():
    node.bc_clamp_label = True
    node.boundary_conditions = np.array([1,1,1])

for volume_fraction in [0.1,0.2,0.3,0.4,0.5]:
    for en,example in enumerate(selection):
        rect = Truss.Truss()
        rect.load(fr'{paths[en]}')
        for bar in rect.bars:
            bar.area=1
            bar.material=Materials.Inconel718mm()

        for node in rect.nodes:
            node.bc_clamp_label = False
            node.boundary_conditions = np.array([0,0,1])
            if node.x <= 0.100/15:
                set_bcs()
            if node.x == 0.100 and node.y == 0:
                set_forces()
            if node.x == 0.100 and node.y == 0.100/14:
                set_forces()
            if node.x == 13*0.100/14 and node.y == 0:
                set_forces()

        for node in rect.nodes:
            node.x = node.x
            node.y = node.y
        for bar in rect.bars:
            bar.length = bar.length

        plotter = Plots.TrussPlotter(rect)

        rect.bar_path_from_node_num_path(rect.best_aco_path)

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
            cooling_path=rect.best_aco_path, 
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
        rect.save(fr"save/results/here")
        rect = None

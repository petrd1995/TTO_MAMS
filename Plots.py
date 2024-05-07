import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
import Node
import os

# nord_seg = (colors.ListedColormap(
#     ['#A3BE8C', '#8FBCBB', '#B48EAD', '#D08770', '#BF616A']))
# nord2_seg = (colors.ListedColormap(
#     ['#EBCB8B', '#A3BE8C', '#B48EAD', '#2E3440']))
# nord2_seg = (colors.ListedColormap(
#     ['#2E3440', '#3B4252', '#4C566A', '#EBCB8B']))
# nord_seg = (colors.ListedColormap(
#     ['#B48EAD', '#EBCB8B', '#D8DEE9', '#D08770', '#BF616A']))
# nord2_seg = (colors.ListedColormap(
#     ['#8FBCBB', '#88C0D0', '#81A1C1', '#5E81AC']))
ctu_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("",['#0072bb', '#6bb1e3', '#9d9d9c', '#000000', ])

class TrussPlotter:
    def __init__(self, truss):
        self.truss = truss

    def get_thickness(self, optimized_areas, norm, default_factor=2.99, default_min=0.01, scale_factor=1.0, enlarge=False, threshold=None, thickness_factor=1.0):
        thickness = optimized_areas if optimized_areas is not None else [bar.area for bar in self.truss.bars]

        thickness = np.array(thickness)

        if threshold is not None:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"
            max_thickness = np.max(thickness)
            thickness = np.where(thickness / max_thickness >= threshold, thickness, 0)

        if enlarge:
            thickness = thickness / np.max(thickness)
            thickness = thickness * scale_factor + default_min
        elif norm:
            thickness = (thickness - np.min(thickness)) / (np.max(thickness) - np.min(thickness))


        return thickness / np.max(thickness) * thickness_factor if optimized_areas is None else thickness * thickness_factor


    def plot_bars_function(self, optimized_areas, norm, view, alpha_bars, zorder=5, default_factor=2.99, default_min=0.01, scale_factor=1.0, enlarge=False,
                           plot_bar_thickness_colors=False, threshold=None, cmap_type='viridis', thickness_factor=1.0, displacements=False, 
                           deformation_scale=10, plot_original_areas=False, original_areas_alpha=0.3, original_areas_scale_factor=0.5,
                           original_areas_color='#000000', optimized_areas_color='#0072bb', plot_cooling_path_res=False, cooling_path_color='#f9b002'):
        
        # colors_list = ['#0072bb', '#000000', '#6bb1e3', '#9d9d9c', '#d70929', '#f9b002','#a0b002', '#e9520e', '#005864', '#a20e3e','#01a9ac']

        default_bar_color = '#000000'
        thickness = self.get_thickness(optimized_areas, norm, default_factor=default_factor, default_min=default_min,
                                    scale_factor=scale_factor, enlarge=enlarge, threshold=threshold, thickness_factor=thickness_factor)

        if plot_original_areas:
            original_areas_thickness = [bar.original_area*original_areas_scale_factor for bar in self.truss.bars]

        segments = []
        deformed_segments = []  
        linewidths = []

        if plot_bar_thickness_colors:
            cmap = cm.get_cmap(cmap_type)
            default_bar_color = cmap(thickness)
            linewidths = [1] * len(self.truss.bars)

        if plot_cooling_path_res:
            optimized_areas_color = [optimized_areas_color] * len(self.truss.bars)
            for en,bar in enumerate(self.truss.bars):
                if bar.pipe:
                    optimized_areas_color[en] = cooling_path_color

        for i, bar in enumerate(self.truss.bars):
            if view == '3D':
                segments.append([(bar.node1.x, bar.node1.y, bar.node1.z),
                                (bar.node2.x, bar.node2.y, bar.node2.z)])
            elif view == '2D':
                segments.append([(bar.node1.x, bar.node1.y),
                                (bar.node2.x, bar.node2.y)])
            if not plot_bar_thickness_colors:
                linewidths.append(thickness[i])

            if displacements:
                node1_disp = np.array(bar.node1.displacements) * deformation_scale
                node2_disp = np.array(bar.node2.displacements) * deformation_scale
                if view == '3D':
                    deformed_segments.append([(bar.node1.x + node1_disp[0], bar.node1.y + node1_disp[1], bar.node1.z + node1_disp[2]),
                                              (bar.node2.x + node2_disp[0], bar.node2.y + node2_disp[1], bar.node2.z + node2_disp[2])])
                elif view == '2D':
                    deformed_segments.append([(bar.node1.x + node1_disp[0], bar.node1.y + node1_disp[1]),
                                              (bar.node2.x + node2_disp[0], bar.node2.y + node2_disp[1])])

        if displacements is not None:
            deformed_color = 'red' 
            if view == '3D':
                deformed_lc = Line3DCollection(deformed_segments, colors=deformed_color, linewidths=linewidths, alpha=alpha_bars, zorder=zorder)
                self.ax.add_collection(deformed_lc)
            else:
                deformed_lc = LineCollection(deformed_segments, colors=deformed_color, linewidths=linewidths, alpha=alpha_bars, zorder=zorder)
                self.ax.add_collection(deformed_lc)

        if view == '3D':
            lc = Line3DCollection(segments, colors=optimized_areas_color, linewidths=linewidths, alpha=alpha_bars, zorder=zorder)
            if plot_original_areas:
                original_areas_lc = Line3DCollection(segments, colors=original_areas_color, linewidths=original_areas_thickness, alpha=original_areas_alpha, zorder=zorder)
                self.ax.add_collection(original_areas_lc)
            self.ax.add_collection(lc)
            all_x = [bar.node1.x for bar in self.truss.bars] + [bar.node2.x for bar in self.truss.bars]
            all_y = [bar.node1.y for bar in self.truss.bars] + [bar.node2.y for bar in self.truss.bars]
            all_z = [bar.node1.z for bar in self.truss.bars] + [bar.node2.z for bar in self.truss.bars]

            buffer = 1  
            self.ax.set_xlim([min(all_x) - buffer, max(all_x) + buffer])
            self.ax.set_ylim([min(all_y) - buffer, max(all_y) + buffer])
            self.ax.set_zlim([min(all_z) - buffer, max(all_z) + buffer])
        else:            
            if plot_original_areas:
                original_areas_lc = Line3DCollection(segments, colors=original_areas_color, linewidths=original_areas_thickness, alpha=original_areas_alpha, zorder=zorder)
                self.ax.add_collection(original_areas_lc)
            lc = LineCollection(segments, colors=optimized_areas_color, linewidths=linewidths, alpha=alpha_bars, zorder=zorder)
            self.ax.add_collection(lc)

    def plot_nodes_function(self, plot_temperatures, norm_temperatures, cmap, view, plot_node_numbers,
                            plot_vicinity_volumes=False, plot_vicinity_areas=False, vicinity_threshold=None, alpha_nodes=1, node_edgecolors='none',
                            plot_bc_clamp_nodes=True, plot_bc_force_nodes=True, plot_bc_heat_nodes=True,
                            plot_unlabeled_nodes=True, size=30, cmap_type='viridis',plot_force_arrows=False, 
                            force_scale=1, plot_dof_arrows=False, force_arrow_color="#d70929", dof_arrow_color="#0072bb", dof_arrow_length=10):
        x, y, z, colors, markers, sizes = [], [], [], [], [], []

        texts, text_coords = [], []
        
        for i, node in enumerate(self.truss.nodes):
            if vicinity_threshold is not None and plot_vicinity_volumes:
                if node.vicinity_volume < vicinity_threshold:
                    continue
            if vicinity_threshold is not None and plot_vicinity_areas:
                if node.vicinity_area < vicinity_threshold:
                    continue

            if plot_vicinity_volumes:  
                color = cm.get_cmap(cmap_type)(node.vicinity_volume)  
                marker = 'o'
            elif plot_vicinity_areas:
                color = cm.get_cmap(cmap_type)(node.vicinity_area)
                marker = 'o'
            else:
                color, marker, size = self.get_node_attributes(node, plot_temperatures, norm_temperatures, cmap, i, size=size)

            if plot_force_arrows:
                if node.bc_force_label:  
                    self.plot_force_arrow(node, force_arrow_color, force_scale)

            if plot_dof_arrows:
                if node.bc_clamp_label:  
                    self.plot_dof_arrow(node, dof_arrow_color, arrow_length=dof_arrow_length)

            if node.bc_clamp_label and not plot_bc_clamp_nodes:
                continue
            if node.bc_force_label and not plot_bc_force_nodes:
                continue
            if node.bc_heat_label and not plot_bc_heat_nodes:
                continue
            if not any([node.bc_clamp_label, node.bc_force_label, node.bc_heat_label]) and not plot_unlabeled_nodes:
                continue

            x.append(node.x)
            y.append(node.y)
            z.append(node.z)
            colors.append(color)
            markers.append(marker)
            sizes.append(size)
            
            if plot_node_numbers:
                texts.append(node.num)
                text_coords.append((node.x, node.y, node.z))


        x, y, z = np.array(x), np.array(y), np.array(z)
        colors = np.array(colors)
        sizes = np.array(sizes)

        unique_markers = set(markers)
        
        for marker in unique_markers:
            mask = np.array([m == marker for m in markers])
            if view == '3D':
                self.ax.scatter(x[mask], y[mask], z[mask], c=colors[mask], edgecolors=node_edgecolors, marker=marker, s=sizes[mask], alpha=alpha_nodes,zorder=10)
            elif view == '2D':
                self.ax.scatter(x[mask], y[mask], c=colors[mask], edgecolors=node_edgecolors, marker=marker, s=sizes[mask], alpha=alpha_nodes,zorder=10)

        if plot_node_numbers:
            for num, (tx, ty, tz) in zip(texts, text_coords):
                if view == '3D':
                    self.ax.text(tx, ty, tz, num, color='r')
                elif view == '2D':
                    self.ax.text(tx, ty, num, color='r')

    def plot_force_arrow(self, node, color, scale):
        force_vector = np.array(node.forces) * scale
        self.ax.quiver(node.x, node.y, node.z, force_vector[0], force_vector[1], force_vector[2], color=color)

    def plot_dof_arrow(self, node, color, arrow_length=10):
        for i, constrained in enumerate(node.boundary_conditions):
            if constrained:
                direction = np.zeros(3)
                direction[i] = arrow_length
                self.ax.quiver(node.x, node.y, node.z, direction[0], direction[1], direction[2], color=color)

    @staticmethod
    def get_node_attributes(node, plot_temperatures, norm_temperatures, cmap, i, size=30):
        marker_dict = {True: 'o', False: 'o'}
        if norm_temperatures is None:
            color_dict={True: "white", False: "k"}
        else:
            color_dict = {True: cmap(norm_temperatures[i]), False: "white"}
        

        if node.bc_clamp_label:
            marker_dict[False], color_dict[False] = ">", "#0072bb" # #5e81ac
        elif node.bc_force_label:
            marker_dict[False], color_dict[False] = "s", "#d70929" # #A3BE8C
        elif node.bc_heat_label:
            marker_dict[False], color_dict[False] = "*", "#f9b002" # #EBCB8B

        return color_dict[plot_temperatures], marker_dict[plot_temperatures], size

    def get_cmap_and_norm_temperatures(self):
        temperatures = np.array([node.temperature for node in self.truss.nodes])
        if temperatures.any() is None:
            return [None]*4
        norm_temperatures = (temperatures - temperatures.min()) / (temperatures.max() - temperatures.min())
        node_to_temperature = {node.num: temp for node, temp in zip(self.truss.nodes, norm_temperatures)}
        cmap = cm.get_cmap('inferno')
        return cmap, norm_temperatures, node_to_temperature, temperatures

    def plot_cooling_path(self, cooling_path, plot_arrows, view, optimized_areas, norm):
        for i in range(len(cooling_path) - 1):
            if type(cooling_path[0]) == Node.Node:
                node1 = cooling_path[i]
                node2 = cooling_path[i+1]
            else:
                node1 = self.truss.nodes[cooling_path[i]]
                node2 = self.truss.nodes[cooling_path[i+1]]

            thickness = self.get_thickness(optimized_areas, norm)+1
            lw = max(thickness)
            if plot_arrows:
                if view == '3D':
                    self.ax.quiver(node1.x, node1.y, node1.z, node2.x-node1.x, node2.y-node1.y, node2.z-node1.z, color='#0072bb', alpha=1, zorder=10)
                elif view == '2D':
                    self.ax.quiver(node1.x, node1.y, node2.x-node1.x, node2.y-node1.y, color='#0072bb', alpha=1, scale=1, scale_units='xy', angles='xy', zorder=10)
            else:
                if view == '3D':
                    self.ax.plot([node1.x, node2.x], [node1.y, node2.y], [node1.z, node2.z], lw=lw, color='#0072bb', alpha=1, zorder=10)
                elif view == '2D':
                    self.ax.plot([node1.x, node2.x], [node1.y, node2.y], lw=lw, color='#0072bb', alpha=1, zorder=10)

    def plot_selected_nodes(self, selected_nodes_sets, plot_temperatures, plot_other_nodes, view, cmap, norm_temperatures, plot_node_numbers, node_to_temperature, set_labels=None,alpha_nodes=1, node_edgecolors='none',size=30):
        # colors_list = ['#BF616A', '#A3BE8C', '#5E81AC', '#EBCB8B', '#B48EAD']  # Extend this list if you have more node sets
        colors_list = ['#BF616A','#0072bb', '#000000', '#6bb1e3', '#9d9d9c', '#d70929', '#f9b002','#a0b002', '#e9520e', '#005864', '#a20e3e','#01a9ac']


        if not isinstance(selected_nodes_sets[0], list):
            selected_nodes_sets = [selected_nodes_sets]

        legend_elements = []

        for idx, selected_nodes in enumerate(selected_nodes_sets):
            selected_node_color = colors_list[idx % len(colors_list)]  

            if not isinstance(selected_nodes, list):
                selected_nodes = [selected_nodes]

            for node in selected_nodes:
                if view == '3D':
                    if plot_temperatures:
                        color = cmap(node_to_temperature[node.num])
                        self.ax.scatter(node.x, node.y, node.z, s=size,color=color, edgecolors=node_edgecolors, marker='o', zorder=20)
                    else:
                        self.ax.scatter(node.x, node.y, node.z, s=size,color=selected_node_color, edgecolors=node_edgecolors, zorder=20)
                    if plot_node_numbers:
                        self.ax.text(node.x, node.y, node.z, node.num, color='k')
                elif view == '2D':
                    if plot_temperatures:
                        color = cmap(node_to_temperature[node.num])
                        self.ax.scatter(node.x, node.y, s=size,color=color, edgecolors=node_edgecolors, marker='o', zorder=20)
                    else:
                        self.ax.scatter(node.x, node.y, s=size,color=selected_node_color, edgecolors=node_edgecolors, zorder=20)
                    if plot_node_numbers:
                        self.ax.text(node.x, node.y, node.num, color='k')

            if set_labels:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{set_labels[idx]}', markersize=10, markeredgecolor='k', markerfacecolor=selected_node_color))

        if plot_other_nodes:
            other_node_color = 'white'
            for node in self.truss.nodes:
                if view == '3D':
                    self.ax.scatter(node.x, node.y, node.z, color=other_node_color, marker='o', edgecolors='k', zorder=15, alpha=1)
                elif view == '2D':
                    self.ax.scatter(node.x, node.y, color=other_node_color, marker='o', edgecolors='k', zorder=15, alpha=1)

        if selected_nodes_sets and set_labels:
            self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.9), borderaxespad=0.) # 

    def add_colorbar(self, cmap=None, points_colorbar=None, temperatures=None, 
    temperatures_colorbar=False, 
    thickness_colorbar=False,
    cmap_type='viridis'):

        if temperatures_colorbar:
            vmin = np.min(temperatures)
            vmax = np.max(temperatures)
            scalar_mappable = cm.ScalarMappable(norm=None, cmap=cm.get_cmap(cmap_type))
            scalar_mappable.set_array(temperatures)
            cbar = plt.colorbar(scalar_mappable, ax=self.ax, shrink=0.8)
            cbar.set_label('Temperature [K]')
        
        if thickness_colorbar:
            scalar_mappable = cm.ScalarMappable(cmap=cm.get_cmap(cmap_type))
            scalar_mappable.set_array([])  
            plt.colorbar(scalar_mappable, ax=self.ax, orientation='vertical')

        if points_colorbar:
            scalar_mappable = cm.ScalarMappable(cmap=cm.get_cmap(cmap_type))
            scalar_mappable.set_array([])
            plt.colorbar(scalar_mappable, ax=self.ax, orientation='vertical')

    def finalize_plot(self, view, save_path=None, dpi=300, layout_rect=[0, 0, 1, 1]):
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        if view == '3D':
            self.ax.set_zlabel('Z [m]')
        plt.axis('off')
        if view == '3D':
            self.ax.set_box_aspect([1,1,1])
        else:
            plt.axis('equal')

        if view == '3D':
            x_range = abs(max(node.x for node in self.truss.nodes) - min(node.x for node in self.truss.nodes))
            y_range = abs(max(node.y for node in self.truss.nodes) - min(node.y for node in self.truss.nodes))
            z_range = abs(max(node.z for node in self.truss.nodes) - min(node.z for node in self.truss.nodes))
            max_range = max([x_range, y_range, z_range])

            mid_x = (max(node.x for node in self.truss.nodes) + min(node.x for node in self.truss.nodes)) * 0.5
            mid_y = (max(node.y for node in self.truss.nodes) + min(node.y for node in self.truss.nodes)) * 0.5
            mid_z = (max(node.z for node in self.truss.nodes) + min(node.z for node in self.truss.nodes)) * 0.5
            self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        plt.tight_layout() 
        if save_path: 
            save_path = os.path.normpath(save_path)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight') 
            plt.close() 
        else: 
            plt.show()
        
    def init_plot(self, view, figsize=(5, 5), dpi=300):
        if view == '3D':
            self.fig = plt.figure(figsize=figsize, dpi=dpi)
            self.ax = self.fig.add_subplot(111, projection='3d')
        elif view == '2D':
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

    def plot_truss(self, 
        optimized_areas=None, 
        view='3D',
        view3D="default",
        figsize=(10, 7),
        dpi=300,
        save_path=None,
        plot_bars=False, 
        plot_node_numbers=False, 
        plot_nodes=False, 
        norm=False, 
        plot_temperatures=False, 
        cooling_path=None, 
        selected_nodes=None, 
        plot_other_nodes=False, 
        plot_arrows=False, 
        alpha_bars=1,
        alpha_nodes=1, 
        node_edgecolors='none',
        vicinity_threshold=None,
        plot_vicinity_volumes=False,
        plot_vicinity_areas=False,
        plot_bc_clamp_nodes=True, 
        plot_bc_force_nodes=True, 
        plot_bc_heat_nodes=True,
        plot_unlabeled_nodes=True,
        node_size=30,
        default_bar_scale=2.99, 
        bar_scale_min=0.01,
        enlarge_scale_factor=1.0,
        enlarge_bars=False,
        thickness_factor=1.0,
        plot_bar_thickness_colors=False,
        set_labels=None,
        layout_rect=[0, 0, 1, 1],
        bar_thickness_threshold=None,
        cmap_type='viridis',
        displacements=False, 
        deformation_scale=10,
        plot_force_arrows=False, 
        force_scale=1, 
        force_arrow_color='#d70929',
        plot_dof_arrows=False, 
        dof_arrow_color='#0072bb', 
        dof_arrow_length=10,
        plot_original_areas=False, 
        original_areas_alpha=0.3, 
        original_areas_scale_factor=0.5, 
        original_areas_color='#000000', 
        optimized_areas_color='#0072bb',
        plot_cooling_path_res=False, cooling_path_color='#f9b002',
        animate_modes=None,
        animation_pause=0.01,
        animation_steps=100,
        save_animation=None
        ):
        
        self.init_plot(view,figsize=figsize, dpi=dpi)
        cmap, norm_temperatures, node_to_temperature, temperatures = self.get_cmap_and_norm_temperatures()


        if animate_modes is not None:
            self.animate_modes = animate_modes
            self.eigenfreq, self.eigenvec = self.truss.eigenfreqs[animate_modes], self.truss.eigenvecs[:, animate_modes]
            
            time_step = np.linspace(-1, 1, animation_steps)
            time_step2 = np.linspace(1, -1, animation_steps)
            interpolated_displacements = np.array([(eig_vec * time_step) for eig_vec in self.eigenvec])
            interpolated_displacements2 = np.array([(eig_vec * time_step2) for eig_vec in self.eigenvec])

            interpolated_displacements_hstack = np.hstack((interpolated_displacements,interpolated_displacements2))


            max_x = max(node.x for node in self.truss.nodes)
            min_x = min(node.x for node in self.truss.nodes)
            max_y = max(node.y for node in self.truss.nodes)
            min_y = min(node.y for node in self.truss.nodes)
            x_margin = 0.1 * max_x
            y_margin = 0.1 * max_y

            plt.axis('equal')
            self.ax.set_axis_off()
            def on_key(event):
                if event.key == 'escape':
                    self.stop_animation = True  
            def on_close(event):
                self.stop_animation = True 

            self.fig.canvas.mpl_connect('key_press_event', on_key)
            self.fig.canvas.mpl_connect('close_event', on_close)
            self.stop_animation = False
            time_step_hstack = np.append(time_step, time_step2)
            if save_animation:
                def update(frame):
                    if self.stop_animation:
                        plt.close()
                        return
                    self.truss.eigenvecs[:, animate_modes] = interpolated_displacements_hstack[:, frame]
                    self.truss.assign_displacements(animate_modes)
                    self.ax.clear()
                    self.plot_bars_function(
                        optimized_areas,
                        norm,
                        view,
                        alpha_bars,
                        default_factor=default_bar_scale,
                        default_min=bar_scale_min,
                        scale_factor=enlarge_scale_factor,
                        enlarge=enlarge_bars,
                        plot_bar_thickness_colors=plot_bar_thickness_colors,
                        threshold=bar_thickness_threshold,
                        cmap_type=cmap_type,
                        thickness_factor=thickness_factor,
                        displacements=displacements,
                        deformation_scale=deformation_scale,
                        plot_original_areas=plot_original_areas,
                        original_areas_alpha=original_areas_alpha,
                        original_areas_scale_factor=original_areas_scale_factor,
                        original_areas_color=original_areas_color,
                        optimized_areas_color=optimized_areas_color,
                        plot_cooling_path_res=plot_cooling_path_res,
                        cooling_path_color=cooling_path_color)
                    self.ax.set_xlim(min_x - x_margin, max_x + x_margin)
                    self.ax.set_ylim(min_y - y_margin, max_y + y_margin)
                    self.ax.set_axis_off()

                ani = FuncAnimation(self.fig, update, frames=len(time_step_hstack), interval=animation_pause)
                ani.save(rf'{save_animation}.gif', writer='pillow')
                plt.close()
                return
            else:
                while True:
                    for en, time in enumerate(time_step):
                        if self.stop_animation:
                            plt.close()
                            return
                        self.truss.eigenvecs[:, animate_modes] = interpolated_displacements[:, en]
                        self.truss.assign_displacements(animate_modes)
                        self.ax.clear()
                        self.plot_bars_function(
                            optimized_areas, 
                            norm, 
                            view, 
                            alpha_bars, 
                            default_factor=default_bar_scale, 
                            default_min=bar_scale_min, 
                            scale_factor=enlarge_scale_factor, 
                            enlarge=enlarge_bars,
                            plot_bar_thickness_colors=plot_bar_thickness_colors,
                            threshold=bar_thickness_threshold,
                            cmap_type=cmap_type,
                            thickness_factor=thickness_factor,
                            displacements=displacements, 
                            deformation_scale=deformation_scale,
                            plot_original_areas=plot_original_areas, 
                            original_areas_alpha=original_areas_alpha, 
                            original_areas_scale_factor=original_areas_scale_factor, 
                            original_areas_color=original_areas_color, 
                            optimized_areas_color=optimized_areas_color,
                            plot_cooling_path_res=plot_cooling_path_res, 
                            cooling_path_color=cooling_path_color)
                        self.ax.set_xlim(min_x - x_margin, max_x + x_margin)
                        self.ax.set_ylim(min_y - y_margin, max_y + y_margin)
                        plt.pause(animation_pause)  
                    interpolated_displacements = np.flip(interpolated_displacements, axis=1) 
        if plot_bars:
            self.plot_bars_function(optimized_areas, norm, view, alpha_bars, default_factor=default_bar_scale, default_min=bar_scale_min, 
            scale_factor=enlarge_scale_factor, 
            enlarge=enlarge_bars,
            plot_bar_thickness_colors=plot_bar_thickness_colors,
            threshold=bar_thickness_threshold,
            cmap_type=cmap_type,
            thickness_factor=thickness_factor,
            displacements=displacements, deformation_scale=deformation_scale,
            plot_original_areas=plot_original_areas, original_areas_alpha=original_areas_alpha, 
            original_areas_scale_factor=original_areas_scale_factor, 
            original_areas_color=original_areas_color, optimized_areas_color=optimized_areas_color,
            plot_cooling_path_res=plot_cooling_path_res, cooling_path_color=cooling_path_color)

        if cooling_path is not None:
            self.plot_cooling_path(cooling_path, plot_arrows, view, optimized_areas, norm)

        if selected_nodes is not None:
            self.plot_selected_nodes(selected_nodes, plot_temperatures, plot_other_nodes, view, cmap, norm_temperatures, plot_node_numbers, node_to_temperature, set_labels=set_labels,alpha_nodes=alpha_nodes, node_edgecolors=node_edgecolors,size=node_size)

        if plot_nodes:
            self.plot_nodes_function(
            plot_temperatures, 
            norm_temperatures, 
            cmap, 
            view,
            plot_node_numbers,
            alpha_nodes=alpha_nodes,
            node_edgecolors=node_edgecolors,
            vicinity_threshold=vicinity_threshold,
            plot_vicinity_volumes=plot_vicinity_volumes,
            plot_vicinity_areas=plot_vicinity_areas,
            plot_bc_clamp_nodes=plot_bc_clamp_nodes, 
            plot_bc_force_nodes=plot_bc_force_nodes, 
            plot_bc_heat_nodes=plot_bc_heat_nodes,
            plot_unlabeled_nodes=plot_unlabeled_nodes,
            size=node_size,
            cmap_type=cmap_type,
            plot_force_arrows=plot_force_arrows, 
            force_arrow_color=force_arrow_color,
            force_scale=force_scale, 
            plot_dof_arrows=plot_dof_arrows, 
            dof_arrow_color=dof_arrow_color, 
            dof_arrow_length=dof_arrow_length)

        if plot_temperatures:
            self.add_colorbar(cmap=cmap, temperatures=temperatures, 
            temperatures_colorbar=True,
            cmap_type=cmap_type)
        if plot_bars and plot_bar_thickness_colors:
            self.add_colorbar(thickness_colorbar=True,cmap_type=cmap_type)
        if plot_vicinity_volumes or plot_vicinity_areas:
            self.add_colorbar(points_colorbar=True,cmap_type=cmap_type)

        if view == '3D':
            if view3D == "default":
                pass  
            elif view3D == "xy":
                self.ax.view_init(elev=90, azim=90)
            elif view3D == "xz":
                self.ax.view_init(elev=0, azim=90)
            elif view3D == "yz":
                self.ax.view_init(elev=0, azim=0)
            elif isinstance(view3D, dict):
                self.ax.view_init(elev=view3D["elev"], azim=view3D["azim"])
                
        self.finalize_plot(view, dpi=dpi, save_path=save_path, layout_rect=layout_rect)

    @staticmethod
    def multiline(xs, ys, cs, ax=None, **kwargs):
        """Plot lines with different colorings in 2D

        Parameters
        ----------
        xs : iterable container of x coordinates
        ys : iterable container of y coordinates
        cs : iterable container of colors
        ax (optional): Axes to plot on.
        kwargs (optional): passed to LineCollection

        Returns
        -------
        lc : LineCollection instance.
        """
        ax = plt.gca() if ax is None else ax

        segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
        lc = LineCollection(segments, **kwargs)

        lc.set_array(np.asarray(cs))

        ax.add_collection(lc)
        ax.autoscale()
        return lc

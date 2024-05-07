from Node import Node
from Bar import Bar
import Materials
import AntColony

import os
from datetime import datetime
import numpy as np
import cvxpy as cp
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from scipy import sparse
import pickle

class Truss:
    def __init__(self):
        self.nodes = []
        self.bars = []
        self.F = None 
        self.K = None 
        self.M = None 
        self.u = None 
        self.nodes_dict = {}
        self.bars_dict = {}
        self.bar_path = []
        self.node_path = []
        self.bar_num_path = []
        self.node_num_path = []
        self.res = None
        self.dofs = None
        self.global_forces = None
        self.geometryMatrix = None
        self.geometry_matrix_transposed = None

    @property
    def number_of_bars(self):
        return len(self.bars)

    @property
    def number_of_nodes(self):
        return len(self.nodes)

    def add_node(self, node):
        self.nodes.append(node)
        self.nodes_dict[node.num] = node

    def add_nodes(self, nodes):
        for en,node in enumerate(nodes):
            x = node[0]
            y = node[1]
            z = node[2]
            node_to_add = Node(x, y, z, en)
            self.add_node(node_to_add)

    def add_bar(self, bar):
        self.bars.append(bar)
        self.bars_dict[bar.num] = bar

    def add_bars(self, bars):
        for bar in bars:
            self.add_bar(bar)

    def remove_node(self, node):
        self.nodes.remove(node)
        self.bars = [b for b in self.bars if b.node1 != node and b.node2 != node]

    def remove_bar(self, bar):
        self.bars.remove(bar)

    def remove_bar_by_length(self, length):
        self.bars = [b for b in self.bars if b.length <= length]

    def remove_bar_by_diagonal_multiple(self, multiple=1.1):
        self.bars = [b for b in self.bars if b.length < np.sqrt(self.dx**2 + self.dy**2)*multiple]

    def create_bars(self, bar_area, bar_material, bar_max_length=None, verbose=False):
        n = len(self.nodes)
        bar_num = 0
        for i in range(n):
            if verbose:
                print(f'Creating bars from node: {i}/{n}')
            for j in range(i+1, n):
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                length = node1.distance_to(node2)
                if  bar_max_length and length > bar_max_length:
                    continue
                bar = Bar(node1, node2, bar_material, bar_area, length, num=bar_num)
                self.add_bar(bar)
                bar_num += 1

    def create_rectangular_grid(self, nx, ny, max_x_length, max_y_length):
        self.dx = max_x_length / (nx - 1)
        self.dy = max_y_length / (ny - 1)
        num = 0
        for j in range(ny):
            for i in range(nx):
                x = i * self.dx
                y = j * self.dy
                node = Node(x, y, 0, num)
                num += 1
                self.add_node(node)

    def create_cuboid_grid(self, nx, ny, nz, max_x_length, max_y_length, max_z_length):
        dx = max_x_length / (nx - 1)
        dy = max_y_length / (ny - 1)
        dz = max_z_length / (nz - 1)

        node_num = 0
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x = i * dx
                    y = j * dy
                    z = k * dz
                    self.add_node(Node(x, y, z, num=node_num))
                    node_num += 1

    def is_within_center_distance(self, node, x_center, y_center, z_center, x_dist, y_dist, z_dist):
        return abs(node.x - x_center) < x_dist and abs(node.y - y_center) < y_dist and abs(node.z - z_center) < z_dist

    def set_node_labels(self, node, bc_force_label, bc_clamp_label, bc_heat_label):
        if bc_force_label is not None:
            node.bc_force_label = bc_force_label
        if bc_clamp_label is not None:
            node.bc_clamp_label = bc_clamp_label
        if bc_heat_label is not None:
            node.bc_heat_label = bc_heat_label

    def set_surface_nodes_BC(self, surface, bc_force_label=None, bc_clamp_label=None, bc_heat_label=None, distance_ratio=0.25, force_values=None, clamp_values=None, temperature=None):
        x_min, x_max = min(node.x for node in self.nodes), max(node.x for node in self.nodes)
        y_min, y_max = min(node.y for node in self.nodes), max(node.y for node in self.nodes)
        z_min, z_max = min(node.z for node in self.nodes), max(node.z for node in self.nodes)

        x_dist, y_dist, z_dist = (x_max - x_min) * distance_ratio, (y_max - y_min) * distance_ratio, (z_max - z_min) * distance_ratio

        for node in self.nodes:
            within_boundary_condition = False

            if surface == 'bottom' and node.z == z_min:
                y_center, x_center = (y_min + y_max) / 2, (x_min + x_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, x_center, y_center, node.z, x_dist, y_dist, z_dist)
            elif surface == 'top' and node.z == z_max:
                y_center, x_center = (y_min + y_max) / 2, (x_min + x_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, x_center, y_center, node.z, x_dist, y_dist, z_dist)
            elif surface == 'front' and node.x == x_min:
                y_center, z_center = (y_min + y_max) / 2, (z_min + z_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, node.x, y_center, z_center, x_dist, y_dist, z_dist)
            elif surface == 'back' and node.x == x_max:
                y_center, z_center = (y_min + y_max) / 2, (z_min + z_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, node.x, y_center, z_center, x_dist, y_dist, z_dist)
            elif surface == 'left' and node.y == y_min:
                x_center, z_center = (x_min + x_max) / 2, (z_min + z_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, x_center, node.y, z_center, x_dist, y_dist, z_dist)
            elif surface == 'right' and node.y == y_max:
                x_center, z_center = (x_min + x_max) / 2, (z_min + z_max) / 2
                within_boundary_condition = self.is_within_center_distance(node, x_center, node.y, z_center, x_dist, y_dist, z_dist)

            if within_boundary_condition:
                self.set_node_labels(node, bc_force_label, bc_clamp_label, bc_heat_label)
                if force_values:
                    node.forces = force_values
                if clamp_values:
                    node.boundary_conditions = clamp_values
                if temperature is not None:
                    node.temperature = temperature

    def assemble_truss_stiffness_mass_matrix(self, areas=None, mass_matrix=False):
        num_dofs = self.number_of_nodes * 3

        self.K = np.zeros((num_dofs, num_dofs))
        self.F = np.zeros((num_dofs, 1))
        if mass_matrix:
            self.M = np.zeros((num_dofs, num_dofs))

        for en,bar in enumerate(self.bars):
            if areas is not None:
                bar.area = areas[en]

            node1 = bar.node1
            node2 = bar.node2
            idx1 = node1.num * 3
            idx2 = node2.num * 3

            bar.truss_stiffness_matrix()
            k_bar = bar.k_global
            if isinstance(bar.area,(float,int)):
                self.K[idx1:idx1+3, idx1:idx1+3] = self.K[idx1:idx1+3, idx1:idx1+3] +  k_bar[:3, :3]
                self.K[idx1:idx1+3, idx2:idx2+3] = self.K[idx1:idx1+3, idx2:idx2+3] +  k_bar[:3, 3:]
                self.K[idx2:idx2+3, idx1:idx1+3] = self.K[idx2:idx2+3, idx1:idx1+3] +  k_bar[3:, :3]
                self.K[idx2:idx2+3, idx2:idx2+3] = self.K[idx2:idx2+3, idx2:idx2+3] +  k_bar[3:, 3:]
            else:
                k_bar_global = np.zeros((num_dofs, num_dofs))
                k_bar_global[idx1:idx1+3, idx1:idx1+3] = k_bar[:3, :3]
                k_bar_global[idx1:idx1+3, idx2:idx2+3] = k_bar[:3, 3:]
                k_bar_global[idx2:idx2+3, idx1:idx1+3] = k_bar[3:, :3]
                k_bar_global[idx2:idx2+3, idx2:idx2+3] = k_bar[3:, 3:]

                self.K = self.K + bar.area*k_bar_global

            if mass_matrix:

                bar.truss_lumped_mass_matrix()
                m_bar = bar.mass_matrix_global
                if isinstance(bar.area,(float,int)):
                    self.M[idx1:idx1+3, idx1:idx1+3] = self.M[idx1:idx1+3, idx1:idx1+3] + m_bar[:3, :3]
                    self.M[idx1:idx1+3, idx2:idx2+3] = self.M[idx1:idx1+3, idx2:idx2+3] + m_bar[:3, 3:]
                    self.M[idx2:idx2+3, idx1:idx1+3] = self.M[idx2:idx2+3, idx1:idx1+3] + m_bar[3:, :3]
                    self.M[idx2:idx2+3, idx2:idx2+3] = self.M[idx2:idx2+3, idx2:idx2+3] + m_bar[3:, 3:]
                else:
                    m_bar_global = np.zeros((num_dofs, num_dofs))
                    m_bar_global[idx1:idx1+3, idx1:idx1+3] = m_bar[:3, :3]
                    m_bar_global[idx1:idx1+3, idx2:idx2+3] = m_bar[:3, 3:]
                    m_bar_global[idx2:idx2+3, idx1:idx1+3] = m_bar[3:, :3]
                    m_bar_global[idx2:idx2+3, idx2:idx2+3] = m_bar[3:, 3:]

                    self.M = self.M + bar.area*m_bar_global

    def truss_stiffness_matrix(self, areas=None):
        num_dofs = self.num_nodes * 3

        self.K = np.zeros((num_dofs, num_dofs))
        self.F = np.zeros((num_dofs, 1))

        for en,bar in enumerate(self.bars):
            if areas is not None:
                bar.area = areas[en]
            node1 = bar.node1
            node2 = bar.node2
            idx1 = node1.num * 3
            idx2 = node2.num * 3
            bar.truss_stiffness_matrix()
            k_bar = bar.k_global
            self.K[idx1:idx1+3, idx1:idx1+3] = self.K[idx1:idx1+3, idx1:idx1+3] + k_bar[:3, :3]
            self.K[idx1:idx1+3, idx2:idx2+3] = self.K[idx1:idx1+3, idx2:idx2+3] + k_bar[:3, 3:]
            self.K[idx2:idx2+3, idx1:idx1+3] = self.K[idx2:idx2+3, idx1:idx1+3] + k_bar[3:, :3]
            self.K[idx2:idx2+3, idx2:idx2+3] = self.K[idx2:idx2+3, idx2:idx2+3] + k_bar[3:, 3:]

        return self.K

    def truss_mass_matrix(self, areas=None):
        num_dofs = self.number_of_nodes * 3
        self.M = np.zeros((num_dofs, num_dofs))

        for en,bar in enumerate(self.bars):
            if areas is not None:
                bar.area = areas[en]
            node1 = bar.node1
            node2 = bar.node2
            idx1 = node1.num * 3
            idx2 = node2.num * 3

            bar.truss_lumped_mass_matrix()
            m_bar = bar.mass_matrix_global

            if isinstance(bar.area,(float,int)):
                self.M[idx1:idx1+3, idx1:idx1+3] = self.M[idx1:idx1+3, idx1:idx1+3] + m_bar[:3, :3]
                self.M[idx1:idx1+3, idx2:idx2+3] = self.M[idx1:idx1+3, idx2:idx2+3] + m_bar[:3, 3:]
                self.M[idx2:idx2+3, idx1:idx1+3] = self.M[idx2:idx2+3, idx1:idx1+3] + m_bar[3:, :3]
                self.M[idx2:idx2+3, idx2:idx2+3] = self.M[idx2:idx2+3, idx2:idx2+3] + m_bar[3:, 3:]
            else:
                m_bar_global = np.zeros((num_dofs, num_dofs))
                m_bar_global[idx1:idx1+3, idx1:idx1+3] = m_bar[:3, :3]
                m_bar_global[idx1:idx1+3, idx2:idx2+3] = m_bar[:3, 3:]
                m_bar_global[idx2:idx2+3, idx1:idx1+3] = m_bar[3:, :3]
                m_bar_global[idx2:idx2+3, idx2:idx2+3] = m_bar[3:, 3:]

                self.M = self.M + bar.area*m_bar_global
        for node in self.nodes:
            if node.concentrated_mass:
                idx = node.num * 3
                node_mass_matrix = np.zeros(np.shape(self.M))
                node_mass_matrix[idx:idx+3, idx:idx+3] = np.diag([node.concentrated_mass, node.concentrated_mass, node.concentrated_mass])
                self.M += node_mass_matrix

        return self.M

    def apply_boundary_conditions(self):
        for node in self.nodes:
            bc = node.boundary_conditions
            forces = node.forces
            idx = node.num * 3

            if bc is not None:

                if bc[0]:  
                    self.K[idx, :] = 0
                    self.K[:, idx] = 0
                    self.K[idx, idx] = 1

                if bc[1]:  
                    self.K[idx+1, :] = 0
                    self.K[:, idx+1] = 0
                    self.K[idx+1, idx+1] = 1

                if bc[2]:  
                    self.K[idx+2, :] = 0
                    self.K[:, idx+2] = 0
                    self.K[idx+2, idx+2] = 1

            if forces is not None or node.thermal_load is not None:
                forces = forces if forces is not None else np.zeros(3)
                thermal_load = node.thermal_load if node.thermal_load is not None else np.zeros(3)
                self.F[idx] = forces[0] + thermal_load[0]
                self.F[idx+1] = forces[1] + thermal_load[1]
                self.F[idx+2] = forces[2] + thermal_load[2]

            if node.concentrated_mass:
                self.M[idx,idx] += node.concentrated_mass 
                self.M[idx+1,idx+1] += node.concentrated_mass 
                self.M[idx+2,idx+2] += node.concentrated_mass 

    def apply_boundary_conditions_cvxpy(self):
        num_dofs = self.number_of_nodes * 3
        unconstrained_dofs = np.ones(num_dofs, dtype=bool)

        nonstructural_mass = np.zeros((num_dofs, num_dofs))

        for node in self.nodes:
            idx = node.num * 3
            if node.boundary_conditions is not None:
                for i, bc in enumerate(node.boundary_conditions):
                    if bc:
                        unconstrained_dofs[idx + i] = False

        self.M = self.M + nonstructural_mass

        self.K = self.K[unconstrained_dofs, :][:, unconstrained_dofs]
        self.M = self.M[unconstrained_dofs, :][:, unconstrained_dofs]
        self.F = self.F[unconstrained_dofs]

    def zero_cross_section(self):
        p1 = np.diag(self.K)
        p1 = np.array(p1)

        findzeros = np.where(p1 < 1e-5)
        for i in findzeros:
            p1[i] = 1
        np.fill_diagonal(self.K, 0)
        self.K = self.K + np.diag(p1)

    def solve(self):
        self.U = np.linalg.solve(self.K, self.F)

        for node in self.nodes:
            idx = node.num * 3

            node.displacements = self.U[idx:idx+3]

    def assign_temperatures(self, comsol_file_path):
        with open(comsol_file_path, 'r') as f:
            lines = f.readlines()

        dimensions = 2 if '% Dimension:          2\n' in lines else 3
        comsol_data = np.loadtxt(comsol_file_path, skiprows=9)

        tree = cKDTree(comsol_data[:, :dimensions])

        for node in self.nodes:
            dist, idx = tree.query(node.coordinates[:dimensions])

            node.temperature = comsol_data[idx, -1]

    def InitiateGraph(self, selected_nodes=None):
        self.G_selected = nx.Graph()
        self.G_ground_structure = nx.Graph()

        for node in self.nodes:
            self.G_ground_structure.add_node(node.num)

        for bar in self.bars:
            self.G_ground_structure.add_edge(bar.node1.num, bar.node2.num, weight=bar.length)

        if selected_nodes is not None:
            selected_nodes = [node.num for node in selected_nodes]
            for node in selected_nodes:
                self.G_selected.add_node(node, prize=self.nodes[node].temperature)
            for bar in self.bars:
                if bar.node1.num in selected_nodes and bar.node2.num in selected_nodes:
                    self.G_selected.add_edge(bar.node1.num, bar.node2.num, weight=bar.length)
            for node1 in self.G_selected.nodes:
                for node2 in self.G_selected.nodes:
                    if node1 != node2 and not self.G_selected.has_edge(node1, node2):
                        self.G_selected.add_edge(node1, node2, weight=100*self.nodes_dict[node1].distance_to(self.nodes_dict[node2]), )

    def InitiateGraphNode(self, selected_nodes=None):
        self.G_selected = nx.Graph()
        self.G_ground_structure = nx.Graph()

        for node in self.nodes:
            self.G_ground_structure.add_node(node)

        for bar in self.bars:
            self.G_ground_structure.add_edge(bar.node1, bar.node2, weight=bar.length)

        if selected_nodes is not None:
            selected_nodes = [node for node in selected_nodes]
            for node in selected_nodes:
                self.G_selected.add_node(node, prize=node.temperature)
            for bar in self.bars:
                if bar.node1 in selected_nodes and bar.node2 in selected_nodes:
                    self.G_selected.add_edge(bar.node1, bar.node2, weight=bar.length)
            for node1 in self.G_selected.nodes:
                for node2 in self.G_selected.nodes:
                    if node1 != node2 and not self.G_selected.has_edge(node1, node2):
                        self.G_selected.add_edge(node1, node2, weight=node1.distance_to(node2)*1, )

    def tsp_aco(self, 
        graph, 
        main_graph, 
        n_ants, 
        n_iterations, 
        alpha, 
        beta, 
        evaporation_rate,
        prescribed_start=None,
        prescribed_end=None,
        start_node=None, 
        end_node=None, 
        # q=1,
        loop=False,
        plot_progress=False):

        required_nodes_list = [node.num for node in self.hottest_nodes]

        self.colony = AntColony.AntColony(
                        graph=graph, 
                        main_graph=main_graph, 
                        n_ants=n_ants, 
                        n_iterations=n_iterations, 
                        alpha=alpha, 
                        beta=beta,
                        evaporation_rate=evaporation_rate,
                        prescribed_start=prescribed_start,
                        prescribed_end=prescribed_end,
                        start_node=start_node,
                        end_node=end_node,
                        required_nodes_list=required_nodes_list,
                        loop=loop,
                        plot_progress=plot_progress,)
        self.best_aco_path = self.colony.run()

    # %%
    ###########

    def create_global_dofs_and_force_vector(self):
        self.dofs = []
        self.global_forces = []

        for node in self.nodes:
            # Process boundary conditions
            if node.boundary_conditions is None:
                node.boundary_conditions = [False, False, False]
            node_dofs = [not bc for bc in node.boundary_conditions]
            self.dofs.extend(node_dofs)

            if node.forces is None:
                node.forces = [0, 0, 0] ###
            # Process forces
            self.global_forces.extend(node.forces) 

        self.dofs = np.array(self.dofs)
        self.global_forces = np.array(self.global_forces)

    def transf_matrix(self, angle, axis):
        self.T = np.zeros((6,6))

        if axis == 'x':
            self.T = np.array([[1, 0, 0, ],
                          [0, np.cos(angle), np.sin(angle)],
                          [0, -np.sin(angle), np.cos(angle)]])
        elif axis == 'y':
            self.T = np.array([[np.cos(angle), 0, -np.sin(angle)],
                          [0, 1, 0],
                          [np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'z':
            self.T = np.array([[np.cos(angle), np.sin(angle),0],
                          [-np.sin(angle), np.cos(angle),0],
                          [0, 0, 1]])

    def geometry_matrix(self, method = None, transform=False, angle=0, axis='x', store_transposed=False):
        # Initialize an empty list to store the geometry matrix data
        rows = []
        cols = []
        data = []

        if transform:
            self.transform_coordinates(angle, axis)

        for idx, bar in enumerate(self.bars):
            node1 = bar.node1.num
            node2 = bar.node2.num

            c1, c2, c3 = bar.direction_cosines.flatten()  # Assuming 3D, for 2D use c1, c2

            # Fill the geometry matrix data
            rows.extend([idx, idx, idx, idx, idx, idx])
            cols.extend([node1*3, node1*3+1, node1*3+2, node2*3, node2*3+1, node2*3+2])
            data.extend([-c1, -c2, -c3, c1, c2, c3])

        # Create the sparse global geometry matrix
        if method == 'cvxopt':
            geometryMatrix = cpo.spmatrix(data, rows, cols)
        else:
            geometryMatrix = sparse.csr_matrix((data, (rows, cols)), shape=(self.number_of_bars, self.number_of_nodes*3))

        # If dofs (degrees of freedom) are provided, select the relevant columns
        if self.dofs is not None:
            self.global_forces = self.global_forces[self.dofs]
            if method == 'cvxopt':
                dofs_indices = [i for i, keep in enumerate(self.dofs) if keep]
                geometryMatrix = geometryMatrix[:, dofs_indices]
                self.global_forces = cpo.matrix(self.global_forces)
            else:
                geometryMatrix = geometryMatrix[:, self.dofs]

        self.geometryMatrix = geometryMatrix
        if store_transposed:
            self.geometry_matrix_transposed = geometryMatrix.T

    def instantiate_E_lengths_vectors(self, areas_method=False, areas=None):
        self.E_array = np.array([bar.material.E for bar in self.bars])
        self.L_array = np.array([bar.length for bar in self.bars])

        if areas_method == ('cvxpy' or 'cvxopt'):
            self.A_array = areas
        elif areas_method == 'numpy':
            self.A_array = np.array([bar.area for bar in self.bars])
        elif areas_method == None:
            raise ValueError('areas_method must be provided')

    def diagonal_properties_matrix(self, areas_method=False, areas=None):

        if areas_method == 'cvxpy':
            D_values = cp.diag(cp.multiply(self.E_array / self.L_array, areas))
        elif areas_method ==  'numpy':
            D_values = sparse.diags(self.E_array * self.A_array / self.L_array)
        elif areas_method ==  None:
            raise ValueError('areas_method must be provided')

        return D_values

    def efficient_stiffness_matrix(self, diagonal_matrix=None):

        if diagonal_matrix is None:
            raise ValueError('diagonal_matrix must be provided')

        if self.geometry_matrix_transposed is not None:
            K_global = self.geometry_matrix_transposed @ diagonal_matrix @ self.geometryMatrix
        else:
            K_global = self.geometryMatrix.T @ diagonal_matrix @ self.geometryMatrix

        return K_global

    def prepare_for_solving(self):
        self.create_global_dofs_and_force_vector()
        self.geometry_matrix()

    def efficient_solve(self, diagonal_matrix=None, write_to_nodes=False):

        # Solve for the displacements
        self.K = self.efficient_stiffness_matrix(diagonal_matrix=diagonal_matrix)

        self.U = spsolve(self.K, self.global_forces)

        # Expand the displacement vector to include the constrained DOFs
        expanded_U = np.zeros(len(self.dofs))
        expanded_U[self.dofs] = self.U

        # If write_to_nodes is True, write the displacements back to the nodes
        if write_to_nodes:
            for i, node in enumerate(self.nodes):
                node_displacements = expanded_U[i*3:(i+1)*3]  # Assuming 3 DOFs per node
                node.set_displacements(node_displacements)

    def transform_coordinates(self, angle, axis):

        self.transf_matrix(angle, axis)

        for node in self.nodes:
            node.coordinates = self.T @ node.coordinates
            node.x = node.coordinates[0]
            node.y = node.coordinates[1]
            node.z = node.coordinates[2]

    ###########
    # %%
    def bar_path_from_node_path(self, node_path=None):
        self.bar_path = []
        self.bar_num_path = []
        if node_path is None:
            node_path = self.node_num_path
        # while node < len(self.node_path)-1:
        for index in range(len(node_path)-1):
            node1 = node_path[index]
            node2 = node_path[index+1]
            selected_bar = self.select_bar(node1, node2)
            selected_bar.pipe = True
            selected_bar.node_in = node1
            selected_bar.node_out = node2
            self.bar_path.append(selected_bar)
            self.bar_num_path.append(selected_bar.num)

    def bar_path_from_node_num_path(self, node_num_path=None):
        self.bar_path = []
        self.bar_num_path = []
        if node_num_path is None:
            node_num_path = self.node_num_path
        # while node < len(self.node_num_path)-1:
        for index in range(len(node_num_path) - 1):
            node1 = node_num_path[index]
            node2 = node_num_path[index + 1]
            selected_bar = self.select_bar_nums(node1, node2)
            selected_bar.pipe = True
            selected_bar.node_in = node1
            selected_bar.node_out = node2
            self.bar_path.append(selected_bar)
            self.bar_num_path.append(selected_bar.num)

    def select_bar(self, node1, node2):
        for bar in self.bars:
            if (bar.node1.num == node1 and bar.node2.num == node2) or (bar.node1.num == node2 and bar.node2.num == node1):
                return bar

    def select_bar_nums(self, node1, node2):
        for bar in self.bars:
            if (bar.node1.num == node1 and bar.node2.num == node2) or (
                bar.node1.num == node2 and bar.node2.num == node1
            ):
                return bar

    def node_path_from_node_nums(self, path):
        for node_in_path in path:
            self.node_path.append(self.nodes_dict[node_in_path])

    def find_invalid_segments(self, node_path):
        invalid_segments = []
        for i in range(len(node_path)-1):
            if not self.is_edge_valid(node_path[i],node_path[i+1]):
                invalid_segments.append((node_path[i], node_path[i+1]))
        return invalid_segments

    def is_edge_valid(self, node1, node2):
        if self.G_ground_structure.has_edge(node1, node2):
            return True
        return False

    def find_and_insert_shortest_paths(self,node_path, invalid_segments):
        for segment in invalid_segments:
            node_start, node_end = segment

            index_start = node_path.index(node_start)
            index_end = node_path.index(node_end)

            G_temp = self.G_ground_structure.copy()

            for node in node_path:
                if node not in segment:
                    G_temp.remove_node(node)

            try:
                shortest_path = nx.dijkstra_path(G_temp, node_start, node_end)
            except nx.exception.NetworkXNoPath:

                shortest_path = [node_start, node_end]
                self.G_ground_structure.add_edge(node_start, node_end, weight=self.nodes_dict[node_start].distance_to(self.nodes_dict[node_end]))
                self.add_bar(Bar(self.nodes_dict[node_start], self.nodes_dict[node_end], self.bars[0].material, self.bars[0].area, 
                                self.nodes_dict[node_start].distance_to(self.nodes_dict[node_end]), 
                                num=len(self.bars))
                                )

            node_path = node_path[:index_start+1] + shortest_path[1:-1] + node_path[index_end:]
        self.best_aco_path_fixed = node_path
        return node_path

    def fix_path(self):
        i = 0
        while i < len(self.path_dij) - 1:
            if not self.is_edge_valid(self.path_dij[i], self.path_dij[i+1]):
                sub_path = self.find_alternate_path(self.path_dij[i], self.path_dij[i+1])

                self.path_dij = self.path_dij[:i+1] + sub_path + self.path_dij[i+2:]

                i = max(i - len(sub_path), 0)
            else:
                i += 1

    def find_alternate_path(self, node1, node2):

        subgraph = self.G_ground_structure.copy()
        for i in range(len(self.path_dij) - 1):
            if subgraph.has_edge(self.path_dij[i], self.path_dij[i+1]):
                subgraph.remove_edge(self.path_dij[i], self.path_dij[i+1])
        try:
            return nx.dijkstra_path(subgraph, node1, node2, weight='weight')[1:-1]
        except nx.NetworkXNoPath:
            return []

    def select_hottest_nodes(self, percentage):
        self.selected_node_percentage = percentage
        temperatures = np.array([node.temperature for node in self.nodes])

        threshold = np.percentile(temperatures, 100 - percentage)
        self.hottest_nodes = [node for node in self.nodes if node.temperature >= threshold]

    def cooling(self, r_inner, width, v, Tf_in, path=None,verbose=False):
        temp_along_path = [Tf_in]
        Qdots = []
        if path is None:
            path = self.bar_path

        for en,bar in enumerate(path):
            if not bar.pipe:
                print('bar is not pipe') 
            bar.v = v
            bar.Tf_in = Tf_in
            bar.coolant = Materials.Water(Tf_in, r_inner, v)
            bar.Ts = (bar.node1.temperature + bar.node2.temperature)/2
            bar.ntu = bar.coolant.Nu*bar.coolant.k*bar.length*2/(bar.coolant.rho*v*bar.coolant.Cp*r_inner**2)
            bar.Tf_out = bar.Ts -(bar.Ts - Tf_in)*np.exp(-bar.ntu)
            bar.Qf_dot = bar.coolant.rho*bar.coolant.Cp*v*np.pi*r_inner**2*(bar.Tf_out- Tf_in)

            if verbose:
                print(f'en: {en}, Tf_out: {bar.Tf_out}')
            Tf_in = bar.Tf_out
            temp_along_path.append(Tf_in)
            Qdots.append(bar.Qf_dot)

        Qdot_sum = np.sum(Qdots)
        return Qdot_sum, temp_along_path, Qdots

    def assign_displacements(self, mode_num:int):
        for node in self.nodes:
            node.displacements = np.zeros(3)

        unconstrained_node_indices = [node.num for node in self.nodes if not node.bc_clamp_label]
        for idx, node_idx in enumerate(unconstrained_node_indices):
            displacement_indices = range(idx * 2, idx * 2 + 2)
            node = self.nodes[node_idx]
            node.displacements = np.append(self.eigenvecs[:, mode_num][displacement_indices],0)

    def save(self, filename=None):
        if filename is None:
            filename = datetime.now().strftime('%Y-%m-%d_%H-%M') + '.pkl'
        filename = os.path.normpath(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        filename = os.path.normpath(filename)
        with open(filename, 'rb') as f:
            loaded_obj = pickle.load(f)
        self.__dict__.update(loaded_obj.__dict__)

    def load_nodes_and_elements(self, 
    filename, 
    create_bars_from_elements=True, 
    create_nodes=True,
    bar_materials=Materials.Inconel718(),
    bar_area=1
    ):
        filename = os.path.normpath(filename)

        with open(f'{filename}.pkl', 'rb') as f:
            data = pickle.load(f)

        self.nodes = data['nodes']
        self.elements = data['elements']

        if create_nodes:
            self.create_nodes_from_data()
        if create_bars_from_elements:
            self.create_bars_from_data(material=bar_materials,area=bar_area)

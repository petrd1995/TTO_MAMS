import numpy as np
import cvxpy as cp
from time import time

class Optimization:
    def __init__(self, truss, min_radius=0.01, max_radius=10, vol_frac=0.5,min_r_inner=0.1e-3,max_r_inner=4e-3):
        self.truss = truss
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.vol_frac = vol_frac
        self.min_r_inner = min_r_inner
        self.max_r_inner = max_r_inner

    def efficient_sparse_optimality_criteria(self, 
                        maxiter=1000, 
                        miniter=5, 
                        max_area=None, 
                        min_area=1e-6, 
                        store_transposed_geometry_matrix=False,
                        transform=False,
                        axis='z',
                        angle=0):

        self.epsilon = 100
        self.epsilon_old = None
        self.epsilon_diff = None
        self.convergence = 0.01
        self.relative_change = None
        self.relative_threshold = 0.01  
        self.iteration = 0
        self.epsilon_history = []  

        start_time_opt = time.time()

        areas_method = 'numpy'                                      ##

        self.truss.create_global_dofs_and_force_vector()                  ##

        self.truss.geometry_matrix(store_transposed=store_transposed_geometry_matrix)
        self.truss.instantiate_E_lengths_vectors(areas_method=areas_method)

        directions = np.array([bar.direction_cosines for bar in self.truss.bars])
        directions = np.squeeze(directions, axis=-1)

        while self.iteration < maxiter:
            self.iteration += 1

            old_areas = self.truss.A_array.copy()

            start_time_iter = time.time()

            actual_volume = np.sum([bar.area * bar.length for bar in self.truss.bars])
            
            print(f'Iteration {self.iteration}, rel_change = {self.relative_change}, epsilon_diff = {self.epsilon_diff}, epsilon = {self.epsilon}, act/max = {actual_volume/self.truss.max_volume}')

            D = self.truss.diagonal_properties_matrix(areas_method=areas_method)

            self.truss.efficient_solve(write_to_nodes=True, diagonal_matrix=D)

            displacements = np.array([np.array(bar.node2.displacements) - np.array(bar.node1.displacements) for bar in self.truss.bars])

            inner_product = np.einsum('ij,ij->i', directions, displacements)
            inner_forces = self.truss.E_array * self.truss.A_array / self.truss.L_array * inner_product

            Af = (inner_forces**2)/(2 * self.truss.E_array)
            
            sq = np.sqrt(Af)
            if max_area is not None:
                areas = np.minimum((self.truss.max_volume * sq) / (sq @ self.truss.L_array), max_area)
            else:
                areas = (self.truss.max_volume * sq) / (sq @ self.truss.L_array)

            areas = np.maximum(areas, min_area)

            self.truss.A_array = areas        

            elapsed_time_iter  = time.time() - start_time_iter
            hours, rem = divmod(elapsed_time_iter, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Elapsed time iter: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

            self.epsilon_old = self.epsilon
            self.epsilon = np.max(np.abs(old_areas - areas))
            self.epsilon_history.append(self.epsilon)  

            if self.epsilon_old is not None:
                self.epsilon_diff = np.abs(self.epsilon - self.epsilon_old)
                self.relative_change = self.epsilon_diff / self.epsilon_old

            if self.iteration > miniter:

                if (self.epsilon < self.convergence) or (self.relative_change < self.relative_threshold):
                    for i, bar in enumerate(self.truss.bars):
                        bar.area = areas[i]
                    break

        for i, bar in enumerate(self.truss.bars):
            bar.area = areas[i]
            
        elapsed_time  = time.time() - start_time_opt
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time opt: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    def sdp(self, volume_fraction=0.5, solver=cp.MOSEK, path=None, path_ratio=0.1, A_inner=None, A_max=None, max_volume=None, verbose=True):
        
        areas_method = 'cvxpy'
        compliance = cp.Variable()
        areas = cp.Variable(self.truss.number_of_bars, nonneg=True)

        if max_volume is not None:
            self.truss.max_volume = max_volume
        else:
            self.truss.max_volume = np.ones(self.truss.number_of_bars)@np.array([bar.length for bar in self.truss.bars])
        self.truss.max_volume_fraction = volume_fraction * self.truss.max_volume

        self.truss.create_global_dofs_and_force_vector()

        self.truss.geometry_matrix()
        self.truss.instantiate_E_lengths_vectors(areas_method=areas_method, areas=areas)

        D = self.truss.diagonal_properties_matrix(areas_method=areas_method, areas=areas)
        self.truss.K = self.truss.efficient_stiffness_matrix(diagonal_matrix=D)

        if self.truss.global_forces.ndim == 1:
            self.truss.global_forces = self.truss.global_forces.reshape(-1, 1)

        compliance = cp.reshape(compliance, (1, 1))

        objective = cp.Minimize(compliance)
        semi_definite_constraint = cp.bmat([[compliance, -self.truss.global_forces.T], [-self.truss.global_forces, self.truss.K]])
        constraints = [semi_definite_constraint >> 0]
        if A_max is not None:
            constraints += [cp.sum(cp.multiply(areas, self.truss.L_array)) <= cp.sum(A_max*self.truss.L_array)*volume_fraction]
        else:
            constraints += [cp.sum(cp.multiply(areas, self.truss.L_array)) <= self.truss.max_volume_fraction]
        if path is not None:
            if A_inner is not None:
                for entry in path:
                    constraints += [areas[path] >= A_inner]
                print(f'A_inner: {A_inner}')
            else:
                for entry in path:
                    constraints += [areas[entry] >= path_ratio*cp.max(areas)]
                print(f'path_ratio: {path_ratio}')
        
        if A_max is not None:
            constraints += [areas <= A_max]

        self.prob = cp.Problem(objective, constraints)
        self.res = self.prob.solve(solver=solver, verbose=verbose)

        for en,bar in enumerate(self.truss.bars):
            bar.area = areas.value[en]

    @staticmethod
    def select(arr, target=None):
        min_1 = np.argmin(arr[:, 0])
        min_2 = np.argmin(arr[:, 1])
        if target is None:
            return min_1, min_2
        else:
            closest = np.argmin(np.sum((arr - target)**2, axis=1))
            return min_1, min_2, closest

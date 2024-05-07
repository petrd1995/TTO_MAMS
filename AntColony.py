import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, graph, main_graph, n_ants, n_iterations, alpha, beta, evaporation_rate, required_nodes_list=None, start_node=None, end_node=None, prescribed_start=None, prescribed_end=None, q=1, loop=False, plot_progress=False):
        self.graph = graph
        self.main_graph = main_graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.required_nodes_list = required_nodes_list
        self.prescribed_start = prescribed_start
        self.prescribed_end = prescribed_end

        if start_node is None:
            distances = nx.single_source_dijkstra_path_length(main_graph, prescribed_start)
            sorted_remaining_nodes = sorted([(dist, node) for node, dist in distances.items() if node in self.required_nodes_list], key=lambda x: (x[1] == end_node, x[0]))
            self.start_node = sorted_remaining_nodes[0][1]
        else:
            self.start_node = start_node
        
        if end_node is None:
            distances = nx.single_source_dijkstra_path_length(main_graph, prescribed_end)
            sorted_remaining_nodes = sorted([(dist, node) for node, dist in distances.items() if node in self.required_nodes_list], key=lambda x: (x[1] == end_node, x[0]))
            self.end_node = sorted_remaining_nodes[0][1]
        else:
            self.end_node = end_node

        self.q = q  
        self.pheromones = {edge: 1 for edge in self.graph.edges()}
        self.solutions = {}
        self.combined_paths = {}
        self.loop = loop
        self.plot_progress = plot_progress

    def deploy_ants(self):
        solutions = []
        for _ in range(self.n_ants):
            node = self.start_node if self.start_node else np.random.choice(list(self.graph.nodes()))
            visited = [node]
            while len(visited) < len(self.graph.nodes()):
                next_node = self.choose_next_node(visited[-1], visited)
                if next_node:
                    visited.append(next_node)
                else:
                    break
            if self.end_node and self.end_node not in visited:
                if self.end_node in list(self.graph.neighbors(visited[-1])):
                    visited.append(self.end_node)

            if self.loop:
                final_segment = (visited[-1], visited[0])
                if final_segment in self.graph.edges() or self.graph.has_edge(*reversed(final_segment)):
                    visited.append(visited[0])
                else:
                    LARGE_WEIGHT = 1e9
                    self.graph.add_edge(visited[-1], visited[0], weight=LARGE_WEIGHT)
                    visited.append(visited[0])

            solutions.append(visited)
        return solutions

    def choose_next_node(self, current_node, visited):
        if self.end_node and len(visited) == len(self.graph.nodes()) - 1 and self.end_node not in visited:
            return self.end_node
        neighbors = [n for n in self.graph.neighbors(current_node) if n not in visited and n != self.end_node]
        if not neighbors:
            return None
        probabilities = [self.probability(current_node, neighbor, visited) for neighbor in neighbors]
        probabilities = [p/sum(probabilities) for p in probabilities]
        return np.random.choice(neighbors, p=probabilities)

    def probability(self, current_node, next_node, visited):
        pheromone = self.pheromones.get((current_node, next_node), 1) ** self.alpha
        desirability = (1.0 / self.graph[current_node][next_node]['weight']) ** self.beta
        total = sum([(self.pheromones.get((current_node, n), 1) ** self.alpha) * 
                     (1.0 / self.graph[current_node][n]['weight']) ** self.beta for n in self.graph.neighbors(current_node) if n not in visited])
        return (pheromone * desirability) / total

    def update_pheromones(self, solutions):
        for solution in solutions:
            for i in range(len(solution) - 1):
                edge = (solution[i], solution[i+1])
                if edge not in self.pheromones:
                    self.pheromones[edge] = 1 
                self.pheromones[edge] += self.q / self.graph[solution[i]][solution[i+1]]['weight']

            if self.loop:
                final_segment = (solution[-1], solution[0])
                if final_segment in self.graph.edges():
                    if final_segment not in self.pheromones:
                        self.pheromones[final_segment] = 1
                    self.pheromones[final_segment] += self.q / self.graph[solution[-1]][solution[0]]['weight']
                else:
                    LARGE_WEIGHT = 1e9
                    self.graph.add_edge(solution[-1], solution[0], weight=LARGE_WEIGHT)
                    if final_segment not in self.pheromones:
                        self.pheromones[final_segment] = 1
                    self.pheromones[final_segment] += self.q / LARGE_WEIGHT
    
    def find_path_to_tsp(self, main_graph, start_node, tsp_start_node):
        if start_node in main_graph.nodes() and tsp_start_node in main_graph.nodes():
            if nx.has_path(main_graph, source=start_node, target=tsp_start_node):
                path = nx.shortest_path(main_graph, source=start_node, target=tsp_start_node, weight='weight')
                return path
            else:
                return None
        else:
            return None

    def setup_plot(self):        
        plt.ion() 
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Convergence Plot')
        return fig, ax, line

    def update_plot(self, fig, ax, line):
        line.set_ydata(self.best_fitness_over_time)
        line.set_xdata(range(len(self.best_fitness_over_time)))
        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)

    def run(self):
        best_solution = None
        best_length = float('inf')
        best_solution_combined = None
        self.best_fitness_over_time = []
        if self.plot_progress:
            fig,ax,line = self.setup_plot()

        for it in tqdm(range(self.n_iterations)):
            solutions = self.deploy_ants()
            self.update_pheromones(solutions)
            
            for solution in solutions:
                tsp_length = sum([self.graph[solution[i]][solution[i+1]]['weight'] for i in range(len(solution) - 1)])
                
                solution_key = f"solution_{len(self.solutions) + 1}"
                self.solutions[solution_key] = {
                    "path": solution,
                    "length": tsp_length,
                    "rank": None 
                }

                for config in [True, False]:
                    main_graph = self.main_graph.copy()
                    main_graph.remove_nodes_from(solution[1:-1])
                    if config:
                        path_to = self.find_path_to_tsp(main_graph, self.prescribed_start, solution[0])
                        if path_to is not None:
                            main_graph.remove_nodes_from(path_to)
                            path_from = self.find_path_to_tsp(main_graph, solution[-1], self.prescribed_end)
                    if path_to and path_from:
                        path_to_length = sum([self.main_graph[path_to[i]][path_to[i+1]]['weight'] for i in range(len(path_to) - 1)])
                        path_from_length = sum([self.main_graph[path_from[i]][path_from[i+1]]['weight'] for i in range(len(path_from) - 1)])
                        
                        total_length = tsp_length + path_to_length + path_from_length
                        combined_key = f"combined_path_{len(self.combined_paths) + 1}"
                        if config:
                            self.combined_paths[combined_key] = {
                                "path": path_to + solution[1:-1] + path_from,
                                "length": total_length,
                                "rank": None 
                            }
                        else:
                            self.combined_paths[combined_key] = {
                                "path": path_from + solution[1:-1] + path_to,
                                "length": total_length,
                                "rank": None 
                            }
                        
                        if total_length < best_length:
                            best_solution = solution
                            best_length = total_length
                            best_solution_combined = self.combined_paths[combined_key]["path"]
                            self.best_fitness_over_time.append(best_length)
                    else:
                        self.solutions[solution_key]["length"] = float('inf')
                    
                if self.plot_progress:
                    self.update_plot(fig,ax,line)

        if self.plot_progress:
            plt.ioff()
            plt.show()
        self.rank_solutions()
        self.rank_combined_paths()
        return best_solution_combined

    def rank_solutions(self):
        ranked_solutions = sorted(self.solutions.items(), key=lambda x: x[1]['length'])
        for rank, (key, value) in enumerate(ranked_solutions, start=1):
            self.solutions[key]['rank'] = rank

    def rank_combined_paths(self):
        ranked_paths = sorted(self.combined_paths.items(), key=lambda x: x[1]['length'])
        for rank, (key, value) in enumerate(ranked_paths, start=1):
            self.combined_paths[key]['rank'] = rank

    def get_solution_by_rank(self, rank):
        return next((v for k, v in self.solutions.items() if v['rank'] == rank), None)

    def get_combined_path_by_rank(self, rank):
        return next((v for k, v in self.combined_paths.items() if v['rank'] == rank), None)

    def restructure_by_rank(self):
        new_solutions = {}
        for key, value in self.solutions.items():
            new_key = f'solution_{value["rank"]}'
            new_solutions[new_key] = value
        self.solutions = new_solutions

        new_combined_paths = {}
        for key, value in self.combined_paths.items():
            new_key = f'combined_path_{value["rank"]}'
            new_combined_paths[new_key] = value
        self.combined_paths = new_combined_paths


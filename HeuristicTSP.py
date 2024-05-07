import networkx as nx

def custom_tsp(graph, required_nodes, current_node, end_node, path=None, removed_edges=None, discarded_nodes=None, start_node=None):
    if path is None:
        path = []
    if removed_edges is None:
        removed_edges = set()
    if discarded_nodes is None:
        discarded_nodes = set()
    if start_node is None: 
        start_node = current_node
    if current_node == end_node:
        return path

    if set(required_nodes) == {end_node}:
        graph_copy = graph.copy()

        graph_copy.remove_nodes_from(path[:-1])

        graph_copy.remove_edges_from([edge for edge in removed_edges if edge[0] != current_node and edge[1] != current_node])

        if nx.has_path(graph_copy, current_node, end_node):
            path_to_end_node = nx.dijkstra_path(graph_copy, current_node, end_node)
            return path + path_to_end_node[1:] 
        return None  

    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(path[:-1])
    graph_copy.remove_edges_from([edge for edge in removed_edges if edge[0] != current_node and edge[1] != current_node])
    
    distances = nx.single_source_dijkstra_path_length(graph_copy, current_node)
    sorted_remaining_nodes = sorted([(dist, node) for node, dist in distances.items() if node in required_nodes and node not in discarded_nodes and node != start_node], key=lambda x: (x[1] == end_node, x[0]))

    for _, next_node in sorted_remaining_nodes:
        if next_node not in path:
            if nx.has_path(graph_copy, current_node, next_node):
                path_to_next_node = nx.dijkstra_path(graph_copy, current_node, next_node)
                if current_node == start_node:
                    path_to_add = path_to_next_node[:]
                else:
                    path_to_add = path + path_to_next_node[1:]
                result_path = custom_tsp(graph, 
                                         [n for n in required_nodes if n != next_node], 
                                         next_node, 
                                         end_node, 
                                         path_to_add, 
                                         removed_edges.union(graph.edges(next_node)),
                                         discarded_nodes,
                                         start_node)
                if result_path:
                    return result_path
                else:
                    discarded_nodes.add(next_node)
    return None


import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_sorted_adjacency_matrix(graph, is_directed=False):
    sorted_nodes = sorted(set(node for edge in graph for node in edge[:2]))

    if is_directed:
        adjacency_matrix = np.zeros((len(sorted_nodes), len(sorted_nodes)))
        for edge in graph:
            i = sorted_nodes.index(edge[0])
            j = sorted_nodes.index(edge[1])
            weight = edge[2]
            adjacency_matrix[i][j] = weight
    else:
        adjacency_matrix = np.zeros((len(sorted_nodes), len(sorted_nodes)))
        for edge in graph:
            i = sorted_nodes.index(edge[0])
            j = sorted_nodes.index(edge[1])
            weight = edge[2]
            adjacency_matrix[i][j] = weight
            adjacency_matrix[j][i] = weight

    return adjacency_matrix


def generate_reachability_matrix(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    reach_matrix = np.array(adjacency_matrix)

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                reach_matrix[i][j] = reach_matrix[i][j] or (reach_matrix[i][k] and reach_matrix[k][j])

    np.fill_diagonal(reach_matrix, 1)
    return (reach_matrix > 0).astype(int)


def generate_mutual_reachability_matrix(adjacency_matrix):
    reach_matrix = generate_reachability_matrix(adjacency_matrix)
    mutual_reach_matrix = np.logical_and(reach_matrix, np.transpose(reach_matrix))

    np.fill_diagonal(mutual_reach_matrix, 1)
    return mutual_reach_matrix.astype(int)


def get_all_nodes(graph_data):
    g = nx.DiGraph()
    g.add_weighted_edges_from(graph_data)
    return sorted(list(g.nodes()))


def find_shortest_path(graph_data, start, end, allowed_nodes=None):
    g = nx.DiGraph()
    g.add_weighted_edges_from(graph_data)

    if allowed_nodes:
        g = g.subgraph(allowed_nodes)

    try:
        shortest_path = nx.shortest_path(g, source=start, target=end, weight='weight')
        distance = nx.shortest_path_length(g, source=start, target=end, weight='weight')
        if len(shortest_path) > 1:
            return [shortest_path[len(shortest_path) - 2], distance]
        return [shortest_path[0], distance]
    except nx.NetworkXNoPath:
        return ["-", "-"]
    except ValueError:
        return ["-", "-"]
    except nx.NodeNotFound:
        return ["-", "-"]


def create_shortest_paths(graph, start_node):
    nodes = get_all_nodes(graph)
    step = 0
    checked_nodes = sorted({start_node})
    list_steps = []
    while checked_nodes != nodes:
        distances = []
        paths = []
        for i in nodes:
            end_node = i
            allowed_nodes = checked_nodes.copy()
            allowed_nodes.append(end_node)
            path_dis = find_shortest_path(graph, start_node, end_node, allowed_nodes)
            if path_dis[0] == end_node:
                paths.append("-")
            else:
                paths.append(path_dis[0])
            distances.append(path_dis[1])

        distances = [float('inf') if isinstance(elem, str) else elem for elem in distances]
        sup_dis = distances.copy()
        for i in range(step + 1):
            sup_dis.remove(min(sup_dis))

        min_dis = min(sup_dis)
        min_index = distances.index(min_dis)
        add_node = nodes[min_index]
        distances = ['-' if isinstance(elem, float) else elem for elem in distances]
        list_steps.append([step, checked_nodes.copy(), add_node, min_dis, distances, paths])
        checked_nodes.append(add_node)
        checked_nodes.sort()
        step += 1

    return list_steps


def is_eulerians(graph, is_directed=False):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for edge in graph:
        out_degree[edge[0]] += 1
        in_degree[edge[1]] += 1

    odd_degree_count = sum(1 for vertex in set(in_degree.keys()) | set(out_degree.keys()) if
                           (in_degree[vertex] + out_degree[vertex]) % 2 != 0)

    is_eulerian = odd_degree_count == 0
    is_semi_eulerian = odd_degree_count == 2 if not is_directed else False

    return is_eulerian, is_semi_eulerian


def create_mst_steps(graph_edges, start_node):
    selected_nodes = {start_node}

    selected_edges = []

    while len(selected_nodes) < len(set(node for edge in graph_edges for node in edge[:2])):
        possible_edges = [edge for edge in graph_edges if
                          edge[0] in selected_nodes and edge[1] not in selected_nodes or
                          edge[1] in selected_nodes and edge[0] not in selected_nodes]

        min_edge = min(possible_edges, key=lambda x: x[2])
        selected_edges.append(min_edge)
        selected_nodes.add(min_edge[0] if min_edge[1] in selected_nodes else min_edge[1])

    mst_list = []
    addonlist = [start_node]

    for i in range(len(selected_edges)):
        element1 = str()
        for element in selected_edges[i][:2]:
            if element not in addonlist:
                element1 = element
        support_list = [i + 1, sorted(addonlist.copy()), selected_edges[i][:-1], element1, selected_edges[i][2]]
        mst_list.append(support_list)
        addonlist.append(element1)

    return mst_list


def strong_components_weighted(graph_edges):
    g = nx.DiGraph()
    g.add_weighted_edges_from(graph_edges)

    return sorted(list(nx.strongly_connected_components(g)))


def show_graph(graph, is_directed: bool):
    if is_directed:
        g = nx.DiGraph()
        g.add_weighted_edges_from(graph)

        pos = nx.circular_layout(g)
        labels = nx.get_edge_attributes(g, 'weight')

        nx.draw(g, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, font_color="red")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.title("Ориентированный граф")
        plt.show()
    else:
        g = nx.Graph()
        g.add_weighted_edges_from(graph)

        min_spanning_tree = nx.minimum_spanning_tree(g)

        colors = nx.coloring.greedy_color(g, strategy='largest_first')

        pos = nx.circular_layout(g)
        labels = nx.get_edge_attributes(g, 'weight')

        node_colors = [colors[node] for node in g.nodes()]
        edge_colors = ['red' if edge in min_spanning_tree.edges() else 'gray' for edge in g.edges()]
        nx.draw(g, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=12, font_color="red",
                edge_color=edge_colors)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.title("Неориентированный граф с минимальным остовным деревом")
        plt.show()


def format_string(input_str):
    pattern = re.compile(r"\(([^,]+),\s*([^,]+),\s*(\d+)\)")
    formatted_str = re.sub(pattern, r"('\1', '\2', \3)", input_str)
    formatted_str = re.sub(r"''", "'", formatted_str)
    return formatted_str


def validation_graph(test_graph: str):
    try:
        tuple_set = eval(test_graph)
        if isinstance(tuple_set, set):
            if all(isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str) and isinstance(item[1],
               str) and isinstance(item[2], int) for item in tuple_set):
                return True
    except Exception:
        pass

    return False

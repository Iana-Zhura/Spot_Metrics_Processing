import os
from bosdyn.api.graph_nav import map_pb2
import json
import networkx as nx

def load_map(path):
    """
    Load a graph from the given file path.
    """
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)
        print(f'Loaded graph with {len(current_graph.waypoints)} waypoints and {len(current_graph.anchoring.anchors)} anchors')
        return current_graph

def extract_anchor_map(graph):
    """
    Build a dictionary mapping anchor IDs to the 3D coordinates of the corresponding anchor.
    """
    anchor_map = {}
    for anchor in graph.anchoring.anchors:
        pos = anchor.seed_tform_waypoint.position
        anchor_map[anchor.id] = (pos.x, pos.y, pos.z)
    return anchor_map

def extract_anchor_ids(graph):
    """
    Return a list of anchor IDs in the order they appear in the graph.
    """
    anchor_ids = []
    for anchor in graph.anchoring.anchors:
        anchor_ids.append(anchor.id)
    return anchor_ids

def extract_edges_from_graph(graph):
    """
    Extract all edges from the graph.
    Returns a list of tuples: (from_anchor_id, to_anchor_id, cost)
    (The original cost is ignored and will be recalculated.)
    """
    edge_list = {}
    for edge in graph.edges:
        from_id = edge.id.from_waypoint
        to_id = edge.id.to_waypoint
        cost = edge.annotations.cost.value  # original cost (ignored in recalculation)
        edge_list[(from_id, to_id)] =  cost
    return edge_list

def scale_weights(original_weights, new_weights):
    """
    Scale new_weights so that their minimum and maximum match those of original_weights.
    Uses linear scaling.
    """
    orig_min, orig_max = min(original_weights), max(original_weights)
    new_min, new_max = min(new_weights), max(new_weights)
    scaled = []
    for w in new_weights:
        # Avoid division by zero in case all new weights are identical
        if new_max - new_min != 0:
            new_w = orig_min + ((w - new_min) / (new_max - new_min)) * (orig_max - orig_min)
        else:
            new_w = orig_min  # or any constant value within the desired range
        scaled.append(new_w)
    return scaled

#################
# shortest path #
#################
def order_waypoints_ids(graph):
    """
    Order waypoints based on the time when they were created.
    """
    ordered_waypoints = sorted(graph.waypoints, key=lambda wp: wp.annotations.creation_time.seconds)
    return [ow.id for ow in ordered_waypoints]

def print_order_waypoints_ids(graph):
    """
    Print each index (as key) and its corresponding waypoint ID (as value)
    """
    waypoint_ids = order_waypoints_ids(graph)
    for index, waypoint_id in enumerate(waypoint_ids):
        print(f"Key: {index}, Value: {waypoint_id}")

def find_lowest_cost_path_bidirectional_dijkstra(cost_graph, start_id, goal_id):
    """
    Find the path with the lowest cost using Bidirectional Dijkstra's algorithm.
    """
    try:
        dijstra_result = nx.bidirectional_dijkstra(cost_graph, source=start_id, target=goal_id, weight='weight')
        lowest_cost_path = dijstra_result[1]
        total_cost = dijstra_result[0]
        return lowest_cost_path, total_cost
    except nx.NetworkXNoPath:
        print(f"No path found between {start_id} and {goal_id}.")
        return None, float('inf')

def translate_wp_id_to_anchor_id(wp_id, mapping_file):
    """
    Translate a waypoint ID to the corresponding anchor ID using a saved JSON mapping.
    
    This function assumes that the JSON mapping file is structured with keys as anchor IDs
    (which are the desired outputs) and values as waypoint IDs.
    
    Args:
        wp_id (int or str): The waypoint ID to translate.
        mapping_file (str): Path to the JSON file containing the mapping.
    
    Returns:
        The corresponding anchor ID if found, or None if no matching waypoint is found.
    """
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Iterate over the mapping items: key = anchor_id, value = waypoint id
    for anchor_id, mapped_wp_id in mapping.items():
        if str(mapped_wp_id) == str(wp_id):
            return anchor_id
    return None

def translate_wp_id_path_to_anchor_id_path(wp_id_path, mapping_file):
    """
    Translate a path of waypoint IDs to a path of anchor IDs.
    """
    anchor_path = [translate_wp_id_to_anchor_id(wp_id, mapping_file) for wp_id in wp_id_path]
    return anchor_path

def build_cost_graph(edge_values):
    G = nx.DiGraph()
    for key, value in edge_values.items():
        from_id = key[0]
        to_id = key[1]
        weight = value
        G.add_edge(from_id, to_id, weight=weight)
        G.add_edge(to_id, from_id, weight=weight)
    return G


def main_shortest_path(graph_folder, start_id=None, goal_id=None):
    # 1) Load the graph from the folder
    graph = load_map(graph_folder)

    # 2) Load the edge data
    edge_data = extract_edges_from_graph(graph)

    # 3) Build the cost graph
    cost_graph = build_cost_graph(edge_data)

    # 4) Order the waypoints based on their creation time
    ordered_waypoints = order_waypoints_ids(graph)
    print_order_waypoints_ids(graph)

    # 5) Find the lowest cost path between the first and last waypoints
    if start_id is None or goal_id is None:
        start_id, goal_id = ordered_waypoints[0], ordered_waypoints[6]
    else:
        start_id = ordered_waypoints[start_id]
        goal_id = ordered_waypoints[goal_id]
    
    lowest_cost_path, total_cost = find_lowest_cost_path_bidirectional_dijkstra(cost_graph, start_id, goal_id)
    
    # lowest_cost_path_anchors = translate_wp_id_path_to_anchor_id_path(lowest_cost_path, id_mapping_path)
    lowest_const_string = "".join([str(wp_id)+" " for wp_id in lowest_cost_path])

    print(f"Lowest cost path: {lowest_cost_path}")
    print(f"Total cost: {total_cost}")
    print(f"Lowest cost path in string: {lowest_const_string}")

    # print(f"Lowest cost path in anchor IDs: {lowest_cost_path_anchors}")

def add_extra_edges(graph, edge_data):
    """
    Add extra edges to the graph based on the edge data.
    """

    new_from_id = "gabby-snake-GPHX1+oqSU0HJq.KSqKn5Q=="
    new_to_id = "skimpy-rodent-OEdIKag2mGSIuDu55.RM7A=="
    
    # Add the extra edges to the graph
    for key, value in edge_data.items():
        from_id = value["from"]
        to_id = value["to"]
        cost = value["raw_weight"]
        # Create a new edge with the given cost
        new_edge = graph.edges.add()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.annotations.cost.value = cost

    return graph


#########################
## new edge extra stuff #
#########################
import numpy as np

def find_waypoint_by_id(waypoints, waypoint_id):
    for wp in waypoints:
        if wp.id == waypoint_id:
            return wp
    return None

####

if __name__ == '__main__':
    # Update graph_folder to the folder containing 'graph' and 'edge_weights.txt'
    location = "greenhouse_march"
    graph_pth = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"

    if location == "greenhouse_final":
        from_index = 0
        to_index = 6

    elif location == "greenhouse_march":        
        from_index = 0
        to_index = 5


    id_mapping_pth = "/media/martin/Elements/ros-recordings/fit_sdk_odometry/greenhouse_march_id_map.json"
    # Shortest path
    main_shortest_path(graph_pth, from_index, to_index)

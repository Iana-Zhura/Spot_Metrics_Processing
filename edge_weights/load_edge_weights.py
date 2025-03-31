import os
from bosdyn.api.graph_nav import map_pb2
import json
import networkx as nx
import numpy as np

############################
# Load/Save Graph Helpers #
############################

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

def save_graph(graph, path):
    """
    Save the updated graph to a new file.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)
    
    out_path = os.path.join(path, 'graph')
    with open(out_path, 'wb') as f:
        f.write(graph.SerializeToString())
    print(f"New graph saved to {out_path}")

##############################
# Edge & Weighting Helpers  #
##############################
def extract_edges_from_graph(graph):
    """
    Extract all edges from the graph.
    Returns a list of tuples: (from_anchor_id, to_anchor_id, cost)
    (The original cost is ignored and will be recalculated.)
    """
    edge_list = []
    for edge in graph.edges:
        from_id = edge.id.from_waypoint
        to_id = edge.id.to_waypoint
        cost = edge.annotations.cost.value  # original cost (ignored in recalculation)
        edge_list.append((from_id, to_id, cost))
    return edge_list

def load_new_edge_weights(edge_weights_pth):
    """
    Load new edge weights from a JSON file.
    Expects the JSON to have keys formatted as "from -> to" and a dictionary value with keys:
    "from", "to", "raw_weight", "normalized_weight", and "mean_time_spent".
    Returns a dictionary mapping (from, to) tuples to raw_weight.
    """
    with open(edge_weights_pth, 'r') as f:
        data = json.load(f)
    
    new_weights = {}
    for edge_key, edge_data in data.items():
        # Extract the from and to node identifiers (these should match the graph's edge IDs)
        from_id = edge_data["from"]
        to_id = edge_data["to"]
        # We use raw_weight as the new weight
        new_weights[(from_id, to_id)] = edge_data["raw_weight"]
    
    return new_weights

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
# main function #
#################

def main(graph_folder, edge_weights_pth, updated_graph_folder):
    
    # 0) check if updated_graph_folder exists
    if not os.path.exists(updated_graph_folder):
        os.makedirs(updated_graph_folder)
        print(f"Created directory: {updated_graph_folder}")

    # 1) Load the graph from the folder
    graph = load_map(graph_folder)

    # 2) Read the current edge weights
    original_edges = extract_edges_from_graph(graph)
    original_weights = [edge[2] for edge in original_edges]
    print("Original edge weights:", original_weights)

    # 3) Load new edge weights (JSON format, as a dict mapping (from, to) to raw_weight)
    new_weights_dict = load_new_edge_weights(edge_weights_pth)

    # Build a new weight list corresponding to the order of graph.edges.
    new_weight_list = []
    for edge in graph.edges:
        key = (edge.id.from_waypoint, edge.id.to_waypoint)
        if key in new_weights_dict:
            new_weight_list.append(new_weights_dict[key])
        else:
            print(f"Warning: Edge {key} not found in the new weights file; using original weight.")
            new_weight_list.append(edge.annotations.cost.value)

    # 4) Scale the new edge weights to the same range as the original weights
    scaled_weights = scale_weights(original_weights, new_weight_list)
    print("Scaled new edge weights:", scaled_weights)

    # 5) Enter the new edge weights into the graph
    for edge, new_cost in zip(graph.edges, scaled_weights):
        edge.annotations.cost.value = new_cost

    # 6) Save the updated graph
    save_graph(graph, updated_graph_folder)


if __name__ == '__main__':
    # Update graph_folder to the folder containing 'graph' and 'edge_weights.txt'
    location = "greenhouse_very_final"
    
    if location == "greenhouse_final":
        graph_pth = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph/"
        updated_graph_pth = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        edge_weights_pth = "/media/martin/Elements/ros-recordings/edge_weights/greenhouse_final_edge_values.json"

    elif location == "greenhouse_march":
        graph_pth = "/media/martin/Elements/ros-recordings/recordings/march_11/downloaded_graph/"
        updated_graph_pth = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        edge_weights_pth = "/media/martin/Elements/ros-recordings/edge_weights/greenhouse_march_edge_values.json"
    
    elif location == "greenhouse_very_final":
        graph_pth = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph/"
        updated_graph_pth = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/updated_graph/"
        edge_weights_pth = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/edge_weights/greenhouse_edge_values.json"

    main(graph_pth, edge_weights_pth, updated_graph_pth)


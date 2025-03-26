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
    """
    Build a directed cost graph using the computed edge values.
    
    Parameters:
        edge_values (dict): Dictionary mapping "from_id -> to_id" to a dict with keys:
            - "from": starting anchor ID
            - "to": ending anchor ID
            - "raw_weight": the computed raw edge weight
            - "normalized_weight": the normalized edge weight (between 0 and 1)
            - "mean_time_spent": the mean time spent at the two endpoints (optional)
    
    Returns:
        G (nx.DiGraph): A directed graph with bidirectional edges, where each edge has:
            - weight: the normalized edge weight
            - mean_time_spent: mean time spent at the edge endpoints
    """
    G = nx.DiGraph()
    for key, value in edge_values.items():
        from_id = value["from"]
        to_id = value["to"]
        weight = value["normalized_weight"]
        mean_time = value.get("mean_time_spent", None)
        # Add edge in the forward direction.
        if mean_time is not None:
            G.add_edge(from_id, to_id, weight=weight, mean_time_spent=mean_time)
            G.add_edge(to_id, from_id, weight=weight, mean_time_spent=mean_time)
        else:
            G.add_edge(from_id, to_id, weight=weight)
            G.add_edge(to_id, from_id, weight=weight)
    return G

def load_edge_data(edge_data_pth):
    """
    Load the edge dat afrom assing_edge_weights.py.
    """
    with open(edge_data_pth, 'r') as f:
        data = json.load(f)
    return data

def main_shortest_path(graph_folder, edge_data_pth, id_mapping_path, start_id=None, goal_id=None):
    # 1) Load the graph from the folder
    graph = load_map(graph_folder)

    # 2) Load the edge data
    edge_data = load_edge_data(edge_data_pth)

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

def pose_to_matrix_ko(ko_pose):
    # Convert the waypoint_tform_ko pose to a 4x4 matrix.
    position = ko_pose.position
    rotation = ko_pose.rotation
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [position.x, position.y, position.z]
    return T

def compute_relative_transform_waypoints(wp_from, wp_to):
    T_from = pose_to_matrix_ko(wp_from.waypoint_tform_ko)
    T_to   = pose_to_matrix_ko(wp_to.waypoint_tform_ko)
    T_edge = invert_transform(T_from) @ T_to
    translation = T_edge[:3, 3].tolist()
    rotation = matrix_to_quaternion(T_edge)
    return {
        "position": {"x": translation[0], "y": translation[1], "z": translation[2]},
        "rotation": rotation
    }

def compute_relative_transform_waypoints(wp_from, wp_to):
    # Compute the transform using the waypoint_tform_ko values.
    T_from = pose_to_matrix_ko(wp_from.waypoint_tform_ko)
    T_to   = pose_to_matrix_ko(wp_to.waypoint_tform_ko)
    T_edge = invert_transform(T_from) @ T_to
    translation = T_edge[:3, 3].tolist()
    rotation = matrix_to_quaternion(T_edge)
    return {
        "position": {"x": translation[0], "y": translation[1], "z": translation[2]},
        "rotation": rotation
    }


def invert_transform(T):
    """
    Invert a 4x4 transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def matrix_to_quaternion(T):
    """
    Extract a quaternion (x, y, z, w) from a 4x4 transformation matrix.
    """
    R = T[:3, :3]
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return {"x": x, "y": y, "z": z, "w": w}

####


#################
# main function #
#################

def main(graph_folder, edge_weights_pth, updated_graph_folder):
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

    # 5) Save the new edge weights into the graph
    for edge, new_cost in zip(graph.edges, scaled_weights):
        edge.annotations.cost.value = new_cost

    # 6) add extra edges
    """
    new_edge = dict()
    new_edge["id"] = {
        "from_waypoint": "gabby-snake-GPHX1+oqSU0HJq.KSqKn5Q==",
        "to_waypoint": "skimpy-rodent-OEdIKag2mGSIuDu55.RM7A=="
    }
    new_edge["snapshot_id"] = None
    new_edge["from_tform_to"] = {
        "postion": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        },
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
        }
    }
    new_edge["annotations"] = {
        "direction_constraint": DIRECTION_CONSTRAINT_NONE,
        "flat_ground": {},
        "cost": {
            "value": 0.0
        },
        "edge_source": EDGE_SOURCE_ODOMETRY
    }
    """
    # take the first edge as a template -> CAREFUL: THE TRANSFORMS ARE NOT CORRECT
    # make it an independent copy


    # add extra greenhouse edge using waypoint poses
    if False:
        import copy

        new_edge = copy.deepcopy(graph.edges[0])

        # Update the new edge's waypoint IDs.
        new_edge.id.from_waypoint = "gabby-snake-GPHX1+oqSU0HJq.KSqKn5Q=="
        new_edge.id.to_waypoint   = "skimpy-rodent-OEdIKag2mGSIuDu55.RM7A=="

        # Update the cost value.
        new_edge.annotations.cost.value = 0.40510477245859056

        # Use waypoints (the ones that are actually plotted) for the transformation.
        wp_from = find_waypoint_by_id(graph.waypoints, new_edge.id.from_waypoint)
        wp_to   = find_waypoint_by_id(graph.waypoints, new_edge.id.to_waypoint)
        if wp_from is None or wp_to is None:
            raise ValueError("Could not find one of the specified waypoints in graph.waypoints")
        else:
            # Compute the relative transform using waypoint_tform_ko values.
            relative_transform = compute_relative_transform_waypoints(wp_from, wp_to)
            # Update the edge's transformation field by modifying each subfield.
            new_edge.from_tform_to.position.x = relative_transform["position"]["x"]
            new_edge.from_tform_to.position.y = relative_transform["position"]["y"]
            new_edge.from_tform_to.position.z = relative_transform["position"]["z"]

            new_edge.from_tform_to.rotation.x = relative_transform["rotation"]["x"]
            new_edge.from_tform_to.rotation.y = relative_transform["rotation"]["y"]
            new_edge.from_tform_to.rotation.z = relative_transform["rotation"]["z"]
            new_edge.from_tform_to.rotation.w = relative_transform["rotation"]["w"]

        # Append the new edge to the graph.
        graph.edges.append(new_edge)

    # 7) Save the updated graph
    save_graph(graph, updated_graph_folder)


if __name__ == '__main__':
    # Update graph_folder to the folder containing 'graph' and 'edge_weights.txt'
    location = "greenhouse_march"
    
    if location == "greenhouse_final":
        graph_pth = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph/"
        updated_graph_pth = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        edge_weights_pth = "/media/martin/Elements/ros-recordings/edge_weights/greenhouse_final_edge_values.json"
        from_index = 0
        to_index = 6

    elif location == "greenhouse_march":
        graph_pth = "/media/martin/Elements/ros-recordings/recordings/march_11/downloaded_graph/"
        updated_graph_pth = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        edge_weights_pth = "/media/martin/Elements/ros-recordings/edge_weights/greenhouse_march_edge_values.json"
        
        from_index = 0
        to_index = 5

    if True:
        main(graph_pth, edge_weights_pth, updated_graph_pth)
    else:
        id_mapping_pth = "/media/martin/Elements/ros-recordings/fit_sdk_odometry/greenhouse_march_id_map.json"
        # Shortest path
        main_shortest_path(graph_pth, edge_weights_pth, id_mapping_pth, from_index, to_index)

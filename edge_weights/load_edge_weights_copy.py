import os
import json
import numpy as np
import networkx as nx
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.recording import GraphNavRecordingServiceClient

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
        print(f'Loaded graph with {len(current_graph.waypoints)} waypoints and '
              f'{len(current_graph.anchoring.anchors)} anchors')
        return current_graph

def save_graph(graph, path):
    """
    Save the updated graph to a new file named 'graph'.
    """
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
    Returns a list of tuples: (from_waypoint_id, to_waypoint_id, cost).
    """
    edge_list = []
    for edge in graph.edges:
        from_id = edge.id.from_waypoint
        to_id   = edge.id.to_waypoint
        cost    = edge.annotations.cost.value
        edge_list.append((from_id, to_id, cost))
    return edge_list

def load_new_edge_weights(edge_weights_pth):
    """
    Load new edge weights from a JSON file. 
    The JSON should map (from->to) to a dict with 'from', 'to', and 'raw_weight'.
    """
    with open(edge_weights_pth, 'r') as f:
        data = json.load(f)
    new_weights = {}
    for edge_key, edge_data in data.items():
        from_id = edge_data["from"]
        to_id   = edge_data["to"]
        new_weights[(from_id, to_id)] = edge_data["raw_weight"]
    return new_weights

def scale_weights(original_weights, new_weights):
    """
    Linearly scale new_weights to the range of original_weights.
    """
    orig_min, orig_max = min(original_weights), max(original_weights)
    new_min, new_max   = min(new_weights), max(new_weights)
    scaled = []
    for w in new_weights:
        if new_max - new_min != 0:
            new_w = orig_min + ((w - new_min) / (new_max - new_min)) * (orig_max - orig_min)
        else:
            new_w = orig_min
        scaled.append(new_w)
    return scaled

###############################
# Waypoint & Transform Helpers
###############################

def find_waypoint_by_id(waypoints, waypoint_id):
    for wp in waypoints:
        if wp.id == waypoint_id:
            return wp
    return None

def pose_to_matrix_ko(ko_pose):
    """
    Convert the waypoint_tform_ko pose to a 4x4 matrix.
    """
    position = ko_pose.position
    rotation = ko_pose.rotation
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [position.x, position.y, position.z]
    return T

def invert_transform(T):
    """
    Invert a 4x4 transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv

def matrix_to_quaternion(T):
    """
    Extract a quaternion (x, y, z, w) from a 4x4 transformation matrix.
    """
    R = T[:3, :3]
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return {"x": x, "y": y, "z": z, "w": w}

def compute_relative_transform_from_anchors(graph, from_wp_id, to_wp_id):
    """
    Compute the relative transform between two waypoints using their corresponding anchors.
    Assumes that the ordering of graph.anchoring.anchors corresponds to that of graph.waypoints.
    
    Args:
        graph: The full graph protobuf.
        from_wp_id: The ID of the starting waypoint.
        to_wp_id: The ID of the destination waypoint.
        
    Returns:
        A dict with keys "position" and "rotation" (a quaternion), computed as:
        T_edge = inverse(T_anchor_from) @ T_anchor_to.
    """
    # Find indices of the waypoints.
    from_index = None
    to_index = None
    for i, wp in enumerate(graph.waypoints):
        if wp.id == from_wp_id:
            from_index = i
        if wp.id == to_wp_id:
            to_index = i
    if from_index is None or to_index is None:
        raise ValueError("One or both waypoint IDs not found in the graph.")
    
    # Get corresponding anchors.
    anchor_from = graph.anchoring.anchors[from_index]
    anchor_to   = graph.anchoring.anchors[to_index]
    
    # Convert the seed_tform_waypoint from each anchor to a 4x4 matrix.
    T_from = pose_to_matrix_ko(anchor_from.seed_tform_waypoint)
    T_to   = pose_to_matrix_ko(anchor_to.seed_tform_waypoint)
    
    # Compute the relative transform.
    T_edge = invert_transform(T_from) @ T_to
    translation = T_edge[:3, 3].tolist()
    rotation = matrix_to_quaternion(T_edge)
    return {
        "position": {"x": translation[0], "y": translation[1], "z": translation[2]},
        "rotation": rotation
    }

from bosdyn.client.math_helpers import SE3Pose

def compute_relative_transform_from_anchors_sdk(graph, from_wp_id, to_wp_id):
    """
    Compute the relative transform between two waypoints using the corresponding anchor transforms.
    Assumes the ordering of graph.anchoring.anchors corresponds to that of graph.waypoints.
    
    Returns:
        A geometry_pb2.SE3Pose protobuf message representing the transform from the 'from' anchor's
        frame to the 'to' anchor's frame.
    """
    # Find the indices corresponding to the waypoint IDs.
    from_index = None
    to_index = None
    for i, wp in enumerate(graph.waypoints):
        if wp.id == from_wp_id:
            from_index = i
        if wp.id == to_wp_id:
            to_index = i
    if from_index is None or to_index is None:
        raise ValueError("One or both waypoint IDs not found in the graph.")

    # Get the corresponding anchors.
    anchor_from = graph.anchoring.anchors[from_index]
    anchor_to   = graph.anchoring.anchors[to_index]
    
    # Convert the anchor SE(3) poses from proto to SE3Pose objects.
    pose_from = SE3Pose.from_proto(anchor_from.seed_tform_waypoint)
    pose_to   = SE3Pose.from_proto(anchor_to.seed_tform_waypoint)
    
    # Compute the relative transform: T_edge = inverse(pose_from) * pose_to.
    relative_pose = pose_from.inverse().mult(pose_to)
    
    # Return the computed relative transform as a protobuf.
    return relative_pose.to_proto()

###########################
# Offline Edge Creation   #
###########################

def generate_new_edge(graph, from_id, to_id, cost):
    """
    Offline approach: create a brand-new edge using anchor transforms.
    """
    wp_from = find_waypoint_by_id(graph.waypoints, from_id)
    wp_to   = find_waypoint_by_id(graph.waypoints, to_id)
    if wp_from is None or wp_to is None:
        raise ValueError("Could not find one of the specified waypoints in graph.waypoints.")
    
    # Instead of using waypoint_tform_ko, use the anchors.
    relative_transform = compute_relative_transform_from_anchors(graph, from_id, to_id)
    
    new_edge = map_pb2.Edge()
    new_edge.id.from_waypoint = from_id
    new_edge.id.to_waypoint   = to_id
    new_edge.annotations.cost.value = cost

    new_edge.from_tform_to.position.x = relative_transform["position"]["x"]
    new_edge.from_tform_to.position.y = relative_transform["position"]["y"]
    new_edge.from_tform_to.position.z = relative_transform["position"]["z"]
    new_edge.from_tform_to.rotation.x = relative_transform["rotation"]["x"]
    new_edge.from_tform_to.rotation.y = relative_transform["rotation"]["y"]
    new_edge.from_tform_to.rotation.z = relative_transform["rotation"]["z"]
    new_edge.from_tform_to.rotation.w = relative_transform["rotation"]["w"]
    
    return new_edge


###########################
# Live (RPC) Edge Creation
###########################

def rpc_create_edge(graph, from_id, to_id):
    """
    Live approach: uses GraphNavRecordingServiceClient to create an edge,
    but now computes the relative transform using anchors.
    """
    client = GraphNavRecordingServiceClient()
    
    wp_from = find_waypoint_by_id(graph.waypoints, from_id)
    wp_to   = find_waypoint_by_id(graph.waypoints, to_id)
    if wp_from is None or wp_to is None:
        raise ValueError("Specified waypoint(s) not found in the graph.")
    
    relative_transform = compute_relative_transform_from_anchors(graph, from_id, to_id)
    
    new_edge = client.make_edge(
        from_waypoint_id=from_id,
        to_waypoint_id=to_id,
        from_tform_to=relative_transform
    )
    return new_edge

from bosdyn.api.graph_nav import map_pb2

def generate_new_edge_using_sdk(graph, from_id, to_id, cost):
    """
    Create a new edge using the SDK's transformation functions from the anchors.
    """
    # Compute the relative transform using the SDK's SE3Pose methods.
    relative_transform = compute_relative_transform_from_anchors_sdk(graph, from_id, to_id)
    
    # Create a new Edge message.
    new_edge = map_pb2.Edge()
    new_edge.id.from_waypoint = from_id
    new_edge.id.to_waypoint   = to_id
    new_edge.annotations.cost.value = cost

    # Directly assign the computed SE3Pose (relative_transform) to the edge.
    new_edge.from_tform_to.CopyFrom(relative_transform)
    
    return new_edge

###########################
# Main Example
###########################

def main(graph_folder, edge_weights_pth, updated_graph_folder):
    # 1) Load the graph
    graph = load_map(graph_folder)

    # 2) Read the current edge weights
    original_edges = extract_edges_from_graph(graph)
    original_weights = [e[2] for e in original_edges]
    print("Original edge weights:", original_weights)

    # 3) Load new edge weights from JSON
    new_weights_dict = load_new_edge_weights(edge_weights_pth)

    # 4) Build a new weight list in the order of the existing edges
    new_weight_list = []
    for edge in graph.edges:
        key = (edge.id.from_waypoint, edge.id.to_waypoint)
        if key in new_weights_dict:
            new_weight_list.append(new_weights_dict[key])
        else:
            print(f"Warning: Edge {key} not found in the new weights file; using original weight.")
            new_weight_list.append(edge.annotations.cost.value)

    # 5) Scale them to the same range as the original weights
    scaled_weights = scale_weights(original_weights, new_weight_list)
    print("Scaled new edge weights:", scaled_weights)

    # 6) Apply the scaled weights to the graph
    for edge, new_cost in zip(graph.edges, scaled_weights):
        edge.annotations.cost.value = new_cost

    # 7) Example of adding a new edge (OFFLINE):
    from_waypoint_id = "gabby-snake-GPHX1+oqSU0HJq.KSqKn5Q=="
    to_waypoint_id   = "skimpy-rodent-OEdIKag2mGSIuDu55.RM7A=="
    extra_cost       = 0.40510477245859056

    # If you're doing this offline, do:
    #new_edge = generate_new_edge(graph, from_waypoint_id, to_waypoint_id, extra_cost)
    #new_edge = rpc_create_edge(graph, from_waypoint_id, to_waypoint_id)
    new_edge = generate_new_edge_using_sdk(graph, from_waypoint_id, to_waypoint_id, extra_cost)

    graph.edges.append(new_edge)
    print(f"Appended offline edge from {from_waypoint_id} to {to_waypoint_id}.")

    # If you wanted to do it live (during an active recording session), you would call:
    # live_edge = rpc_create_edge(graph, from_waypoint_id, to_waypoint_id)
    # and then pass that to client.create_edge(...) with a lease. 
    # But for offline usage, we just skip that.

    # 8) Save the updated graph
    save_graph(graph, updated_graph_folder)

if __name__ == "__main__":
    graph_pth         = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph/"
    updated_graph_pth = "/media/martin/spot_extern/martin/new_graphs/greenhouse_final/edge_by_sdk"
    edge_weights_pth  = "/media/martin/Elements/ros-recordings/edge_weights/greenhouse_final_edge_values.json"

    main(graph_pth, edge_weights_pth, updated_graph_pth)

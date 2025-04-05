import os
import json
import numpy as np
import matplotlib.pyplot as plt
from bosdyn.api.graph_nav import map_pb2
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import networkx as nx

import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

BASE_PTH = os.getcwd()

#############################
# Helper Functions
#############################
def load_transformation_params(trans_json_path):
    """
    Load transformation parameters from a JSON file."
    """
    import json
    # Check if the file exists
    
    with open(trans_json_path, 'r') as f:
        loaded_json = json.load(f)
    
    # Check if the loaded JSON has the expected keys
    required_keys = ["rotation_z", "rotation_y", "translation"]
    for key in required_keys:
        if key not in loaded_json:
            raise KeyError(f"Missing key '{key}' in the loaded JSON.")
    
    return loaded_json

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
        cost = edge.annotations.cost.value  # original cost (ignored)
        edge_list.append((from_id, to_id, cost))
    return edge_list

def load_metrics(metrics_file_path, anchor_ids=None, metric_key="average_effort"):
    """
    Load metrics from a JSON file exported from previous processing.
    
    Expected structure (numeric keys):
      {
        "0": { "waypoint_id": "<wp_id1>", "energy": ..., "distance": ..., "average_effort": ..., "time_spent": ... },
        "1": { "waypoint_id": "<wp_id2>", ... },
        ...
      }
    
    This function builds a dictionary mapping each waypoint (anchor) ID to its weight,
    using the value corresponding to metric_key.
    """
    with open(metrics_file_path, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    try:
        first_key = next(iter(data))
        int(first_key)
        numeric_keys = True
    except (ValueError, TypeError):
        numeric_keys = False

    if numeric_keys and anchor_ids is not None:
        # this is the default case
        # sort the keys numerically and assign weights to the corresponding anchor IDs
        for key in sorted(data.keys(), key=lambda k: int(k)):
            info = data[key]
            weight = float(info.get(metric_key, 0))
            anchor_id = info.get("waypoint_id")
            if anchor_id is not None:
                metrics[anchor_id] = weight
            else:
                idx = int(key)
                if idx < len(anchor_ids):
                    metrics[anchor_ids[idx]] = weight
    else:
        for waypoint_key, info in data.items():
            anchor_id = info.get("waypoint_id")
            weight = float(info.get(metric_key, 0))
            if weight == 0:
                print(f"Warning: Weight for {waypoint_key} is 0. Very likely because the metric key was not present in the JSON.")
            if anchor_id is not None:
                metrics[anchor_id] = weight
    return metrics

def calc_edge_value(anchor_id1, anchor_id2, metrics, pos1, pos2):
    """
    Calculate the raw edge weight as the mean of the two anchors' metric values plus
    the Euclidean distance between their positions.
    """
    metric_mean = (metrics.get(anchor_id1, 0) + metrics.get(anchor_id2, 0)) / 2.0
    edge_length = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return metric_mean + edge_length

def normalize_edge_weights(raw_edges):
    """
    Normalize raw edge weights to the range [0,1].
    
    Args:
        raw_edges (list): List of tuples (from_id, to_id, raw_weight)
        
    Returns:
        list: List of tuples (from_id, to_id, normalized_weight)
    """
    if not raw_edges:
        return []
    raw_values = [w for _, _, w in raw_edges]
    min_val = min(raw_values)
    max_val = max(raw_values)

    print(f"Min: {min_val}, Max: {max_val}")

    normalized_edges = []
    for from_id, to_id, raw_weight in raw_edges:
        if max_val > min_val:
            norm_weight = (raw_weight - min_val) / (max_val - min_val)
        else:
            norm_weight = 0
        normalized_edges.append((from_id, to_id, norm_weight))
    return normalized_edges

def apply_transformations(coords, rotation_z, rotation_y, translation):
    """
    Apply rotation and translation transformations to the coordinates.
    """
    theta_z = np.radians(rotation_z)
    theta_y = np.radians(rotation_y)
    rotation_matrix_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z),  np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    rotation_matrix_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]])
    translation_matrix = np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]])
    coords_homogeneous = np.hstack((coords, np.ones((coords.shape[0], 1))))
    coords_transformed = (rotation_matrix_y @ translation_matrix @ rotation_matrix_z @ coords_homogeneous.T).T[:, :3]
    return coords_transformed

def transform_anchor_map(anchor_map, rotation_z, rotation_y, translation):
    """
    Apply the specified transformation to all anchor coordinates.
    """
    keys = list(anchor_map.keys())
    coords = np.array(list(anchor_map.values()))
    transformed_coords = apply_transformations(coords, rotation_z, rotation_y, translation)
    transformed_map = {key: tuple(transformed_coords[i]) for i, key in enumerate(keys)}
    return transformed_map

def set_axes_equal(ax):
    """
    Set equal scaling for the 3D axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#############
## plot 2D ##
#############
def plot_edge_values_2d(edge_data, transformed_anchor_map, waypoint_to_numeric, metrics):
    """
    Plot edges between anchors in 2D.
    
    For each anchor:
      - Plot the (x,y) location (ignoring z).
      - Annotate with the first 10 characters of its ID.
    For each edge:
      - Draw a line between the anchors.
      - Annotate the midpoint with the normalized edge weight.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot anchors with truncated IDs.
    for wp_id, coord in transformed_anchor_map.items():
        x, y = coord[0], coord[1]
        truncated_id = wp_id[:10]  # Only first 10 characters.
        ax.scatter(x, y, color='red', s=50)
        ax.text(x, y, truncated_id, fontsize=8, color='black',
                ha='right', va='bottom')
    
    # Plot edges and annotate with normalized weights.
    for key, value in edge_data.items():
        from_id = value["from"]
        to_id = value["to"]
        norm_weight = value["normalized_weight"]
        pt1 = transformed_anchor_map[from_id]
        pt2 = transformed_anchor_map[to_id]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='gray', alpha=0.5)
        midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        ax.text(midpoint[0], midpoint[1], f"{norm_weight:.2f}",
                fontsize=7, color='blue', ha='center', va='center')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Edge Weights with Truncated Anchor IDs')
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.show()

#############################
# Core Functions
#############################
def compute_edge_values(graph, metrics_file, rotation_z=0, rotation_y=0, translation=(0,0,0), metric_key="average_effort"):
    """
    Compute raw and normalized edge weights between anchors.
    
    Steps:
      1. Load anchor IDs, anchor positions, and metrics.
      2. Transform anchor positions.
      3. For each edge, compute raw weight as:
           raw_weight = (average metric of the two anchors) + (Euclidean distance between anchors)
      4. Normalize all raw weights to [0,1].
      5. Additionally, compute mean time spent at the two endpoints.
      
    Returns:
        export_data (dict): Dictionary mapping "from -> to" to a dictionary containing:
             - "from": from_id
             - "to": to_id
             - "raw_weight": computed raw weight
             - "normalized_weight": normalized weight (0 to 1)
             - "mean_time_spent": average of the two endpoints' time_spent
        transformed_anchor_map (dict): Transformed anchor positions.
        waypoint_to_numeric (dict): Mapping of each waypoint_id (from the JSON) to its original numerical key.
    """
    anchor_ids = extract_anchor_ids(graph)
    anchor_map = extract_anchor_map(graph)
    transformed_anchor_map = transform_anchor_map(anchor_map, rotation_z, rotation_y, translation)
    metrics = load_metrics(metrics_file, anchor_ids, metric_key=metric_key)
    
    # Load raw metrics JSON to get original numerical keys and time_spent values.
    with open(metrics_file, 'r') as f:
        raw_json = json.load(f)
    waypoint_to_numeric = {}
    time_spent_map = {}
    for key, info in raw_json.items():
        wp = info.get("waypoint_id")
        if wp is not None:
            waypoint_to_numeric[wp] = key
            time_spent_map[wp] = float(info.get("time_spent", 0))
    
    edges = extract_edges_from_graph(graph)
    raw_edges = []
    for from_id, to_id, _ in edges:
        if from_id in transformed_anchor_map and to_id in transformed_anchor_map:
            pos1 = transformed_anchor_map[from_id]
            pos2 = transformed_anchor_map[to_id]
            raw_weight = calc_edge_value(from_id, to_id, metrics, pos1, pos2)
            raw_edges.append((from_id, to_id, raw_weight))
    normalized_edges = normalize_edge_weights(raw_edges)
    
    export_data = {}
    for (from_id, to_id, raw_weight), (_, _, norm_weight) in zip(raw_edges, normalized_edges):
        # Compute mean time spent from the two endpoints (if available).
        time1 = time_spent_map.get(from_id, 0)
        time2 = time_spent_map.get(to_id, 0)
        mean_time = (time1 + time2) / 2.0
        key = f"{from_id} -> {to_id}"
        export_data[key] = {
            "from": from_id,
            "to": to_id,
            "raw_weight": raw_weight,
            "normalized_weight": norm_weight,
            "mean_time_spent": mean_time
        }
    return export_data, transformed_anchor_map, waypoint_to_numeric

def export_edge_values_to_file(edge_data, output_file):
    """
    Export computed edge values to a JSON file.
    """
    # if folder does not exist, create it
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # if file exists, overwrite it
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Overwriting.")
        os.remove(output_file)

    with open(output_file, "w") as f:
        json.dump(edge_data, f, indent=2)
    print(f"Exported edge values to {output_file}")

def plot_edge_values(edge_data, transformed_anchor_map, waypoint_to_numeric, metrics):
    """
    Plot edges between anchors with labels.
    
    Each anchor displays:
      - Its waypoint ID (from metrics JSON) at its location.
      - Its metric weight (displayed above).
      - The original numerical anchor id (displayed below).
    Each edge is annotated with its normalized weight.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs, ys, zs = [], [], []
    for wp_id, coord in transformed_anchor_map.items():
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])
        ax.text(coord[0], coord[1], coord[2], wp_id, fontsize=8, color='black')
        weight = metrics.get(wp_id, 0)
        ax.text(coord[0], coord[1], coord[2] + 0.1, f"Weight: {weight:.2f}", fontsize=8, color='green')
        numeric_id = waypoint_to_numeric.get(wp_id, "N/A")
        ax.text(coord[0], coord[1], coord[2] - 0.1, f"Anchor: {numeric_id}", fontsize=8, color='purple')
    ax.scatter(xs, ys, zs, color='red', s=50, label='Transformed Anchor')
    
    # Plot edges.
    for key, value in edge_data.items():
        from_id = value["from"]
        to_id = value["to"]
        norm_weight = value["normalized_weight"]
        pt1 = transformed_anchor_map[from_id]
        pt2 = transformed_anchor_map[to_id]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color='gray', alpha=0.5)
        midpoint = ((pt1[0]+pt2[0])/2, (pt1[1]+pt2[1])/2, (pt1[2]+pt2[2])/2)
        ax.text(midpoint[0], midpoint[1], midpoint[2], f"{norm_weight:.2f}", fontsize=7, color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Edge Weights between Anchors (Normalized)')
    set_axes_equal(ax)
    ax.legend()
    plt.show()

def create_cylinder_between_points(p1, p2, radius, resolution=20, color=[0, 0, 1]):
    """
    Create a cylinder mesh between two 3D points p1 and p2 with the specified radius.
    The cylinder is initially created along the z-axis and then rotated to align with p2-p1.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    height = np.linalg.norm(p2 - p1)
    if height == 0:
        return None

    # Create the cylinder along the z-axis.
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.paint_uniform_color(color)
    
    # Compute the direction vector and the rotation matrix.
    direction = (p2 - p1) / height
    z_axis = np.array([0, 0, 1])
    # If the direction is (almost) identical to z, no rotation is needed.
    if np.allclose(direction, z_axis):
        R = np.eye(3)
    elif np.allclose(direction, -z_axis):
        # 180 degree rotation around any axis perpendicular to z_axis.
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    
    cylinder.rotate(R, center=(0, 0, 0))
    # Translate the cylinder so that its center is at the midpoint between p1 and p2.
    midpoint = p1 + (p2 - p1) / 2
    cylinder.translate(midpoint)
    return cylinder

def visualize_edge_values_with_pointcloud(points, edge_data, transformed_anchor_map, waypoint_to_numeric, metrics, edge_radius=0.03):
    """
    Visualize a point cloud, anchors, and edges between anchors using Open3D.
    
    - The point cloud is rendered in uniform gray.
    - Each anchor is shown as a sphere, colored based on its metric value.
    - Each edge is drawn as a cylinder connecting two anchors, colored based on its normalized weight.
    
    Parameters:
      points (Nx3 array): The point cloud coordinates.
      edge_data (dict): Dictionary of edges with keys "from", "to", and "normalized_weight" (in [0,1]).
      transformed_anchor_map (dict): Mapping from waypoint ID to 3D coordinates.
      waypoint_to_numeric (dict): (Unused here, retained for consistency.)
      metrics (dict): Mapping from waypoint ID to its metric value.
      edge_radius (float): Radius of the cylinder representing each edge.
    """
    geometries = []
    
    # Create and add the point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(pcd)
    
    # Prepare a colormap using matplotlib's viridis.
    cmap = cm.get_cmap("viridis")
    
    # Normalize metric values for anchors.
    if metrics:
        metric_values = list(metrics.values())
        vmin, vmax = min(metric_values), max(metric_values)
        # Avoid division by zero if all metric values are equal.
        if vmin == vmax:
            norm = lambda x: 0.5
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = lambda x: 0.5  # default to mid of colormap if no metrics provided

    # Add anchors as spheres colored based on their metric weight.
    for wp_id, coord in transformed_anchor_map.items():
        metric_value = metrics.get(wp_id, 0.5)
        normalized_metric = norm(metric_value)
        anchor_color = list(cmap(normalized_metric)[:3])
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(coord)
        sphere.paint_uniform_color(anchor_color)
        geometries.append(sphere)
    
    # Add edges as cylinders colored based on normalized weight.
    # It is assumed that each edge's "normalized_weight" is in the range [0, 1].
    for edge in edge_data.values():
        from_id = edge["from"]
        to_id = edge["to"]
        norm_weight = edge["normalized_weight"]
        
        pt1 = transformed_anchor_map[from_id]
        pt2 = transformed_anchor_map[to_id]
        edge_color = list(cmap(norm_weight)[:3])
        
        cylinder = create_cylinder_between_points(pt1, pt2, radius=edge_radius, color=edge_color)
        if cylinder is not None:
            geometries.append(cylinder)
    
    # Open an interactive Open3D window with the generated geometries.
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud, Anchors, and Colored Edges")

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


#############################
# Main Block
#############################
if __name__ == "__main__":
    
    prefix = "greenhouse_very_final"
    # Paths
    # for old versions
    output_file = f"{BASE_PTH}/edge_weights/{prefix}_edge_values.json"
    if prefix == "greenhouse_final":
        sdk_graph_path = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph/"
        metrics_file = "/media/martin/Elements/ros-recordings/metrics_assigned_to_time/greenhouse_final_metrics.json"
        ply_path = '/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/merged_cloud_selected.pcd'

        # Transformation parameters.
        rotation_z = 140
        rotation_y = -5
        translation = [2.4, -2.0, -0.4]

    elif prefix == "greenhouse_feb":
        sdk_graph_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/downloaded_graph/"
        metrics_file = "/media/martin/Elements/ros-recordings/metrics_assigned_to_time/greenhouse_feb_metrics.json"
        ply_path = '/media/martin/Elements/ros-recordings/pointclouds/merged_cloud_selected.pcd'
        rotation_z = 171
        rotation_y = -5
        translation = [2.4, -1.5, -0.5]
    
    elif prefix == "imtek":
        sdk_graph_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/campus_imtek/downloaded_graph"
        metrics_file = "/media/martin/Elements/ros-recordings/metrics_assigned_to_time/imtek_metrics.json"
        ply_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/merged_cloud_selected.pcd"
        rotation_z = 162
        rotation_y = 1
        translation = [2.7, -0.9, -0.4]
    
    elif prefix == "greenhouse_march":
        sdk_graph_path = "/media/martin/Elements/ros-recordings/recordings/march_11/downloaded_graph"
        metrics_file = "/media/martin/Elements/ros-recordings/metrics_assigned_to_time/greenhouse_march_metrics.json"
        ply_path = '/media/martin/Elements/ros-recordings/recordings/march_11/merged_cloud_selected.pcd'
        rotation_z = 205
        rotation_y = -1.5
        translation = [2.25, -0.8, -0.45]
    
    elif prefix == "greenhouse_very_final":
        sdk_graph_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph"
        metrics_file = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/metrics_to_time/greenhouse_time_metrics.json"
        ply_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_selected_large.pcd"
        transformation_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"

        tranformation = load_transformation_params(transformation_path)
        rotation_z = tranformation["rotation_z"]
        rotation_y = tranformation["rotation_y"]
        translation = tranformation["translation"]

        output_file = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/edge_weights/greenhouse_edge_values.json"
    else:
        raise ValueError("Invalid prefix. Please provide a valid prefix for the data.")

    os.environ['XDG_SESSION_TYPE'] = 'x11'

    # Load the graph.
    graph = load_map(sdk_graph_path)
    
    # Compute edge values.
    edge_data, transformed_anchor_map, waypoint_to_numeric = compute_edge_values(
        graph, metrics_file, rotation_z, rotation_y, translation, metric_key="average_effort"
    )

    anchor_ids = extract_anchor_ids(graph)
    metrics = load_metrics(metrics_file, anchor_ids, metric_key="average_effort")

    # Optionally build a cost graph.
    cost_graph = build_cost_graph(edge_data)
    print(f"Built cost graph with {len(cost_graph.nodes)} nodes and {len(cost_graph.edges)} edges.")

    # Optionally export the edge values.
    export_edge_values_to_file(edge_data, output_file)
    
    visualize_3d = False
    if visualize_3d:
        os.environ['XDG _SESSION_TYPE'] = 'x11'
        points = o3d.io.read_point_cloud(ply_path)
        points_origin = np.asarray(points.points)
        visualize_edge_values_with_pointcloud(points_origin, edge_data, transformed_anchor_map, waypoint_to_numeric, metrics)
        exit()

    print("Plot 3d")
    plot_edge_values(edge_data, transformed_anchor_map, waypoint_to_numeric, metrics)
    print("PLot 2d")
    plot_edge_values_2d(edge_data, transformed_anchor_map, waypoint_to_numeric, metrics)
    print("Done!")
import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import json

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from full_processing.read_all_assign_weights_and_plot import load_transformation_params

##############################################
# Visualization functions
##############################################
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

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.paint_uniform_color(color)
    
    # Compute rotation: align the cylinder's z-axis with (p2-p1)
    direction = (p2 - p1) / height
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        R_mat = np.eye(3)
    elif np.allclose(direction, -z_axis):
        R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    
    cylinder.rotate(R_mat, center=(0, 0, 0))
    midpoint = p1 + (p2 - p1) / 2
    cylinder.translate(midpoint)
    return cylinder

def visualize_points_and_edges(colored_pcd, anchors, edge_data, waypoint_rgb_map, edge_radius=0.03):
    """
    Visualize a colored point cloud along with:
      - Large waypoint spheres colored according to the assigned scores (from waypoint_rgb_map)
      - Edge cylinders connecting waypoints (colored based on normalized edge weight)
    
    Parameters:
      colored_pcd (o3d.geometry.PointCloud): The colored point cloud (from your scored cloud).
      anchors (dict): Mapping from waypoint ID to 3D coordinates (e.g. anchors_transformed from fit_wp_to_pc).
      edge_data (dict): Dictionary where each key (e.g. "from -> to") maps to a dict with:
            "from": from waypoint id,
            "to": to waypoint id,
            "raw_weight": raw edge weight,
            "normalized_weight": normalized edge weight in [0,1],
            "mean_time_spent": 0.
      waypoint_rgb_map (dict): Mapping from waypoint ID to RGB color (list of three floats).
      edge_radius (float): Cylinder radius.
    """
    cmap = cm.get_cmap("viridis")
    geometries = [colored_pcd]
    
    # Add waypoint spheres (large spheres) colored by assigned scores.
    for wp_id, coord in anchors.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(coord)
        color = waypoint_rgb_map.get(wp_id, [0.5, 0.5, 0.5])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    # Add edge cylinders.
    for key, ed in edge_data.items():
        from_id = ed["from"]
        to_id = ed["to"]
        norm_weight = ed["normalized_weight"]
        if from_id in anchors and to_id in anchors:
            pt1 = np.array(anchors[from_id])
            pt2 = np.array(anchors[to_id])
            edge_color = list(cmap(norm_weight)[:3])
            cylinder = create_cylinder_between_points(pt1, pt2, radius=edge_radius, resolution=20, color=edge_color)
            if cylinder is not None:
                geometries.append(cylinder)
    
    o3d.visualization.draw_geometries(geometries, window_name="Colored Points, Waypoints, and Edges")

##############################################
# Graph update & saving functions
##############################################
def save_graph(graph, path):
    """
    Save the updated graph to a new file.
    """
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, 'graph')
    with open(out_path, 'wb') as f:
        f.write(graph.SerializeToString())
    print(f"New graph saved to {out_path}")

def update_graph_edges(graph, edge_data):
    """
    Update the graph's edge cost values using the computed edge_data.
    For each edge in the graph, if its key (formatted as "from -> to") is found in edge_data,
    update edge.annotations.cost.value with the new raw_weight.
    """
    for edge in graph.edges:
        key = f"{edge.id.from_waypoint} -> {edge.id.to_waypoint}"
        if key in edge_data:
            new_cost = edge_data[key]["raw_weight"]
            edge.annotations.cost.value = new_cost
        else:
            print(f"Warning: Edge {key} not found in computed edge data; leaving original weight.")
    return graph

def update_and_save_graph(graph, edge_data, updated_graph_folder):
    """
    Update the graph with new edge weights and save the updated graph.
    """
    updated_graph = update_graph_edges(graph, edge_data)
    save_graph(updated_graph, updated_graph_folder)

##############################################
# Remaining functions (fit_wp_to_pc, scoring, etc.)
##############################################
def fit_wp_to_pc(pc_pth, graph_pth, odometry_pth, pc_T):
    """
    Load the graph, extract anchors, apply transformations, assign odometry, refine mappings,
    and return refined waypoint-odometry mapping, the point cloud, transformed anchors, and the graph.
    """
    from fit_sdk_odometry.martin_fit_graph_odo import (
        load_map, extract_anchorings_from_graph, apply_transformations, transform_odometry_to_lidar_frame,
        assign_odometry_to_waypoints, refine_waypoint_odometry_mapping
    )
    rotation_z = pc_T["rotation_z"]
    rotation_y = pc_T["rotation_y"]
    translation = pc_T["translation"]
    
    pcd = o3d.io.read_point_cloud(pc_pth)
    sdk_graph = load_map(graph_pth)
    anchors_dict = extract_anchorings_from_graph(sdk_graph)
    print(f"Extracted {len(anchors_dict)} anchor points from the graph")
    
    # Transform anchors.
    anchors_transformed = apply_transformations(anchors_dict, rotation_z, rotation_y, translation)
    
    lidar2imu = np.eye(4)
    odometry_points_lidar, timestamps = transform_odometry_to_lidar_frame(odometry_pth, lidar2imu)
    waypoint_odometry_mapping = assign_odometry_to_waypoints(anchors_transformed, odometry_points_lidar, timestamps)
    refined_waypoint_odometry_mapping = refine_waypoint_odometry_mapping(
        waypoint_odometry_mapping, anchors_transformed, min_points_threshold=5,
    )
    return refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph

def assign_avg_score_to_waypoints(waypoints, scored_cloud, neighbor_radius=0.3):
    """
    For each waypoint, average the semantic scores of points in scored_cloud (using x,y distance)
    and return a mapping from waypoint ID to the averaged numeric score.
    """
    points = scored_cloud[:, :3]
    scores = scored_cloud[:, 3]
    pc_xy = points[:, :2]
    
    waypoint_scores = {}
    for wp_id, coord in waypoints.items():
        anchor_xy = np.array(coord[:2])
        distances = np.linalg.norm(pc_xy - anchor_xy, axis=1)
        neighbor_mask = distances <= neighbor_radius
        if np.sum(neighbor_mask) == 0:
            closest_idx = np.argmin(distances)
            avg_score = scores[closest_idx]
        else:
            avg_score = np.mean(scores[neighbor_mask])
        waypoint_scores[wp_id] = avg_score
    return waypoint_scores

def compute_edge_values_from_waypoint_scores(graph, waypoint_scores, 
                                             rotation_z=0, rotation_y=0, translation=(0,0,0)):
    """
    Compute raw and normalized edge weights between anchors using waypoint_scores.
    
    Steps:
      1. Extract anchor IDs and positions from the graph.
      2. Transform anchor positions.
      3. For each edge, compute:
           raw_weight = ((score1 + score2) / 2) + Euclidean_distance(pos1, pos2)
      4. Normalize raw weights to [0,1].
      
    Returns:
      export_data (dict): Mapping "from -> to" to a dict with raw_weight, normalized_weight, and mean_time_spent (0).
      transformed_anchor_map (dict): Transformed anchor positions.
      waypoint_to_numeric (dict): Mapping from waypoint ID to itself.
    """
    from edge_weights.assign_edge_weights import (
        extract_anchor_ids, extract_anchor_map, transform_anchor_map, extract_edges_from_graph
    )
    anchor_ids = extract_anchor_ids(graph)
    anchor_map = extract_anchor_map(graph)
    transformed_anchor_map = transform_anchor_map(anchor_map, rotation_z, rotation_y, translation)
    
    waypoint_to_numeric = {wp: wp for wp in anchor_ids}
    
    edges = extract_edges_from_graph(graph)
    raw_edges = []
    for from_id, to_id, _ in edges:
        if from_id in transformed_anchor_map and to_id in transformed_anchor_map:
            pos1 = np.array(transformed_anchor_map[from_id])
            pos2 = np.array(transformed_anchor_map[to_id])
            score1 = waypoint_scores.get(from_id, 0)
            score2 = waypoint_scores.get(to_id, 0)
            avg_score = (score1 + score2) / 2.0
            dist = np.linalg.norm(pos1 - pos2)
            raw_weight = avg_score + dist
            raw_edges.append((from_id, to_id, raw_weight))
    if raw_edges:
        weights = [w for (_, _, w) in raw_edges]
        min_w = min(weights)
        max_w = max(weights)
        normalized_edges = [
            (f, t, (w - min_w) / (max_w - min_w + 1e-8)) for (f, t, w) in raw_edges
        ]
    else:
        normalized_edges = []
    
    export_data = {}
    for (from_id, to_id, raw_weight), (_, _, norm_weight) in zip(raw_edges, normalized_edges):
        key = f"{from_id} -> {to_id}"
        export_data[key] = {
            "from": from_id,
            "to": to_id,
            "raw_weight": raw_weight,
            "normalized_weight": norm_weight,
            "mean_time_spent": 0
        }
    return export_data, transformed_anchor_map, waypoint_to_numeric

####################################
# Main function
####################################
def main():
    visualize_3d = True
    update_graph = True
    data = "greenhouse_very_final"  # Change as needed

    if data == "greenhouse_very_final":
        default_odo = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/odometry_greenhouse.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph"
        trans_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"
        # Load merged scored cloud (NumPy file).
        scored_cloud_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/merged_scored_cloud.npy"
        scored_cloud = np.load(scored_cloud_path)
        # For fit_wp_to_pc, use a dummy PCD with geometry only.
        dummy_pcd_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/dummy_geometry.pcd"
        transformation = load_transformation_params(trans_path)
    
    # Get graph and transformed anchors.
    refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph = fit_wp_to_pc(
        pc_pth=dummy_pcd_path,
        graph_pth=default_sdk,
        odometry_pth=default_odo,
        pc_T=transformation
    )
    
    # Build a colored point cloud from the scored cloud.
    scored_points = scored_cloud[:, :3]
    scored_scores = scored_cloud[:, 3]
    norm_scores = (scored_scores - scored_scores.min()) / (scored_scores.max() - scored_scores.min() + 1e-8)
    cmap = cm.get_cmap("viridis")
    scored_colors = cmap(norm_scores)[:, :3]
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(scored_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(scored_colors)
    
    # Compute averaged numeric scores for each waypoint.
    waypoint_scores = assign_avg_score_to_waypoints(anchors_transformed, scored_cloud, neighbor_radius=0.3)
    
    # Build an RGB mapping for waypoints from these scores.
    waypoint_rgb_map = {}
    min_score = min(waypoint_scores.values())
    max_score = max(waypoint_scores.values())
    for wp, score in waypoint_scores.items():
        norm_score = (score - min_score) / (max_score - min_score + 1e-8)
        waypoint_rgb_map[wp] = list(cmap(norm_score)[:3])
    
    # Compute edge values from waypoint scores.
    edge_data, transformed_anchor_map2, waypoint_to_numeric = compute_edge_values_from_waypoint_scores(
        sdk_graph, waypoint_scores,
        rotation_z=transformation["rotation_z"],
        rotation_y=transformation["rotation_y"],
        translation=transformation["translation"]
    )
    
    if visualize_3d:
        visualize_points_and_edges(colored_pcd, anchors_transformed, edge_data, waypoint_rgb_map, edge_radius=0.03)
    
    if update_graph:
        updated_graph_folder = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/updated_graph/"
        # Update the graph edges with new weights and save the updated graph.
        from edge_weights.load_edge_weights import save_graph  # assuming save_graph is defined there
        updated_graph = update_graph_edges(sdk_graph, edge_data)
        save_graph(updated_graph, updated_graph_folder)

if __name__ == "__main__":
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    main()

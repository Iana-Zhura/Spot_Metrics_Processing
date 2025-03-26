import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from bosdyn.api.graph_nav import map_pb2
import open3d as o3d
from scipy.spatial import KDTree
import random
import json  # for exporting timestamps

# Functions to load and process the graph
def load_map(path):
    """
    Load a map from the given file path.
    """
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)
        print(f'Loaded graph with {len(current_graph.waypoints)} waypoints and {len(current_graph.anchoring.anchors)} anchors')
        return current_graph

def extract_anchorings_from_graph(graph):
    """
    Extract all anchored waypoints and their 3D coordinates from the graph.
    Returns a dictionary mapping each anchor ID to its (x,y,z) coordinate.
    """
    anchors_dict = {}
    for anchor in graph.anchoring.anchors:
        pos = anchor.seed_tform_waypoint.position
        anchors_dict[anchor.id] = np.array([pos.x, pos.y, pos.z])
    return anchors_dict

def apply_transformations(anchors_dict, rotation_z, rotation_y, translation):
    """
    Apply rotation and translation transformations to each anchor coordinate.
    Returns a dictionary with the same keys (anchor IDs) and transformed 3D coordinates.
    """
    theta_z = np.radians(rotation_z)
    theta_y = np.radians(rotation_y)

    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                                  [np.sin(theta_z),  np.cos(theta_z), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                                  [0, 0, 0, 1]])

    translation_matrix = np.array([[1, 0, 0, translation[0]],
                                   [0, 1, 0, translation[1]],
                                   [0, 0, 1, translation[2]],
                                   [0, 0, 0, 1]])

    # Combined transformation matrix
    T = rotation_matrix_y @ translation_matrix @ rotation_matrix_z

    transformed_anchors = {}
    for anchor_id, coord in anchors_dict.items():
        coord_homogeneous = np.append(coord, 1)  # convert to homogeneous coordinate
        transformed = T @ coord_homogeneous
        transformed_anchors[anchor_id] = transformed[:3]
    return transformed_anchors

def assign_odometry_to_waypoints(anchors_dict, odometry_points, timestamps):
    """
    Assign odometry points and timestamps to the nearest anchor (waypoint) using a KDTree.
    Returns a dictionary mapping each anchor ID to its odometry data.
    """
    # Build list of anchor IDs and corresponding coordinates
    anchor_ids = list(anchors_dict.keys())
    anchor_coords = np.array([anchors_dict[aid] for aid in anchor_ids])
    tree = KDTree(anchor_coords)
    
    waypoint_assignments = {aid: {'points': [], 'timestamps': []} for aid in anchor_ids}
    
    for odometry_point, timestamp in zip(odometry_points, timestamps):
        _, nearest_index = tree.query(odometry_point)
        nearest_anchor_id = anchor_ids[nearest_index]
        waypoint_assignments[nearest_anchor_id]['points'].append(odometry_point)
        waypoint_assignments[nearest_anchor_id]['timestamps'].append(timestamp)
    
    return waypoint_assignments

def visualize_point_cloud_with_anchors_open3d(points, anchors_dict, waypoint_odometry_mapping):
    """
    Visualize the point cloud, anchors, and odometry path with unique colors per anchor bin using Open3D.
    Anchors with IDs "13" and "0" are specifically colored blue.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    anchor_spheres = []
    for anchor_id, anchor in anchors_dict.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(anchor)
        if anchor_id in ["13", "0"]:
            sphere.paint_uniform_color([0, 0, 1])  # Blue for specific anchors
        else:
            sphere.paint_uniform_color([1, 0, 0])  # Red for other anchors
        anchor_spheres.append(sphere)

    odometry_geometries = []
    for anchor_id, data in waypoint_odometry_mapping.items():
        if data['points']:
            odometry_pcd = o3d.geometry.PointCloud()
            odometry_pcd.points = o3d.utility.Vector3dVector(data['points'])
            color = [random.random(), random.random(), random.random()]
            odometry_pcd.paint_uniform_color(color)
            
            for point in data['points']:
                odometry_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                odometry_sphere.translate(point)
                odometry_sphere.paint_uniform_color(color)
                odometry_geometries.append(odometry_sphere)

    geometries = [pcd] + anchor_spheres + odometry_geometries
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud & Anchors with Colored Odometry Bins")

def transform_odometry_to_lidar_frame(odometry_csv, lidar2imu):
    """
    Transform odometry points from IMU to LiDAR frame and extract timestamps.
    """
    odometry_df = pd.read_csv(odometry_csv, header=None)
    timestamps = odometry_df.iloc[:, 0].values
    odometry_points = []

    for _, row in odometry_df.iterrows():
        x, y, z = row[1:4]
        point = np.array([x, y, z, 1])
        transformed_point = (np.linalg.inv(lidar2imu) @ point)[:3]
        odometry_points.append(transformed_point)

    return np.array(odometry_points), timestamps

def discretize_timestamps(waypoint_odometry_mapping):
    """
    Discretize the timestamps by grouping odometry timestamps into intervals based on sorted odometry timestamps.
    The first and last brackets are open-ended.
    """
    all_timestamps = sorted([ts for data in waypoint_odometry_mapping.values() for ts in data['timestamps']])
    time_brackets = [all_timestamps[0]] + [(all_timestamps[i] + all_timestamps[i+1]) / 2 for i in range(len(all_timestamps)-1)] + [all_timestamps[-1]]
    
    discretized_waypoint_mapping = {}
    for anchor_id, data in waypoint_odometry_mapping.items():
        discretized_waypoint_mapping[anchor_id] = []
        for timestamp in data['timestamps']:
            for i in range(len(time_brackets) - 1):
                if time_brackets[i] <= timestamp < time_brackets[i+1]:
                    discretized_waypoint_mapping[anchor_id].append((time_brackets[i], time_brackets[i+1]))
                    break
                elif timestamp >= time_brackets[-1]:
                    discretized_waypoint_mapping[anchor_id].append((time_brackets[-2], time_brackets[-1]))
                    break
                elif timestamp < time_brackets[0]:
                    discretized_waypoint_mapping[anchor_id].append((time_brackets[0], time_brackets[1]))
                    break
    
    return discretized_waypoint_mapping

def merge_intervals(intervals, tol=1e-6):
    """
    Merge overlapping or immediately adjacent intervals in a list.
    """
    if not intervals:
        return []
    
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        current_start, current_end = current
        
        if current_start - last_end <= tol:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append(current)
    
    return merged

def discretize_and_merge_timestamps(waypoint_odometry_mapping):
    """
    Discretize timestamps for each anchor, then merge overlapping intervals.
    """
    discretized_mapping = discretize_timestamps(waypoint_odometry_mapping)
    
    merged_mapping = {}
    for anchor_id, intervals in discretized_mapping.items():
        merged_mapping[anchor_id] = merge_intervals(intervals)
    return merged_mapping

def export_merged_timestamps(merged_mapping, output_filename):
    """
    Export the merged timestamp intervals to a JSON file.
    Each exported entry includes the anchor ID as the key, a list of (start, end) intervals, and the overall time spent.
    """
    export_data = {}
    for anchor_id, intervals in merged_mapping.items():
        export_data[anchor_id] = {
            "waypoint_id": anchor_id,
            "intervals": [{"start": start, "end": end} for start, end in intervals],
            "overall_time": sum(end - start for start, end in intervals)
        }
    
    with open(output_filename, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nExported merged time intervals to '{output_filename}'")

def print_nonmerged_timestamps(wp_odo_mapping):
    """
    Pretty print the non-merged time intervals (min/max) for each anchor.
    """
    for anchor_id, data in wp_odo_mapping.items():
        if data['timestamps']:
            start_time = min(data['timestamps'])
            end_time = max(data['timestamps'])
            duration = end_time - start_time
            print(f"Anchor {anchor_id}: Start Time = {start_time:.2f}, End Time = {end_time:.2f}, Duration = {duration:.2f}")

def print_merged_timestamps(merged_mapping):
    """
    Pretty print the merged time intervals for each anchor.
    """
    print("\nMerged Time Intervals per Anchor:")
    for anchor_id, intervals in merged_mapping.items():
        interval_strs = [f"[{start:.2f}, {end:.2f})" for start, end in intervals]
        overall_time = sum(end - start for start, end in intervals)
        print(f"Anchor {anchor_id}: {', '.join(interval_strs)} | Overall Time Spent: {overall_time:.2f}")

def refine_waypoint_odometry_mapping(waypoint_odometry_mapping, anchors_dict, min_points_threshold=5, merge_distance=0.15):
    """
    Refine the waypoint odometry mapping by merging sparse data and sharing odometry between close anchors.
    """
    RESET = "\033[0m"
    YELLOW = "\033[93m"

    # Start with a copy of the original mapping
    refined_mapping = {aid: {'points': list(data['points']), 'timestamps': list(data['timestamps'])}
                       for aid, data in waypoint_odometry_mapping.items()}
    
    # Build a KDTree from the 2D positions (x,y) of the anchors
    anchor_ids = list(anchors_dict.keys())
    anchor_coords = np.array([anchors_dict[aid] for aid in anchor_ids])
    anchor_tree = KDTree(anchor_coords[:, :2])
    
    # Merge anchors with low point counts
    for aid, data in waypoint_odometry_mapping.items():
        if len(data['points']) < min_points_threshold:
            # Query k=2 to get the nearest neighbor (first is self)
            _, nearest_indices = anchor_tree.query(anchors_dict[aid][:2], k=2)
            nearest_aid = anchor_ids[nearest_indices[1]]
            refined_mapping[aid]['points'].extend(waypoint_odometry_mapping[nearest_aid]['points'])
            refined_mapping[aid]['timestamps'].extend(waypoint_odometry_mapping[nearest_aid]['timestamps'])
            print(f"{YELLOW}Merging:{RESET} Anchor {aid} had few points, inherited from nearest {nearest_aid}")

    # Share odometry between close anchors
    for aid, data in waypoint_odometry_mapping.items():
        nearby_indices = anchor_tree.query_ball_point(anchors_dict[aid][:2], merge_distance)
        for idx in nearby_indices:
            neighbor_aid = anchor_ids[idx]
            if neighbor_aid != aid:
                refined_mapping[aid]['points'].extend(waypoint_odometry_mapping[neighbor_aid]['points'])
                refined_mapping[aid]['timestamps'].extend(waypoint_odometry_mapping[neighbor_aid]['timestamps'])
                print(f"{YELLOW}Sharing:{RESET} Anchors {aid} ↔ {neighbor_aid} are close; sharing odometry.")

    # Remove duplicates and sort timestamps
    for aid, data in refined_mapping.items():
        unique_points = list({tuple(p) for p in data['points']})
        unique_timestamps = sorted(set(data['timestamps']))
        refined_mapping[aid]['points'] = [list(p) for p in unique_points]
        refined_mapping[aid]['timestamps'] = unique_timestamps

    print("Refinement Complete!")
    return refined_mapping

def fit_graph2odo(graph_pth, odometry_pth, pc_pth, export_pth, rotation_z=140, rotation_y=-5, translation=[2.4, -2.0, -0.4], logging=False):
    """
    Load the graph, extract anchor points, apply transformations, assign odometry, refine mappings,
    discretize and merge timestamps, then export the merged time intervals (with anchor IDs) to a JSON file.
    """
    points = o3d.io.read_point_cloud(pc_pth)
    points_origin = np.asarray(points.points)

    sdk_graph = load_map(graph_pth)
    anchors_dict = extract_anchorings_from_graph(sdk_graph)
    print(f"Extracted {len(anchors_dict)} anchor points from the graph")

    # Apply transformations to anchors
    anchors_transformed = apply_transformations(anchors_dict, rotation_z, rotation_y, translation)

    # Load and transform odometry points
    lidar2imu = np.eye(4)
    odometry_points_lidar, timestamps = transform_odometry_to_lidar_frame(odometry_pth, lidar2imu)

    # Assign odometry to anchors
    waypoint_odometry_mapping = assign_odometry_to_waypoints(anchors_transformed, odometry_points_lidar, timestamps)

    refined_waypoint_odometry_mapping = refine_waypoint_odometry_mapping(
        waypoint_odometry_mapping, anchors_transformed, min_points_threshold=5,
    )

    discretized_waypoint_mapping = discretize_and_merge_timestamps(refined_waypoint_odometry_mapping)

    if logging:
        print_merged_timestamps(discretized_waypoint_mapping)
    export_merged_timestamps(discretized_waypoint_mapping, export_pth)

    input("Press Enter to visualize the point cloud with anchors and colored odometry bins")
    visualize_point_cloud_with_anchors_open3d(points_origin, anchors_transformed, refined_waypoint_odometry_mapping)

def save_anchor_waypoint_mapping(anchors_dict, output_filename):
    """
    Save a JSON file mapping each anchor ID to its corresponding coordinate.
    (Coordinates are converted to lists for JSON serialization.)
    """
    mapping = {aid: anchors_dict[aid].tolist() for aid in anchors_dict}
    with open(output_filename, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved anchor-waypoint mapping to '{output_filename}'")

# Main block:
if __name__ == "__main__":

    os.environ['XDG_SESSION_TYPE'] = 'x11'
    prefix = "greenhouse_march"

    if prefix=="greenhouse_final":
        sdk_graph = load_map(f'/media/martin/Elements/ros-recordings/recordings/{prefix}/downloaded_graph')
        odometry_csv = "/media/martin/Elements/ros-recordings/odometry/greenhouse_odo.csv"
        ply_path = '/media/martin/Elements/ros-recordings/pointclouds/merged_cloud_selected.pcd'

        id_map_path = f"fit_sdk_odometry/{prefix}_id_map.json"
        save_file = f"fit_sdk_odometry/{prefix}_fit_output.json"

        rotation_z = 140
        rotation_y = -5
        translation = [2.4, -2.0, -0.4]
    elif prefix=="greenhouse_feb":
        sdk_graph = load_map("/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/graph_data/downloaded_graph")
        odometry_csv = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/odo_greenhouse_feb.csv"
        ply_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/merged_cloud_selected.pcd"
        id_map_path = "fit_sdk_odometry/greenhouse_feb_id_map.json"
        save_file = "fit_sdk_odometry/greenhouse_feb_fit_output.json"
        
        rotation_z = 171
        rotation_y = -5
        translation = [2.4, -1.5, -0.5]
    elif prefix=="imtek":
        sdk_graph = load_map("/media/martin/Elements/ros-recordings/recordings/feb_27/campus_imtek/downloaded_graph")
        odometry_csv = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/imtek_feb_odo.csv"
        ply_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/merged_cloud_selected.pcd"
        id_map_path = "fit_sdk_odometry/imtek_id_map.json"
        save_file = "fit_sdk_odometry/imtek_fit_output.json"
        
        rotation_z = 162
        rotation_y = 1
        translation = [2.7, -0.9, -0.4]
    
    elif prefix == "greenhouse_march":
        sdk_graph = load_map("/media/martin/Elements/ros-recordings/recordings/march_11/downloaded_graph")
        odometry_csv = "/media/martin/Elements/ros-recordings/recordings/march_11/greenhouse_march_odo.csv"
        ply_path = "/media/martin/Elements/ros-recordings/recordings/march_11/merged_cloud_selected.pcd"
        id_map_path = "fit_sdk_odometry/greenhouse_march_id_map.json"
        save_file = "fit_sdk_odometry/greenhouse_march_fit_output.json"

        rotation_z = 205
        rotation_y = -1.5
        translation = [2.25, -0.8, -0.45]


    points = o3d.io.read_point_cloud(ply_path)
    points_origin = np.asarray(points.points)

    anchors_dict = extract_anchorings_from_graph(sdk_graph)
    print(f"Extracted {len(anchors_dict)} anchor points from the graph")
    anchors_transformed = apply_transformations(anchors_dict, rotation_z, rotation_y, translation)
    lidar2imu = np.eye(4)
    odometry_points_lidar, timestamps = transform_odometry_to_lidar_frame(odometry_csv, lidar2imu)
    waypoint_odometry_mapping = assign_odometry_to_waypoints(anchors_transformed, odometry_points_lidar, timestamps)

    print_nonmerged_timestamps(waypoint_odometry_mapping)
    discretized_waypoint_mapping = discretize_and_merge_timestamps(waypoint_odometry_mapping)
    print_merged_timestamps(discretized_waypoint_mapping)
    
    # Export merged intervals including anchor IDs
    export_merged_timestamps(discretized_waypoint_mapping, save_file)
    
    # input("Press Enter to visualize the point cloud with anchors and colored odometry bins")
    visualize_point_cloud_with_anchors_open3d(points_origin, anchors_transformed, waypoint_odometry_mapping)

    # Save the anchor to coordinate mapping
    save_anchor_waypoint_mapping(anchors_dict, id_map_path)

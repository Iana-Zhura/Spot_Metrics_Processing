import os
import json
import numpy as np
import matplotlib.pyplot as plt
from bosdyn.api.graph_nav import map_pb2
import open3d as o3d

def load_map(path):
    """
    Load a map from the given file path.
    """
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)
    print(f"Loaded map with {len(current_graph.waypoints)} waypoints and {len(current_graph.anchoring.anchors)} anchors")
    return current_graph

def extract_anchorings_from_graph(graph):
    """
    Extract all anchored waypoints and their 3D coordinates from the graph.
    Returns a dictionary mapping anchor IDs to their (x,y,z) coordinates.
    """
    anchors_dict = {}
    for anchor in graph.anchoring.anchors:
        pos = anchor.seed_tform_waypoint.position
        anchors_dict[anchor.id] = np.array([pos.x, pos.y, pos.z])
    return anchors_dict

def apply_transformations(anchors, rotation_z, rotation_y, translation):
    """
    Apply rotation and translation transformations to the anchor coordinates.
    anchors: dictionary mapping anchor id -> coordinate array.
    Returns a dictionary with the same keys and transformed coordinates.
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

    transformed = {}
    for aid, coord in anchors.items():
        coord_homogeneous = np.append(coord, 1)
        transformed_coord = T @ coord_homogeneous
        transformed[aid] = transformed_coord[:3]
    return transformed

def load_odometry_csv(filename):
    """
    Load odometry data from a CSV file.
    
    Assumes column 0 contains timestamps and columns 1-3 contain x,y,z coordinates.
    """
    df = np.genfromtxt(filename, delimiter=',')
    return df

def order_waypoints_ids(graph):
    """
    Order waypoints based on the time when they were created.
    Returns a list of waypoint IDs ordered by creation time.
    """
    ordered_waypoints = sorted(graph.waypoints, key=lambda wp: wp.annotations.creation_time.seconds)
    return [ow.id for ow in ordered_waypoints]

def plot_anchors_weights_2d(map_path, odometry_file, weights_file, rotation_z=140, rotation_y=-5, translation=[2.4, -2.0, -0.4]):
    # Load the map and extract anchor points as a dictionary
    sdk_graph = load_map(map_path)
    anchors_dict = extract_anchorings_from_graph(sdk_graph)
    
    # Apply transformation (same as used during export)
    anchors_transformed = apply_transformations(anchors_dict, rotation_z, rotation_y, translation)
    
    # Use creation order for the anchor IDs
    ordered_ids = order_waypoints_ids(sdk_graph)
    anchor_ids = [aid for aid in ordered_ids if aid in anchors_transformed]
    anchor_coords_arr = np.array([anchors_transformed[aid] for aid in anchor_ids])
    
    # Load odometry data and extract positions (assuming columns 1-3 are x,y,z)
    odometry_data = load_odometry_csv(odometry_file)
    odom_points = odometry_data[:, 1:4]
    
    # Load the exported results (now keyed by anchor IDs)
    with open(weights_file, 'r') as f:
        waypoint_results = json.load(f)
    
    # Extract metrics from JSON for each anchor in the ordered list
    overall_time_values = np.array([waypoint_results.get(aid, {}).get("time_spent", 0.0) for aid in anchor_ids])
    energy_values = np.array([waypoint_results.get(aid, {}).get("energy", 0.0) for aid in anchor_ids])
    average_effort_values = np.array([waypoint_results.get(aid, {}).get("average_effort", 0.0) for aid in anchor_ids])
    distance_from_wp0 = np.linalg.norm(anchor_coords_arr[:, :2] - anchor_coords_arr[0, :2], axis=1)
    
    # Create a mapping from waypoint string ID to an integer based on creation time
    creation_order = {aid: idx for idx, aid in enumerate(ordered_ids)}

    # print the ordered IDs
    print("\nOrdered Waypoint IDs based on creation time:")
    for idx, aid in enumerate(ordered_ids):
        print(f"ID {idx}: {aid}")
    
    # Create a figure with four subplots (4 rows)
    fig, axs = plt.subplots(4, 1, figsize=(8, 24))
    offset = 0.05  # Vertical offset for annotation
    
    # Subplot 1: Overall Time per Anchor with annotations (original order)
    sc1 = axs[0].scatter(anchor_coords_arr[:, 0],
                         anchor_coords_arr[:, 1],
                         c=overall_time_values,
                         cmap='RdYlGn_r',  # lower time: green, higher: red
                         s=100,
                         edgecolor='k')
    axs[0].set_title('Overall Time per Anchor')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(True)
    axs[0].plot(odom_points[:, 0], odom_points[:, 1], color='blue', linewidth=1.5, alpha=0.7, label='Odometry Path')
    axs[0].set_aspect('equal', 'box')
    fig.colorbar(sc1, ax=axs[0], label='Overall Time (seconds)')
    for i, (x, y) in enumerate(anchor_coords_arr[:, :2]):
        axs[0].text(x, y + offset, f'{overall_time_values[i]:.2f}', color='black', fontsize=9,
                    ha='center', va='bottom')
    
    # Subplot 2: Aggregated Energy per Anchor (showing only the energy values)
    sc2 = axs[1].scatter(anchor_coords_arr[:, 0],
                         anchor_coords_arr[:, 1],
                         c=energy_values,
                         cmap='RdYlGn_r',
                         s=100,
                         edgecolor='k')
    axs[1].set_title('Aggregated Energy per Anchor')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(True)
    axs[1].plot(odom_points[:, 0], odom_points[:, 1], color='blue', linewidth=1.5, alpha=0.7, label='Odometry Path')
    axs[1].set_aspect('equal', 'box')
    fig.colorbar(sc2, ax=axs[1], label='Energy')
    for i, (x, y) in enumerate(anchor_coords_arr[:, :2]):
        axs[1].text(x, y + offset, f'{energy_values[i]:.2f}', color='black', fontsize=9,
                    ha='center', va='bottom')
    
    # Subplot 3: Distance from the first anchor with creation order integer annotations
    sc3 = axs[2].scatter(anchor_coords_arr[:, 0],
                         anchor_coords_arr[:, 1],
                         c=distance_from_wp0,
                         cmap='RdYlGn_r',  # lower distance: green, higher: red
                         s=100,
                         edgecolor='k')
    # Use the creation order for the first anchor as reference (should be 0)
    title_anchor_order = creation_order.get(ordered_ids[0], 0)
    axs[2].set_title('Distance from Anchor ' + str(title_anchor_order))
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].grid(True)
    axs[2].plot(odom_points[:, 0], odom_points[:, 1], color='blue', linewidth=1.5, alpha=0.7, label='Odometry Path')
    axs[2].set_aspect('equal', 'box')
    fig.colorbar(sc3, ax=axs[2], label='Distance')
    for i, (x, y) in enumerate(anchor_coords_arr[:, :2]):
        # Instead of the string id, show the creation order integer
        creation_idx = creation_order.get(anchor_ids[i], -1)
        axs[2].text(x, y + offset, f'ID {creation_idx}', color='black', fontsize=9,
                    ha='center', va='bottom')
    
    # Subplot 4: Average Effort per Anchor with annotations (original order)
    sc4 = axs[3].scatter(anchor_coords_arr[:, 0],
                         anchor_coords_arr[:, 1],
                         c=average_effort_values,
                         cmap='RdYlGn_r',
                         s=100,
                         edgecolor='k')
    axs[3].set_title('Average Effort per Anchor')
    axs[3].set_xlabel('X')
    axs[3].set_ylabel('Y')
    axs[3].grid(True)
    axs[3].plot(odom_points[:, 0], odom_points[:, 1], color='blue', linewidth=1.5, alpha=0.7, label='Odometry Path')
    axs[3].set_aspect('equal', 'box')
    fig.colorbar(sc4, ax=axs[3], label='Average Effort')
    for i, (x, y) in enumerate(anchor_coords_arr[:, :2]):
        axs[3].text(x, y + offset, f'{average_effort_values[i]:.2f}', color='black', fontsize=9,
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File paths (update these paths as needed)
    prefix = "greenhouse_final"
    map_path = '/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph'
    odometry_csv = '/media/martin/Elements/ros-recordings/odometry/greenhouse_odo.csv'
    weights_file = "/media/martin/Elements/ros-recordings/metrics_assigned_to_time/greenhouse_final_metrics.json"
    
    plot_anchors_weights_2d(map_path, odometry_csv, weights_file)

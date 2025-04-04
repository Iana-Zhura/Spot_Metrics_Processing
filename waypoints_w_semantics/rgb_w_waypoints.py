import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from full_processing.read_all_assign_weights_and_plot import load_transformation_params

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
        R_mat = np.eye(3)
    elif np.allclose(direction, -z_axis):
        # 180 degree rotation around any axis perpendicular to z_axis.
        R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    
    cylinder.rotate(R_mat, center=(0, 0, 0))
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

def visualize_point_cloud_with_anchors_open3d(pcd, anchors_dict, waypoint_odometry_mapping):
    """
    Visualize the colored point cloud, anchors, and odometry path with unique colors per anchor bin using Open3D.
    Anchors with IDs "13" and "0" are specifically colored blue.
    
    Parameters:
      pcd (open3d.geometry.PointCloud): The pre-loaded colored point cloud.
      anchors_dict (dict): Mapping from anchor IDs to 3D coordinates.
      waypoint_odometry_mapping (dict): Mapping from anchor IDs to their associated odometry points.
    """
    # Do not override the point cloud's colors; if not present, paint uniformly.
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    anchor_spheres = []
    for anchor_id, anchor in anchors_dict.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(anchor)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for other anchors
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

def fit_wp_to_pc(pc_pth, graph_pth, odometry_pth, pc_T):
    """
    Load the graph, extract anchor points, apply transformations, assign odometry, refine mappings,
    discretize and merge timestamps, then export the merged time intervals (with anchor IDs) to a JSON file.
    """
    from fit_sdk_odometry.martin_fit_graph_odo import (
        load_map, extract_anchorings_from_graph, apply_transformations, transform_odometry_to_lidar_frame,
        assign_odometry_to_waypoints, refine_waypoint_odometry_mapping
    )

    rotation_z = pc_T["rotation_z"]
    rotation_y = pc_T["rotation_y"]
    translation = pc_T["translation"]
    
    # Load the colored point cloud.
    pcd = o3d.io.read_point_cloud(pc_pth)

    sdk_graph = load_map(graph_pth)
    anchors_dict = extract_anchorings_from_graph(sdk_graph)
    print(f"Extracted {len(anchors_dict)} anchor points from the graph")

    # Apply transformations to anchors.
    anchors_transformed = apply_transformations(anchors_dict, rotation_z, rotation_y, translation)

    # Load and transform odometry points.
    lidar2imu = np.eye(4)
    odometry_points_lidar, timestamps = transform_odometry_to_lidar_frame(odometry_pth, lidar2imu)

    # Assign odometry to anchors.
    waypoint_odometry_mapping = assign_odometry_to_waypoints(anchors_transformed, odometry_points_lidar, timestamps)

    refined_waypoint_odometry_mapping = refine_waypoint_odometry_mapping(
        waypoint_odometry_mapping, anchors_transformed, min_points_threshold=5,
    )

    return refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph

#####
def assign_avg_rgb_to_waypoints(waypoints, point_cloud, neighbor_radius=0.3):
    """
    For each waypoint, finds all points in the point cloud (using only x,y distance)
    within a given radius, and assigns the averaged RGB color to the waypoint.
    If no neighbors are found, the RGB value of the closest point is used.

    Parameters:
        waypoints (dict): Mapping from waypoint ID to 3D coordinates.
        point_cloud (open3d.geometry.PointCloud): Point cloud with RGB values.
        neighbor_radius (float): Radius (in the xy-plane) within which to search for neighboring points.

    Returns:
        dict: Mapping from waypoint ID to averaged RGB color [r, g, b].
    """
    if not point_cloud.has_colors():
        raise ValueError("The point cloud does not have RGB color information.")

    pc_points = np.asarray(point_cloud.points)
    pc_colors = np.asarray(point_cloud.colors)
    # Extract only the x,y coordinates for distance calculation.
    pc_xy = pc_points[:, :2]

    waypoint_rgb = {}
    for wp_id, coord in waypoints.items():
        anchor_xy = np.array(coord[:2])
        # Compute the xy distances from the anchor to all point cloud points.
        distances = np.linalg.norm(pc_xy - anchor_xy, axis=1)
        # Identify the indices of points within the neighbor_radius.
        neighbor_mask = distances <= neighbor_radius
        
        if np.sum(neighbor_mask) == 0:
            # If no neighbors found, fallback to the closest point.
            closest_idx = np.argmin(distances)
            avg_color = pc_colors[closest_idx]
        else:
            avg_color = np.mean(pc_colors[neighbor_mask], axis=0)
        
        waypoint_rgb[wp_id] = avg_color.tolist()

    return waypoint_rgb

def visualize_point_cloud_with_rgb_waypoints_and_odometry(pcd, anchors_dict, waypoint_rgb_map, waypoint_odometry_mapping=None):
    """
    Visualizes the colored point cloud, RGB-colored anchor spheres, and odometry minipoints.
    
    Anchors (waypoints) are colored based on the computed RGB values.
    Odometry minipoints are added as small black spheres.
    
    Parameters:
        pcd (open3d.geometry.PointCloud): The pre-loaded colored point cloud.
        anchors_dict (dict): Mapping from anchor IDs to 3D coordinates.
        waypoint_rgb_map (dict): Mapping from anchor IDs to average RGB color [r, g, b].
        waypoint_odometry_mapping (dict, optional): Mapping from anchor IDs to their associated odometry points.
    """
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    geometries = [pcd]

    # Add anchors as spheres colored with the averaged RGB value.
    for anchor_id, anchor in anchors_dict.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(anchor)
        rgb = waypoint_rgb_map.get(anchor_id, [0.5, 0.5, 0.5])
        sphere.paint_uniform_color(rgb)
        geometries.append(sphere)

    # Add odometry minipoints as small black spheres.
    if waypoint_odometry_mapping:
        for anchor_id, data in waypoint_odometry_mapping.items():
            if data['points']:
                for point in data['points']:
                    odometry_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                    odometry_sphere.translate(point)
                    odometry_sphere.paint_uniform_color([0.0, 0.0, 0.0])  # Black color for odometry points
                    geometries.append(odometry_sphere)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with RGB Anchors and Black Odometry")





### Main function to visualize the point cloud with anchors and edges.

if __name__ == "__main__":

    visualize_3d = True
    data = "greenhouse_very_final"  # Change this to "greenhouse" or "greenhouse_very_final" as needed

    if data == "greenhouse_very_final":
        default_odo = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/odometry_greenhouse.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph"
        default_pc = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_selected.pcd"
        default_metrics = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/metrics_csvs/joint_states.csv"
        default_prefix = "greenhouse_final_very_final"

        fit_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/greenhouse_final_fit_output.json"
        metric_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/metrics_to_time/greenhouse_time_metrics.json"
        trans_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"

        rgb_pc_pth = '/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_colored.pcd'

        # this is assuming I ran the martin_fit_graph_odo.py before and saved the transformation params 
        # (that is kind of required anyway, so I say it's a fair assumption)
        transformation = load_transformation_params(trans_path)
    

    refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph = fit_wp_to_pc(pc_pth=rgb_pc_pth, graph_pth=default_sdk, odometry_pth=default_odo, pc_T=transformation)
    
    waypoint_rgb_map = assign_avg_rgb_to_waypoints(anchors_transformed, pcd)


    if visualize_3d:
        os.environ['XDG_SESSION_TYPE'] = 'x11'
        # Load the colored point cloud (RGB-projected) from file.
        # Use the colored point cloud for visualization.
            # Visualize using the pre-colored point cloud.
        # visualize_point_cloud_with_anchors_open3d(pcd, anchors_transformed, refined_waypoint_odometry_mapping)
        
        visualize_point_cloud_with_rgb_waypoints_and_odometry(pcd, anchors_transformed, waypoint_rgb_map, refined_waypoint_odometry_mapping)


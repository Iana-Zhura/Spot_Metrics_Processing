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
        if vmin == vmax:
            norm = lambda x: 0.5
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = lambda x: 0.5

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
    
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud, Anchors, and Colored Edges")

def visualize_point_cloud_with_anchors_open3d(pcd, anchors_dict, waypoint_odometry_mapping):
    """
    Visualize the colored point cloud, anchors, and odometry path with unique colors per anchor bin using Open3D.
    """
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    anchor_spheres = []
    for anchor_id, anchor in anchors_dict.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(anchor)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])
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
    and return the refined waypoint odometry mapping along with the loaded point cloud and transformed anchors.
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

def assign_avg_score_to_waypoints(waypoints, scored_cloud, neighbor_radius=0.3):
    """
    For each waypoint, find all points in the scored_cloud (using only x,y distance)
    within a given radius, average their semantic scores, and then map that average score to an RGB color.
    If no neighbors are found, the score of the nearest point is used.
    
    Parameters:
        waypoints (dict): Mapping from waypoint ID to 3D coordinates.
        scored_cloud (np.array): (N,4) array where columns 0-2 are coordinates and column 3 is the score.
        neighbor_radius (float): Radius in the xy-plane to search for neighbors.
    
    Returns:
        dict: Mapping from waypoint ID to averaged RGB color [r, g, b].
    """
    import matplotlib.cm as cm
    points = scored_cloud[:, :3]
    scores = scored_cloud[:, 3]
    pc_xy = points[:, :2]
    
    vmin = np.min(scores)
    vmax = np.max(scores)
    cmap = cm.get_cmap("viridis")
    
    waypoint_rgb = {}
    for wp_id, coord in waypoints.items():
        anchor_xy = np.array(coord[:2])
        distances = np.linalg.norm(pc_xy - anchor_xy, axis=1)
        neighbor_mask = distances <= neighbor_radius
        if np.sum(neighbor_mask) == 0:
            closest_idx = np.argmin(distances)
            avg_score = scores[closest_idx]
        else:
            avg_score = np.mean(scores[neighbor_mask])
        # Normalize the score.
        norm_score = (avg_score - vmin) / (vmax - vmin + 1e-8)
        rgb = cmap(norm_score)[:3]
        waypoint_rgb[wp_id] = list(rgb)
    return waypoint_rgb

def visualize_point_cloud_with_rgb_waypoints_and_odometry(pcd, anchors_dict, waypoint_rgb_map, waypoint_odometry_mapping=None):
    """
    Visualize the colored point cloud, RGB-colored anchor spheres, and odometry minipoints.
    Anchors are colored based on the computed RGB values from their averaged semantic score.
    Odometry minipoints are added as small black spheres.
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
                    odometry_sphere.paint_uniform_color([0.0, 0.0, 0.0])
                    geometries.append(odometry_sphere)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with RGB Anchors and Black Odometry")

def main():
    visualize_3d = True
    data = "greenhouse_very_final"  # Change as needed

    if data == "greenhouse_very_final":
        default_odo = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/odometry_greenhouse.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph"
        trans_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"
        # Instead of an RGB PCD, load the merged scored cloud (NumPy file)
        scored_cloud_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/merged_scored_cloud.npy"
        scored_cloud = np.load(scored_cloud_path)
        
        # For the fit_wp_to_pc function we still need a PCD file.
        # You can either generate a dummy PCD or use one saved earlier.
        # Here we assume a PCD with geometry only (no RGB) is available.
        dummy_pcd_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/dummy_geometry.pcd"
        transformation = load_transformation_params(trans_path)
    
    # Load the graph and anchors using the dummy PCD.
    refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph = fit_wp_to_pc(
        pc_pth=dummy_pcd_path,
        graph_pth=default_sdk,
        odometry_pth=default_odo,
        pc_T=transformation
    )
    
    # Create a colored point cloud for visualization from the scored cloud.
    scored_points = scored_cloud[:, :3]
    scored_scores = scored_cloud[:, 3]
    norm_scores = (scored_scores - scored_scores.min()) / (scored_scores.max() - scored_scores.min() + 1e-8)
    cmap = cm.get_cmap("viridis")
    scored_colors = cmap(norm_scores)[:, :3]
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(scored_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(scored_colors)
    
    # Compute averaged semantic scores (converted to RGB) for each waypoint.
    waypoint_rgb_map = assign_avg_score_to_waypoints(anchors_transformed, scored_cloud, neighbor_radius=0.3)

    if visualize_3d:
        visualize_point_cloud_with_rgb_waypoints_and_odometry(colored_pcd, anchors_transformed, waypoint_rgb_map, refined_waypoint_odometry_mapping)

if __name__ == "__main__":
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    main()


from utils.path_simulation.waypoints_w_semantics.scores_w_waypoints import assign_avg_score_to_waypoints, fit_wp_to_pc, load_transformation_params
from utils.path_simulation.waypoints_w_semantics.scores_w_waypoints import  compute_edge_values_from_waypoint_scores, visualize_points_and_edges, update_graph_edges
import open3d as o3d
import matplotlib.cm as cm  # For color mapping


def update_graph_with_scores(merged_cloud, sdk_graph_path, updated_graph_folder, odometry_csv, dummy_pcd_path, fit_path, visualize=False, update_graph=True):
    """
    Update the graph with scores from the merged point cloud.
    Args:
        merged_cloud (numpy.ndarray): Merged point cloud with scores.
        sdk_graph (str): Path to the SDK graph.
        odometry_csv (str): Path to the odometry CSV file.
        dummy_pcd_path (str): Path to the dummy PCD file.
        fit_path (str): Path to the fit parameters.
    """
    scored_points = merged_cloud[:, :3]
    scored_scores = merged_cloud[:, 3]

    norm_scores = (scored_scores - scored_scores.min()) / (scored_scores.max() - scored_scores.min() + 1e-8)
    cmap = cm.get_cmap("viridis")
    scored_colors = cmap(norm_scores)[:, :3]
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(scored_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(scored_colors)

    # print("Visualizing final merged point cloud with RGB colors based on semantic scores...")
    # o3d.visualization.draw_geometries([pc])

    # Compute averaged numeric scores for each waypoint.
    # Get graph and transformed anchors.
    transformation = load_transformation_params(fit_path)
    refined_waypoint_odometry_mapping, pcd, anchors_transformed, sdk_graph = fit_wp_to_pc(
        pc_pth=dummy_pcd_path,
        graph_pth=sdk_graph_path ,
        odometry_pth=odometry_csv,
        pc_T=transformation
    )
    waypoint_scores = assign_avg_score_to_waypoints(anchors_transformed, merged_cloud, neighbor_radius=0.3)

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
    
    if visualize:
        visualize_points_and_edges(colored_pcd, anchors_transformed, edge_data, waypoint_rgb_map, edge_radius=0.03)

    if update_graph:
        # Update the graph edges with new weights and save the updated graph.
        from edge_weights.load_edge_weights import save_graph  # assuming save_graph is defined there
        updated_graph = update_graph_edges(sdk_graph, edge_data)
        save_graph(updated_graph, updated_graph_folder)
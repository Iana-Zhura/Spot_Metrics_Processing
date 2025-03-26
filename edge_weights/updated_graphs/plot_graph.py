import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from bosdyn.api.graph_nav import map_pb2
import open3d as o3d

#############################
# Helper Functions
#############################
def load_map(path):
    """
    Load a graph from the given directory path.
    Assumes the graph file is named 'graph' within the directory.
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

def extract_edges_from_graph(graph):
    """
    Extract all edges from the graph.
    Returns a list of tuples: (from_anchor_id, to_anchor_id, cost)
    where cost is read directly from the edge's annotations.
    """
    edge_list = []
    for edge in graph.edges:
        from_id = edge.id.from_waypoint
        to_id = edge.id.to_waypoint
        cost = edge.annotations.cost.value  # original cost from the graph
        edge_list.append((from_id, to_id, cost))
    return edge_list

def apply_transformations(coords, rotation_z, rotation_y, translation):
    """
    Apply rotation (around Z and Y) and translation transformations to the coordinates.
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
    return {key: tuple(transformed_coords[i]) for i, key in enumerate(keys)}

def plot_graph_2d(anchor_map, edges):
    """
    Plot a 2D representation of the graph using matplotlib.
    
    - Each anchor is plotted as a red point and annotated with only the first 8 characters of its ID.
    - Each edge is drawn as a line between anchors, colored by its cost.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot anchors.
    xs, ys = [], []
    for wp_id, coord in anchor_map.items():
        xs.append(coord[0])
        ys.append(coord[1])
        truncated_id = wp_id[:8]
        ax.text(coord[0], coord[1], truncated_id, fontsize=8, color='black')
    ax.scatter(xs, ys, color='red', s=50, label='Anchor')
    
    # Normalize edge costs for colormap mapping.
    costs = [cost for _, _, cost in edges]
    if costs:
        min_cost = min(costs)
        max_cost = max(costs)
    else:
        min_cost, max_cost = 0, 1
    norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)
    cmap = cm.get_cmap("viridis")
    
    # Plot edges with color based on their cost.
    for from_id, to_id, cost in edges:
        if from_id in anchor_map and to_id in anchor_map:
            pt1 = anchor_map[from_id]
            pt2 = anchor_map[to_id]
            normalized_cost = norm(cost)
            edge_color = cmap(normalized_cost)
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=edge_color, alpha=0.8)
            # Annotate the midpoint with the cost value.
            midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
            ax.text(midpoint[0], midpoint[1], f"{cost:.2f}", fontsize=7, color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Graph with Edge Colors by Cost (Matplotlib)')
    ax.legend()
    ax.grid(True)
    plt.show()

def create_cylinder_between_points(p1, p2, radius, resolution=20, color=[0, 0, 1]):
    """
    Create a cylinder mesh between two 3D points p1 and p2 with the specified radius.
    The cylinder is created along the z-axis and then rotated to align with the vector (p2-p1).
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    height = np.linalg.norm(p2 - p1)
    if height == 0:
        return None

    # Create the cylinder along the z-axis.
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.paint_uniform_color(color)
    
    # Align the cylinder with (p2 - p1).
    direction = (p2 - p1) / height
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        R = np.eye(3)
    elif np.allclose(direction, -z_axis):
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    
    cylinder.rotate(R, center=(0, 0, 0))
    # Translate the cylinder so its center is at the midpoint between p1 and p2.
    midpoint = p1 + (p2 - p1) / 2
    cylinder.translate(midpoint)
    return cylinder

def visualize_graph_with_pointcloud(point_cloud, edges, anchor_map, edge_radius=0.03, anchor_radius=0.1):
    """
    Visualize the original point cloud with the graph overlaid using Open3D.
    
    - point_cloud: an open3d.geometry.PointCloud object.
    - edges: list of tuples (from_id, to_id, cost).
    - anchor_map: dictionary mapping anchor IDs to their 3D coordinates.
    - edge_radius: radius of the cylinders representing edges.
    - anchor_radius: radius of the spheres representing anchors.
    """
    geometries = []
    
    # Add the original point cloud (displayed in light gray).
    point_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(point_cloud)
    
    # Prepare a colormap to color edges based on their cost.
    costs = [cost for _, _, cost in edges]
    if costs:
        min_cost, max_cost = min(costs), max(costs)
    else:
        min_cost, max_cost = 0, 1
    norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)
    cmap = cm.get_cmap("viridis")
    
    # Add anchors as spheres (colored red).
    for anchor_id, coord in anchor_map.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=anchor_radius)
        sphere.translate(coord)
        sphere.paint_uniform_color([1.0, 0, 0])
        geometries.append(sphere)
    
    # Add edges as cylinders colored based on their cost.
    for from_id, to_id, cost in edges:
        if from_id in anchor_map and to_id in anchor_map:
            pt1 = anchor_map[from_id]
            pt2 = anchor_map[to_id]
            normalized_cost = norm(cost)
            edge_color = list(cmap(normalized_cost))[:3]  # Discard alpha.
            cylinder = create_cylinder_between_points(pt1, pt2, radius=edge_radius, color=edge_color)
            if cylinder is not None:
                geometries.append(cylinder)
    
    # Launch the Open3D visualization window.
    o3d.visualization.draw_geometries(geometries, window_name="Graph Over Point Cloud")

#############################
# Main Block
#############################
if __name__ == "__main__":
    # Choose dataset and set paths/parameters.
    prefix = "greenhouse_march"
    # sdk_graph_path = '/media/martin/spot_extern/martin/new_graphs/greenhouse_final/edge_by_hand'
    sdk_graph_path = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs"
    
    if prefix == "greenhouse_final":
        ply_path = '/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/merged_cloud_selected.pcd'
        rotation_z = 140
        rotation_y = -5
        translation = [2.4, -2.0, -0.4]
    elif prefix == "greenhouse_feb":
        ply_path = '/media/martin/Elements/ros-recordings/pointclouds/merged_cloud_selected.pcd'
        rotation_z = 171
        rotation_y = -5
        translation = [2.4, -1.5, -0.5]
    elif prefix == "imtek":
        ply_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/merged_cloud_selected.pcd"
        rotation_z = 162
        rotation_y = 1
        translation = [2.7, -0.9, -0.4]
    elif prefix == "greenhouse_march":
        ply_path = "/media/martin/Elements/ros-recordings/recordings/march_11/merged_cloud_selected.pcd"
        rotation_z = 205
        rotation_y = -1.5
        translation = [2.25, -0.8, -0.45]
    
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    
    # Load the graph.
    graph = load_map(sdk_graph_path)
    
    # Extract and transform anchor positions.
    anchor_map = extract_anchor_map(graph)
    transformed_anchor_map = transform_anchor_map(anchor_map, rotation_z, rotation_y, translation)
    
    # Extract edges (with original cost values).
    edges = extract_edges_from_graph(graph)
    
    # --- First: Matplotlib 2D Plot ---
    plot_graph_2d(transformed_anchor_map, edges)
    
    # --- Second: Open3D Point Cloud with Graph Overlay ---
    point_cloud = o3d.io.read_point_cloud(ply_path)
    visualize_graph_with_pointcloud(point_cloud, edges, transformed_anchor_map)

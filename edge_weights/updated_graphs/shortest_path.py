import os
import json
import wandb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from bosdyn.api.graph_nav import map_pb2


##################################
# Transformation Parameter Loader
##################################
def load_transformation_params(json_path):
    """
    Load transformation parameters from a JSON file.
    
    Expects keys: "rotation_z", "rotation_y", "translation".
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    for key in ["rotation_z", "rotation_y", "translation"]:
        if key not in params:
            raise KeyError(f"Missing key '{key}' in the loaded JSON.")
    return params


##################################
# Graph Loading & Edge Extraction
##################################
def load_map(graph_folder):
    """
    Load a graph from the specified folder containing a file named 'graph'.
    """
    graph_file_path = os.path.join(graph_folder, 'graph')
    with open(graph_file_path, 'rb') as f:
        data = f.read()
    graph = map_pb2.Graph()
    graph.ParseFromString(data)
    print(f"Loaded graph with {len(graph.waypoints)} waypoints and {len(graph.anchoring.anchors)} anchors")
    return graph


def extract_anchor_map(graph):
    """
    Extract a mapping of anchor IDs to their 3D coordinates.
    
    Returns:
        dict: {anchor_id: (x, y, z)}
    """
    anchor_map = {}
    for anchor in graph.anchoring.anchors:
        pos = anchor.seed_tform_waypoint.position
        anchor_map[anchor.id] = (pos.x, pos.y, pos.z)
    return anchor_map


def extract_edges_from_graph(graph):
    """
    Extract all edges from the graph.
    
    Returns:
        dict: { (from_anchor_id, to_anchor_id): cost }
    """
    edges = {}
    for edge in graph.edges:
        from_id = edge.id.from_waypoint
        to_id = edge.id.to_waypoint
        cost = edge.annotations.cost.value
        edges[(from_id, to_id)] = cost
    return edges


##################################
# Shortest Path Functions
##################################
def order_waypoints_ids(graph):
    """
    Order waypoints based on their creation time.
    
    Returns:
        list: ordered waypoint IDs.
    """
    ordered = sorted(graph.waypoints, key=lambda wp: wp.annotations.creation_time.seconds)
    return [wp.id for wp in ordered]


def print_order_waypoints_ids(graph):
    """
    Print each waypoint's index and ID based on creation order.
    """
    for i, wp_id in enumerate(order_waypoints_ids(graph)):
        print(f"Key: {i}, Value: {wp_id}")


def build_cost_graph(edge_dict):
    """
    Build a bidirectional NetworkX directed graph from edge data.
    
    Args:
        edge_dict (dict): {(from, to): cost}
        
    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()
    for (u, v), cost in edge_dict.items():
        G.add_edge(u, v, weight=cost)
        G.add_edge(v, u, weight=cost)
    return G


def get_nth_shortest_paths(G, start, goal, n):
    """
    Return the primary (shortest) path and the n-th shortest path between start and goal.
    
    Args:
        G (networkx.DiGraph): the cost graph.
        start: starting anchor ID.
        goal: goal anchor ID.
        n (int): which shortest path to return as the secondary path 
                 (n=1 returns the primary, n=2 returns the second-shortest, etc.).
                 
    Returns:
        tuple: (primary_path, nth_path)
    """

    try:
        paths_gen = nx.shortest_simple_paths(G, source=start, target=goal, weight='weight')
        primary = next(paths_gen)
    except nx.NetworkXNoPath:
        print(f"No path found between {start} and {goal}.")
        return None, None
    
    if n is None or n <= 1:
        return primary, None

    for _ in range(n - 1):
        nth = next(paths_gen, None)
        if nth is None:
            break
    return primary, nth

def find_paths(graph, start, goal, path=[]):
    path = path + [start]  # include the current node in the path
    if start == goal:
        return [path]  # if current node is goal, return path
    if start not in graph:
        return []  # if the node isn't in the graph, return empty list
    paths = []  # list to store all paths
    for node in graph[start]:
        if node not in path:  # avoid cycles
            newpaths = find_paths(graph, node, goal, path)  # recursive call
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def main_shortest_path(graph_folder, start_index=None, goal_index=None, nth=2):
    """
    Load the graph and compute the primary and n-th shortest paths.
    
    Args:
        graph_folder (str): folder containing the graph file.
        start_index (int): index (in creation order) for the start waypoint.
        goal_index (int): index (in creation order) for the goal waypoint.
        nth (int): which shortest path to return as secondary.
        
    Returns:
        tuple: (graph, primary_path, nth_path)
    """
    graph = load_map(graph_folder)
    edge_data = extract_edges_from_graph(graph)
    G = build_cost_graph(edge_data)
    ordered_ids = order_waypoints_ids(graph)
    print_order_waypoints_ids(graph)
    
    if start_index is None or goal_index is None:
        print("No start/goal index provided; using defaults: start = 0, goal = 6.")
        start, goal = ordered_ids[0], ordered_ids[6]
    else:
        start, goal = ordered_ids[start_index], ordered_ids[goal_index]
    
    primary, nth_path = get_nth_shortest_paths(G, start, goal, nth)
    all_paths = find_paths(G, start, goal)
    print(f"\nPrimary shortest path: {primary}")
    print(f"\n{nth}-th shortest path: {nth_path}")
    return graph, primary, nth_path, all_paths


##################################
# 2D Plotting Functions
##################################
def transform_anchor_map(anchor_map, rotation_z, rotation_y, translation):
    """
    Apply rotation and translation to anchor coordinates.
    
    Args:
        anchor_map (dict): {anchor_id: (x, y, z)}
        rotation_z (float): rotation about the Z-axis (degrees).
        rotation_y (float): rotation about the Y-axis (degrees).
        translation (list): [tx, ty, tz].
        
    Returns:
        dict: {anchor_id: (x', y')}
    """
    transformed = {}
    theta_z = np.radians(rotation_z)
    theta_y = np.radians(rotation_y)
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z),  np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]])
    T = np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]])
    M = Ry @ T @ Rz  # Combined transformation
    
    for aid, coord in anchor_map.items():
        vec = np.array([coord[0], coord[1], coord[2], 1])
        transformed_vec = M @ vec
        transformed[aid] = (transformed_vec[0], transformed_vec[1])
    return transformed


def plot_graph_2d(transformed_anchor_map, graph, primary_path, nth_path=None, nth=None, log=False):
    """
    Plot the graph in 2D with:
      - All anchors shown as red dots with abbreviated IDs.
      - All edges drawn in light gray with edge cost annotations.
      - Primary shortest path highlighted in blue.
      - The n-th shortest path highlighted in green.
      - Total cost (score) for both paths displayed in the upper-left corner.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot anchors.
    for aid, (x, y) in transformed_anchor_map.items():
        ax.scatter(x, y, color='red', s=50)
        ax.text(x, y, aid[:8], fontsize=8, color='black')
    
    # Build a dictionary for edge costs.
    edge_costs = {}
    for edge in graph.edges:
        u = edge.id.from_waypoint
        v = edge.id.to_waypoint
        cost = edge.annotations.cost.value
        edge_costs[(u, v)] = cost

    # Plot all edges and annotate with cost.
    for edge in graph.edges:
        u = edge.id.from_waypoint
        v = edge.id.to_waypoint
        if u in transformed_anchor_map and v in transformed_anchor_map:
            x1, y1 = transformed_anchor_map[u]
            x2, y2 = transformed_anchor_map[v]
            ax.plot([x1, x2], [y1, y2], color='lightgray', alpha=0.5)
            mid = ((x1+x2)/2, (y1+y2)/2)
            ax.text(mid[0], mid[1], f"{edge.annotations.cost.value:.2f}", fontsize=7, color='purple')
    
    def compute_path_cost(path):
        total = 0.0
        for u, v in zip(path, path[1:]):
            cost = edge_costs.get((u, v)) or edge_costs.get((v, u)) or 0.0
            total += cost
        return total
    
    primary_cost = compute_path_cost(primary_path) if primary_path else 0.0
    nth_cost = compute_path_cost(nth_path) if nth_path else 0.0
    
    # Plot primary shortest path (blue).
    if primary_path:
        for i in range(len(primary_path) - 1):
            u = primary_path[i]
            v = primary_path[i+1]
            if u in transformed_anchor_map and v in transformed_anchor_map:
                x1, y1 = transformed_anchor_map[u]
                x2, y2 = transformed_anchor_map[v]
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=3,
                        label='Primary Shortest' if i == 0 else "")
    
    # Plot n-th shortest path (green).
    if nth_path:
        for i in range(len(nth_path) - 1):
            u = nth_path[i]
            v = nth_path[i+1]
            if u in transformed_anchor_map and v in transformed_anchor_map:
                x1, y1 = transformed_anchor_map[u]
                x2, y2 = transformed_anchor_map[v]
                ax.plot([x1, x2], [y1, y2], color='green', linewidth=3,
                        label='n-th Shortest' if i == 0 else "")
    
    # Display path cost scores.
    ax.text(0.02, 0.95, f"lowest cost: {primary_cost:.2f}", transform=ax.transAxes,
            fontsize=10, color='blue', verticalalignment='top')
    if nth_path:
        ax.text(0.02, 0.90, f"{nth}-th cost: {nth_cost:.2f}", transform=ax.transAxes,
                fontsize=10, color='green', verticalalignment='top')
    
    ax.set_title("2D Graph Plot with Edge Weights & Shortest Paths")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    # plt.show()
    return fig
   


##################################
# Main Execution Block
##################################
if __name__ == '__main__':
    # Configuration for different datasets.
    location = "greenhouse_very_final"
    plot_enabled = False

    if location == "greenhouse_final":
        graph_folder = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        start_index = 0
        goal_index = 6
    elif location == "greenhouse_march":
        graph_folder = "/media/martin/Elements/ros-recordings/edge_weights/updated_graphs/"
        start_index = 0
        goal_index = 5
    elif location == "greenhouse_very_final":
        graph_folder = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/updated_graph"
        trans_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"
        start_index = 0
        goal_index = 7
        plot_enabled = True

    # Set which shortest path to highlight (e.g., nth = 3 means the third-shortest path). Set to None for primary only.
    nth = 2

    # Compute the primary and nth shortest paths.
    graph, primary_path, nth_path = main_shortest_path(graph_folder, start_index, goal_index, nth)

    if plot_enabled:
        # Load transformation parameters.
        params = load_transformation_params(trans_path)
        rotation_z = params["rotation_z"]
        rotation_y = params["rotation_y"]
        translation = params["translation"]
        
        # Transform anchor coordinates for 2D plotting.
        anchor_map = extract_anchor_map(graph)
        transformed_map = transform_anchor_map(anchor_map, rotation_z, rotation_y, translation)
        
        # Plot the graph with annotated edge costs and highlighted paths.
        plot_graph_2d(transformed_map, graph, primary_path, nth_path, nth)

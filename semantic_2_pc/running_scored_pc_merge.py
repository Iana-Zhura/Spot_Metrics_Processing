import os
import sys
import numpy as np
from PIL import Image
import json
import open3d as o3d  # Only used for visualization
import matplotlib.cm as cm  # For color mapping

# Add parent directory so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_2_pc.project_semantic_2_lidar import project_semantic_to_lidar, load_cam_config
from image_2_pc.project_image_2_pc import create_adjustment_matrix

def transform_points(points, transformation):
    """
    Transform an (N,3) array of points using a 4x4 transformation matrix.
    """
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=points.dtype)
    points_hom = np.hstack((points, ones))  # (N,4)
    transformed = (transformation @ points_hom.T).T[:, :3]
    return transformed

def iteration(semantic_array, lidar_pc, transformation, merged_cloud,
              camera_matrix, adjustment_matrix, cam2lidar):
    """
    1. Call project_semantic_to_lidar to obtain semantic scores for each lidar point.
    2. Only keep those lidar points that received a valid score.
    3. If not the first iteration, transform these points using the provided transformation.
    4. Merge the new valid points along with their scores (as the 4th column) into the merged cloud.
    
    Returns:
        Merged cloud as a NumPy array of shape (N,4), where columns 0:3 are coordinates and column 3 is the score.
    """
    # Get the projection result (semantic scores)
    pc_trans = project_semantic_to_lidar(
        semantic_array, lidar_pc, camera_matrix, adjustment_matrix, cam2lidar,
        visualize=False, save_pc_pth=None
    )
    
    # Print average valid semantic score (ignoring points that remain unassigned, i.e. NaN)
    # print(f"Average valid fitness: {np.nanmean(pc_trans)}")
    
    # Create a mask for valid points (i.e. those that got assigned a score)
    valid_mask = ~np.isnan(pc_trans)
    if np.count_nonzero(valid_mask) == 0:
        print("No valid projected points found in this iteration; skipping.")
        return merged_cloud
    
    # Select only the valid lidar points and corresponding scores
    new_cloud_points = lidar_pc[valid_mask]
    new_scores = pc_trans[valid_mask]
    
    # If merged_cloud already contains points, transform the new points into the common frame.
    if merged_cloud.shape[0] > 0:
        new_cloud_points = transform_points(new_cloud_points, transformation)
    
    # Combine the new points and their scores into one array (shape: [N,4])
    new_cloud_with_scores = np.hstack((new_cloud_points, new_scores.reshape(-1, 1)))
    
    # Merge with previously accumulated points
    if merged_cloud.shape[0] == 0:
        merged = new_cloud_with_scores
    else:
        merged = np.vstack((merged_cloud, new_cloud_with_scores))
    
    return merged

def png_to_numpy(png_path):
    """
    Load a PNG image, resize to 224x224, convert to grayscale,
    and normalize pixel values to [0, 1].
    """
    try:
        image = Image.open(png_path)
        image = image.resize((224, 224))
        image = image.convert("L")
        image_array = np.array(image, dtype=np.float32) / 255.0
        return image_array
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {png_path}: {e}")

def pcd_to_numpy(pcd_path):
    """
    Load a PCD file and convert it to a NumPy array.
    (Uses Open3D for file reading only.)
    """
    try:
        pc = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pc.points)
    except Exception as e:
        raise RuntimeError(f"Failed to load PCD from {pcd_path}: {e}")

def trans_to_numpy(json_path):
    """
    Load a JSON file and convert it to a 4x4 transformation matrix (NumPy array).
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {json_path}: {e}")

def load_fake_semantics():
    """
    Process each 'pair_*' folder:
      - Load the semantic image, lidar point cloud, and transformation.
      - Compute the semantic projection and, if applicable, transform the valid points.
      - Only keep points that received a valid semantic score.
      - Merge the results into a common NumPy array with columns [x, y, z, score].
    Finally, visualize the merged cloud using Open3D, coloring points based on their score.
    """
    # Initialize an empty merged cloud as a NumPy array with 4 columns (x, y, z, score)
    merged_cloud = np.empty((0, 4), dtype=np.float32)

    # Load camera configuration.
    calib_file = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/configs/camera_front_right_2_lidar.json"
    cam2lidar, cam_intrisics = load_cam_config(calib_file)
    fx, fy, cx, cy = cam_intrisics

    # Rescale camera intrinsics for 224x224.
    calib_height, calib_width = 1080, 1920
    scale_x = 224 / calib_width
    scale_y = 224 / calib_height
    camera_matrix = np.array([
        [fx * scale_x, 0,            cx * scale_x],
        [0,            fy * scale_y, cy * scale_y],
        [0,            0,            1]
    ], dtype=np.float64)
    
    # Define adjustment matrices (to fix camera pose).
    adjustment_matrix = create_adjustment_matrix(0.0, 0.0, 0.0, 0.45, 0.0, 0.0)
    adjustment_matrix2 = create_adjustment_matrix(0.0, 0.0, 0.0, 0.0, -0.23, 0)
    adjustment_matrix = adjustment_matrix @ adjustment_matrix2

    base_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/"
    folders = os.listdir(base_path)
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        # Process only folders following the "pair_" naming convention.
        if os.path.isdir(folder_path) and folder.startswith("pair_"):
            print(f"Processing folder: {folder}")
            semantic_path = os.path.join(folder_path, "semantic.png")
            semantic_array = png_to_numpy(semantic_path)

            pc_path = os.path.join(folder_path, "pointcloud.pcd")
            lidar_pc = pcd_to_numpy(pc_path)

            trans_path = os.path.join(folder_path, "transformation.json")
            transformation = trans_to_numpy(trans_path)

            # Update merged_cloud with only the valid new points and their semantic scores.
            merged_cloud = iteration(semantic_array, lidar_pc, transformation,
                                     merged_cloud, camera_matrix, adjustment_matrix, cam2lidar)
        else:
            print(f"Skipping non-directory item: {folder}")

    # Final visualization: color points based on their semantic score.
    if merged_cloud.shape[0] > 0:
        points = merged_cloud[:, :3]
        scores = merged_cloud[:, 3]

        # Normalize scores to [0,1] for colormap mapping.
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        colors = cm.get_cmap('viridis')(norm_scores)[:, :3]  # Get RGB from colormap

        # Create an Open3D point cloud for visualization only.
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        print("Visualizing final merged point cloud with RGB colors based on semantic scores...")
        o3d.visualization.draw_geometries([pc])
    else:
        print("No points in the merged cloud.")
    
    # save the merged cloud as numpy array
    merged_cloud = merged_cloud.astype(np.float32)
    merged_cloud_path = os.path.join(base_path, "merged_scored_cloud.npy")
    np.save(merged_cloud_path, merged_cloud)
    print(f"Saved merged cloud to {merged_cloud_path}")

    
if __name__ == "__main__":
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    load_fake_semantics()

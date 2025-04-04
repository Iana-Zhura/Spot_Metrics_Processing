import os
import sys
import numpy as np
from PIL import Image
import open3d as o3d
import json

# Add parent directory so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_2_pc.project_semantic_2_pc import project_semantic_to_lidar, load_cam_config
from image_2_pc.project_image_2_pc import create_adjustment_matrix

def iteration(semantic_array, lidar_pc, transformation, merged_cloud,
              camera_matrix, adjustment_matrix, cam2lidar):
    """
    1. Use project_semantic_to_lidar to get a projection result (either fitness scores or points).
    2. If the result is 1D (fitness scores), keep all lidar points.
    3. Otherwise, use the projected 3D points directly.
    4. Create a new Open3D point cloud from the selected points.
    5. If this is not the first iteration, transform the new cloud using the provided transformation.
    6. Merge the new point cloud with the existing merged cloud.
    
    Returns:
        The updated merged Open3D point cloud.
    """
    # Get projection result (could be fitness scores or a point cloud)
    pc_trans = project_semantic_to_lidar(
        semantic_array, lidar_pc, camera_matrix, adjustment_matrix, cam2lidar,
        visualize=False, save_pc_pth=None
    )
    
    print(f"Average fitness: {pc_trans.mean()}")
    
    # If the output is 1D, assume it's a fitness score per lidar point.
    # Instead of filtering based on a threshold, we keep all the lidar points.
    if pc_trans.ndim == 1:
        if lidar_pc.shape[0] != pc_trans.shape[0]:
            raise ValueError(f"Mismatch between number of lidar points ({lidar_pc.shape[0]}) "
                             f"and fitness scores ({pc_trans.shape[0]})")
        new_cloud_points = lidar_pc
    else:
        # Otherwise, if a 2D array of points was returned, ensure it is (N,3)
        if pc_trans.ndim == 2 and pc_trans.shape[0] == 3:
            pc_trans = pc_trans.T
        if pc_trans.ndim != 2 or pc_trans.shape[1] != 3:
            raise ValueError(f"Unexpected shape for projected point cloud: {pc_trans.shape}")
        new_cloud_points = pc_trans

    # Create an Open3D point cloud from the selected points.
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_cloud_points)
    
    # If merged_cloud is not empty, transform new_cloud into the common frame.
    if len(merged_cloud.points) > 0:
        new_cloud.transform(transformation)
    
    # Merge: if merged_cloud is empty, use new_cloud; else, concatenate the points.
    if len(merged_cloud.points) == 0:
        merged = new_cloud
    else:
        merged_points = np.vstack((
            np.asarray(merged_cloud.points),
            np.asarray(new_cloud.points)
        ))
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(merged_points)
    
    return merged

def png_to_numpy(png_path):
    """
    Load a PNG image, resize to 224x224, convert to grayscale,
    and normalize to [0, 1].
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
    """
    try:
        pc = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pc.points)
    except Exception as e:
        raise RuntimeError(f"Failed to load PCD from {pcd_path}: {e}")

def trans_to_numpy(json_path):
    """
    Load a JSON file and convert it to a transformation matrix (NumPy array).
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {json_path}: {e}")

def load_fake_semantics():
    # Initialize an empty merged point cloud (as an Open3D point cloud).
    merged_cloud = o3d.geometry.PointCloud()

    # ---------------------
    # Load camera intrinsics and extrinsics.
    calib_file = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/configs/camera_front_right_2_lidar.json"
    cam2lidar, cam_intrisics = load_cam_config(calib_file)
    fx, fy, cx, cy = cam_intrisics

    # --------------------
    # Rescale camera intrinsics for semantic predictions at 224x224.
    calib_height, calib_width = 1080, 1920
    scale_x = 224 / calib_width
    scale_y = 224 / calib_height
    camera_matrix = np.array([
        [fx * scale_x, 0,            cx * scale_x],
        [0,            fy * scale_y, cy * scale_y],
        [0,            0,            1]
    ], dtype=np.float64)
    
    # --------------------
    # Define adjustment matrices (fix camera pose).
    tx, ty, tz = 0.0, 0.0, 0.0
    angle_x, angle_y, angle_z = 0.45, 0.0, 0.0
    adjustment_matrix = create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z)
    adjustment_matrix2 = create_adjustment_matrix(0.0, 0.0, 0.0, 0.0, -0.23, 0)
    adjustment_matrix = adjustment_matrix @ adjustment_matrix2

    base_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/"
    folders = os.listdir(base_path)
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        # Process only folders that follow the "pair_" naming convention.
        if os.path.isdir(folder_path) and folder.startswith("pair_"):
            print(f"Processing folder: {folder}")
            # Load semantic image.
            semantic_path = os.path.join(folder_path, "semantic.png")
            semantic_array = png_to_numpy(semantic_path)
            # Load lidar point cloud as a NumPy array.
            pc_path = os.path.join(folder_path, "pointcloud.pcd")
            lidar_pc = pcd_to_numpy(pc_path)
            # Load transformation matrix.
            trans_path = os.path.join(folder_path, "transformation.json")
            transformation = trans_to_numpy(trans_path)

            # Update the merged cloud using the iteration function.
            merged_cloud = iteration(semantic_array, lidar_pc, transformation,
                                     merged_cloud, camera_matrix, adjustment_matrix, cam2lidar)
        else:
            print(f"Skipping non-directory item: {folder}")

    # Visualize the final merged point cloud.
    if len(merged_cloud.points) > 0:
        print("Visualizing final merged point cloud...")
        o3d.visualization.draw_geometries([merged_cloud])
    else:
        print("No points in the merged cloud.")

if __name__ == "__main__":
    os.environ['XDG_SESSION_TYPE'] = 'x11'

    load_fake_semantics()

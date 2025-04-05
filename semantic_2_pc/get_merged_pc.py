from utils.path_simulation.semantic_2_pc.running_scored_pc_merge import iteration
from utils.path_simulation.image_2_pc.project_image_2_pc import create_adjustment_matrix
from utils.path_simulation.semantic_2_pc.project_semantic_2_lidar import load_cam_config
import numpy as np


def merge_pc(merged_cloud, pc, patch_scores, transform, calib_file):
    """
    Merges point clouds with semantic scores.
    Args:
        calib_file (str): Path to the calibration file.
        patch_scores (numpy.ndarray): Array of semantic scores.
        pc (numpy.ndarray): Point cloud data.
        transform (numpy.ndarray): Transformation matrix for the point cloud.
    Returns:
        merged_cloud (numpy.ndarray): Merged point cloud with semantic scores.
    """
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

    # Update the merged cloud using the iteration function.
    merged_cloud = iteration(patch_scores, pc, transform,
                            merged_cloud, camera_matrix, adjustment_matrix, cam2lidar)
    return merged_cloud
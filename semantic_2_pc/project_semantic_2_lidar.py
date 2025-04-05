import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os

def pcd_2_numpy(pc_path):
    """
    Load a point cloud and convert it to a NumPy array.
    """
    pc = o3d.io.read_point_cloud(pc_path)
    pc_np = np.asarray(pc.points)
    return pc_np

def read_json_file(file_path, top_level_key=None, nested_key=None):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        if top_level_key is not None:
            if top_level_key in data:
                if nested_key is not None:
                    if nested_key in data[top_level_key]:
                        return data[top_level_key][nested_key]
                    else:
                        print(f"Nested key '{nested_key}' not found inside '{top_level_key}'.")
                        return None
                else:
                    return data[top_level_key]
            else:
                print(f"Top-level key '{top_level_key}' not found in JSON data.")
                return None
        else:
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None

def load_cam_config(file_path):
    cam2lidar_list = read_json_file(file_path, top_level_key="results", nested_key="T_lidar_camera")
    
    if cam2lidar_list is None:
        raise ValueError(f"Failed to load camera configuration from {file_path}.")
    
    # start with cam2lidar_T
    cam2lidar_T = np.array(cam2lidar_list[:3], dtype=np.float64)
    cam2lidar_Q = np.array(cam2lidar_list[3:], dtype=np.float64)
    cam2lidar_R = R.from_quat(cam2lidar_Q).as_matrix()
    cam2lidar = np.eye(4, dtype=np.float64)
    cam2lidar[:3, :3] = cam2lidar_R
    cam2lidar[:3, 3] = cam2lidar_T

    # then the intrinsics
    cam_intrinsics = read_json_file(file_path, top_level_key="camera", nested_key="intrinsics")
    if cam_intrinsics is None:
        raise ValueError(f"Failed to load camera intrinsics from {file_path}.")
    
    cam_intrinsics = np.array(cam_intrinsics, dtype=np.float64)

    return cam2lidar, cam_intrinsics


def create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z):
    """
    Attention: This function does rotations in the order:
    Roll (x), Yaw (z), Pitch (y)
    """
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x),  np.cos(angle_x)]
    ])
    ry = np.array([
        [ np.cos(angle_y), 0, np.sin(angle_y)],
        [0,               1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0,               0,               1]
    ])
    R_combined = ry @ rz @ rx
    adjustment_matrix = np.eye(4)
    adjustment_matrix[:3, :3] = R_combined
    adjustment_matrix[:3, 3] = [tx, ty, tz]
    return adjustment_matrix

def project_semantic_to_lidar(patch_scores, points, camera_matrix, adjustment_matrix, cam2lidar, visualize=False, save_pc_pth=None):
    """
    Project semantic scores from camera space to LiDAR point cloud space.
    All computations use NumPy. By default, each point’s semantic score is set to NaN,
    and only points that project within the image bounds receive a valid score.
    
    Args:
        patch_scores: 2D NumPy array (grayscale image) of size (H,W) with values in [0,1].
        points: (N,3) NumPy array of LiDAR points.
        camera_matrix: 3x3 NumPy array with rescaled intrinsics.
        adjustment_matrix: 4x4 NumPy array.
        cam2lidar: 4x4 NumPy array representing the camera-to-LiDAR transformation.
        visualize: if True, show a visualization (using Open3D and OpenCV).
        save_pc_pth: path to save the visualization point cloud (only if visualize is True).
    
    Returns:
        trans_fitness: A NumPy array of shape (N,) containing the semantic score for each point.
                       Points that do not receive a valid projection remain as NaN.
    """
    target_height, target_width = patch_scores.shape

    # Transform points from LiDAR frame to camera frame.
    inv_mat = np.linalg.inv(cam2lidar)
    cam_points = (inv_mat[:3, :3] @ points.T).T + inv_mat[:3, 3]
    
    homog_points = np.hstack((cam_points, np.ones((cam_points.shape[0], 1))))
    adjusted_points = (adjustment_matrix @ homog_points.T).T[:, :3]

    # Project adjusted points using the rescaled camera intrinsics.
    uvw = (camera_matrix @ adjusted_points.T).T
    
    # Initialize semantic scores as NaN (i.e. unassigned)
    trans_fitness = np.full((points.shape[0],), np.nan, dtype=np.float32)

    if visualize:
        import cv2
        visual_colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))
        patch_scores_uint8 = (patch_scores * 255).astype(np.uint8)
        patch_heatmap = cv2.applyColorMap(patch_scores_uint8, cv2.COLORMAP_RAINBOW)
        patch_heatmap = cv2.cvtColor(patch_heatmap, cv2.COLOR_BGR2RGB)

    for i in range(points.shape[0]):
        z = uvw[i, 2]
        if z > 0:
            u = uvw[i, 0] / z
            v = uvw[i, 1] / z
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < target_width and 0 <= v_int < target_height:
                trans_fitness[i] = patch_scores[v_int, u_int]
                if visualize:
                    visual_colors[i] = patch_heatmap[v_int, u_int] / 255.0

    if visualize:
        final_pc = o3d.geometry.PointCloud()
        final_pc.points = o3d.utility.Vector3dVector(adjusted_points)
        final_pc.colors = o3d.utility.Vector3dVector(visual_colors)
        
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        transformed_frame.transform(adjustment_matrix)
        
        o3d.visualization.draw_geometries(
            [final_pc, origin_frame, transformed_frame],
            window_name="Semantic Projection on LiDAR"
        )
        
        if save_pc_pth is not None:
            o3d.io.write_point_cloud(save_pc_pth, final_pc)
            print(f"Saved LiDAR semantic point cloud to {save_pc_pth}")

    return trans_fitness


def main():
    # --------------------
    # Paths and parameters
    number = '000'
    base = f"/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/pair_0{number}"
    lidar_file = f"{base}/pointcloud.pcd"   # could also be a .npy file
    output_file = f"{base}/projected_semantic.pcd"
    
    # --------------------
    # Load semantic patch scores (original predicted values in [0,1] with shape (224,224))
    semantic_path = f"{base}/patch_scores.npy"
    patch_scores = np.load(semantic_path, allow_pickle=True)
    print(f"Loaded patch scores shape: {patch_scores.shape}")
    # The semantic image size (target) is 224 x 224.
    target_height, target_width = patch_scores.shape

    # ---------------------
    # #2 load camera intrinsics and extrinsics
    calib_file = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/configs/camera_front_right_2_lidar.json"
    cam2lidar, cam_intrisics = load_cam_config(calib_file)
    fx, fy = cam_intrisics[0], cam_intrisics[1]
    cx, cy = cam_intrisics[2], cam_intrisics[3]
    
    # --------------------
    # #3 Rescale camera intrinsics for semantic predictions at 224x224.
    calib_height, calib_width = 1080, 1920
    scale_x = target_width / calib_width
    scale_y = target_height / calib_height
    camera_matrix = np.array([
        [fx * scale_x,  0,               cx * scale_x],
        [0,             fy * scale_y,    cy * scale_y],
        [0,             0,               1]
    ], dtype=np.float64)
    
    # --------------------
    # #4 Define by hand adjustment matrices (fix the camera pose)
    tx, ty, tz = 0.0, 0.0, 0.0
    angle_x, angle_y, angle_z = 0.45, -0.0, -0.0
    adjustment_matrix = create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z)
    adjustment_matrix2 = create_adjustment_matrix(tx=0.0, ty=-0.0, tz=0.0, angle_x=0.0, angle_y=-0.23, angle_z=0)
    adjustment_matrix = adjustment_matrix @ adjustment_matrix2
    
    # --------------------
    # #5 Load LiDAR point cloud as a NumPy array - in Iana's case expect a numpy array input
    ext = os.path.splitext(lidar_file)[1]
    if ext == ".npy":
        lidar_pc = np.load(lidar_file)
    else:
        lidar_pc = pcd_2_numpy(lidar_file)
    
    if lidar_pc is None or lidar_pc.shape[0] == 0:
        print("Error: Empty point cloud.")
        return

    # --------------------
    # #6 Project semantic scores to LiDAR point cloud 
    trans_fitness_pc = project_semantic_to_lidar(patch_scores, lidar_pc, camera_matrix, adjustment_matrix, cam2lidar, visualize=True, save_pc_pth=output_file)
    # --------------------

    
if __name__ == "__main__":
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    main()
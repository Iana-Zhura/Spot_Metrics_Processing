#!/usr/bin/env python
import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

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

def load_lidar_cam_calibration(file_path):
    # Expects a JSON with top-level "results" and nested key "T_lidar_camera"
    cam2lidar_list = read_json_file(file_path, top_level_key="results", nested_key="T_lidar_camera")
    if cam2lidar_list is None:
        return None
    cam2lidar_T = np.array(cam2lidar_list[:3], dtype=np.float64)
    cam2lidar_Q = np.array(cam2lidar_list[3:], dtype=np.float64)
    cam2lidar_R = R.from_quat(cam2lidar_Q).as_matrix()
    cam2lidar = np.eye(4, dtype=np.float64)
    cam2lidar[:3, :3] = cam2lidar_R
    cam2lidar[:3, 3] = cam2lidar_T
    return cam2lidar

def create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z):
    """
    Create a 4x4 homogeneous transformation matrix that applies rotations
    (around x, y, and z in radians) and a translation.
    """
    r = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=False)
    adjustment_matrix = np.eye(4)
    adjustment_matrix[:3, :3] = r.as_matrix()
    adjustment_matrix[:3, 3] = [tx, ty, tz]
    return adjustment_matrix

def process_pair(pair_dir, calib_file, calib_resolution, robot_model, tx, ty, tz, angle_x, angle_y, angle_z):
    # Assume the pair folder contains:
    #   - LiDAR point cloud: "pointcloud.pcd"
    #   - Image: "semantic.png"
    lidar_file = os.path.join(pair_dir, "pointcloud.pcd")
    image_file = os.path.join(pair_dir, "semantic.png")
    if not os.path.exists(lidar_file) or not os.path.exists(image_file):
        print(f"Skipping {pair_dir}: missing pointcloud.pcd or image.png")
        return

    # Load LiDAR point cloud
    lidar_pc = o3d.io.read_point_cloud(lidar_file)
    if lidar_pc.is_empty():
        print(f"Empty point cloud in {pair_dir}")
        return

    # Load image (assumed to be in BGR; convert to RGB)
    bgr_image = cv2.imread(image_file)
    if bgr_image is None:
        print(f"Cannot load image in {pair_dir}")
        return
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Build camera intrinsics from calibration resolution to current image size.
    calib_height, calib_width = calib_resolution  # e.g., (1544, 2064)
    image_height, image_width = rgb_image.shape[:2]
    scale_x = image_width / calib_width
    scale_y = image_height / calib_height

    # According to the shared config, assume intrinsics are [fx, fy, cx, cy]:
    fx = 1008.1348616101474
    fy = 1009.6301881044601
    cx = 1261.7336458970644
    cy = 1050.5986007075933
    camera_matrix = np.array([
        [fx * scale_x, 0,             cx * scale_x],
        [0,            fy * scale_y,  cy * scale_y],
        [0,            0,             1]
    ], dtype=np.float64)

    # Load calibration matrix (assume it's provided as camera→LiDAR; invert for Spot)
    cam2lidar = load_lidar_cam_calibration(calib_file)
    if cam2lidar is None:
        print(f"Cannot load calibration in {pair_dir}")
        return

    tx, ty, tz = 0.0, 0.0, 0.0
    angle_x, angle_y, angle_z = 0.46, -0.0, -0.0
    adjustment_matrix = create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z)
    adjustment_matrix2 = create_adjustment_matrix(tx=0.0, ty=-0.0, tz=0.0, angle_x=0.0, angle_y=-0.23, angle_z=0)

    adjustment_matrix = adjustment_matrix @ adjustment_matrix2

    # Transform the entire LiDAR point cloud to camera coordinates.
    points = np.asarray(lidar_pc.points)
    if robot_model == "Spot":
        inv_mat = np.linalg.inv(cam2lidar)
        cam_points = (inv_mat[:3, :3] @ points.T).T + inv_mat[:3, 3]
    else:
        cam_points = (cam2lidar[:3, :3] @ points.T).T + cam2lidar[:3, 3]

    # Apply manual adjustment.
    homog_points = np.hstack((cam_points, np.ones((cam_points.shape[0], 1))))
    adjusted_points = (adjustment_matrix @ homog_points.T).T[:, :3]

    # Project adjusted points using camera intrinsics.
    uvw = (camera_matrix @ adjusted_points.T).T

    # Get image dimensions.
    h, w = rgb_image.shape[:2]
    # Prepare an array for colors. Default remains for points that do not fall inside the image.
    default_color = np.array([0.5, 0.5, 0.5])
    colors = np.tile(default_color, (adjusted_points.shape[0], 1))
    
    # Only keep points that are valid (i.e. that get a proper color from the image).
    valid_indices = []
    for i in range(adjusted_points.shape[0]):
        z = uvw[i, 2]
        if z > 0:
            u = uvw[i, 0] / z
            v = uvw[i, 1] / z
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                colors[i] = rgb_image[v_int, u_int] / 255.0
                valid_indices.append(i)
                
    if len(valid_indices) == 0:
        print(f"No valid points assigned a color in {pair_dir}")
        return

    # Filter the original LiDAR points and colors to include only valid points.
    points_valid = points[valid_indices]
    colors_valid = colors[valid_indices]

    # Create the final colored point cloud using only the valid points.
    colored_pc = o3d.geometry.PointCloud()
    colored_pc.points = o3d.utility.Vector3dVector(points_valid)
    colored_pc.colors = o3d.utility.Vector3dVector(colors_valid)

    # Save the resulting colored point cloud in the same folder.
    projected_pcd_path = os.path.join(pair_dir, "projected_colored.pcd")
    o3d.io.write_point_cloud(projected_pcd_path, colored_pc)
    print(f"Saved projected colored point cloud in {pair_dir} as 'projected_colored.pcd'")

def main():
    # Root directory containing each pair folder.
    root_dir = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/"
    # Calibration file and resolution (from supervisor’s configuration)
    calib_file = "camera_front_right_2_lidar.json"
    calib_resolution =  (1080, 1920)  # (height, width)
    # Robot model: "Spot"
    robot_model = "Spot"
    # Manual adjustment parameters (tweak as needed)
    tx = 0.0
    ty = 0.0
    tz = 0.0
    angle_x = 0.4
    angle_y = -0.2
    angle_z = -0.1

    # Process each subdirectory (each pair folder) in the root directory.
    for item in os.listdir(root_dir):
        pair_dir = os.path.join(root_dir, item)
        if os.path.isdir(pair_dir):
            print(f"Processing pair folder: {pair_dir}")
            process_pair(pair_dir, calib_file, calib_resolution, robot_model,
                         tx, ty, tz, angle_x, angle_y, angle_z)

if __name__ == "__main__":
    main()

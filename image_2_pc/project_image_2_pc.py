#!/usr/bin/env python
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

def create_wall_point_cloud(width, height, resolution=0.05):
    """
    Creates a wall point cloud as a grid of points on a plane in the XY plane.
    The wall is initially created at z=0 and will be translated to a fixed depth.
    """
    xs = np.arange(-width/2, width/2, resolution)
    ys = np.arange(-height/2, height/2, resolution)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.zeros_like(xv)  # Initially on the plane z = 0
    points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
    wall_pc = o3d.geometry.PointCloud()
    wall_pc.points = o3d.utility.Vector3dVector(points)
    return wall_pc

def main():
    # Paths
    number = '345'
    base = f"/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/images/pair_0{number}"
    lidar_file = f"{base}/pointcloud.pcd"
    image_file = f"{base}/semantic.png"
    output_file = f"{base}/projected_colored.pcd"
    
    calib_file = "camera_front_right_2_lidar.json"
    robot_model = "Spot"

    rgb_image = cv2.imread(image_file)
    if rgb_image is None:
        print("Error: Could not load image.")
        return

    print(f"Loaded image shape: {rgb_image.shape}")

    calib_height, calib_width = 1080, 1920
    image_height, image_width = rgb_image.shape[:2]
    scale_x = image_width / calib_width
    scale_y = image_height / calib_height

    camera_matrix = np.array([
        [1008.1348616101474 * scale_x, 0, 1261.7336458970644 * scale_x],
        [0, 1009.6301881044601 * scale_y, 1050.5986007075933 * scale_y],
        [0, 0, 1]
    ], dtype=np.float64)

    tx, ty, tz = 0.0, 0.0, 0.0
    angle_x, angle_y, angle_z = 0.45, -0.0, -0.0
    adjustment_matrix = create_adjustment_matrix(tx, ty, tz, angle_x, angle_y, angle_z)
    adjustment_matrix2 = create_adjustment_matrix(tx=0.0, ty=-0.0, tz=0.0, angle_x=0.0, angle_y=-0.23, angle_z=0)

    adjustment_matrix = adjustment_matrix @ adjustment_matrix2

    # Load point cloud
    lidar_pc = o3d.io.read_point_cloud(lidar_file)
    if lidar_pc.is_empty():
        print("Error: Empty point cloud.")
        return
    cam2lidar = load_lidar_cam_calibration(calib_file)
    if cam2lidar is None:
        print("Error: Calibration load failed.")
        return

    # Transform LiDAR to camera coordinate system
    points = np.asarray(lidar_pc.points)
    if robot_model == "Spot":
        inv_mat = np.linalg.inv(cam2lidar)
        cam_points = (inv_mat[:3, :3] @ points.T).T + inv_mat[:3, 3]
    else:
        cam_points = (cam2lidar[:3, :3] @ points.T).T + cam2lidar[:3, 3]

    homog_points = np.hstack((cam_points, np.ones((cam_points.shape[0], 1))))
    adjusted_points = (adjustment_matrix @ homog_points.T).T[:, :3]

    # Project LiDAR points to image and color them
    uvw = (camera_matrix @ adjusted_points.T).T
    default_color = np.array([0.5, 0.5, 0.5])
    colors = np.tile(default_color, (points.shape[0], 1))
    h, w = rgb_image.shape[:2]

    for i in range(points.shape[0]):
        z = uvw[i, 2]
        if z > 0:
            u = uvw[i, 0] / z
            v = uvw[i, 1] / z
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                colors[i] = rgb_image[v_int, u_int] / 255.0

    final_pc = o3d.geometry.PointCloud()
    final_pc.points = o3d.utility.Vector3dVector(adjusted_points)
    final_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_file, final_pc)
    print(f"Saved LiDAR RGB point cloud to {output_file}")

    # === WALL ===
    # Define wall dimensions and resolution
    wall_width = 6     # Width of the wall in meters
    wall_height = 4    # Height of the wall in meters
    wall_depth = 2     # Fixed depth (distance along z) from the camera
    wall_resolution = 0.05  # Spacing between points

    # Create the wall in the camera coordinate system.
    wall_pc = create_wall_point_cloud(wall_width, wall_height, wall_resolution)
    # Set the entire wall to be at the fixed depth along the z axis.
    wall_points = np.asarray(wall_pc.points)
    wall_points[:, 2] = wall_depth
    wall_pc.points = o3d.utility.Vector3dVector(wall_points)

    # Compute the inverse of the full 4x4 adjustment matrix.
    T_inv = adjustment_matrix

    # Convert wall points to homogeneous coordinates.
    wall_points_h = np.hstack((wall_points, np.ones((wall_points.shape[0], 1))))

    # Apply the inverse transformation.
    wall_points_fixed = (T_inv @ wall_points_h.T).T[:, :3]
    wall_pc.points = o3d.utility.Vector3dVector(wall_points_fixed)

    # Project RGB onto the wall
    wall_uvw = (camera_matrix @ wall_points.T).T
    wall_colors = np.tile(default_color, (wall_points.shape[0], 1))
    for i in range(wall_points.shape[0]):
        z = wall_uvw[i, 2]
        if z > 0:
            u = wall_uvw[i, 0] / z
            v = wall_uvw[i, 1] / z
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                wall_colors[i] = rgb_image[v_int, u_int] / 255.0

    wall_pc.colors = o3d.utility.Vector3dVector(wall_colors)

    # === Coordinate Frames for Visualization ===
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    transformed_frame.transform(adjustment_matrix)
    
    # A frame for the wall (optional)
    wall_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # === Visualization ===
    o3d.visualization.draw_geometries([
        final_pc, origin_frame, transformed_frame, wall_frame
    ], window_name="RGB Projection on LiDAR + Wall")

if __name__ == "__main__":
    import os
    os.environ['XDG_SESSION_TYPE'] = 'x11'
    main()

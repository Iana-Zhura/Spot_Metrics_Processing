#!/usr/bin/env python
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import argparse

# Parse arguments
parser = argparse.ArgumentParser(
    description="Merge pre-colored point clouds using a running (relative) transformation computed from odometry."
)
parser.add_argument("--all", action="store_true", help="Process all point clouds instead of just selected indices")
args = parser.parse_args()

recording_name = "greenhouse_march_24"

if recording_name == "greenhouse_march_6":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings/march_6/v1/march_6/greenhouse_march.bag"
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings/march_6/v1/merged_cloud_selected.pcd"
elif recording_name == "greenhouse_march_11":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings/march_11/march11_greenhouse.bag"
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings/march_11/merged_cloud_selected.pcd"
elif recording_name == "greenhouse_march_24":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/recordings/march24_greenhouse.bag"
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_colored.pcd"

# Topics
POINT_CLOUD_TOPIC = "/ouster/points"
ODOMETRY_TOPIC = "/Odometry"

# Storage for data: record only timestamp and count for each selected point cloud.
point_clouds = []
poses = []

# Select which point clouds to process (if not using --all)
selected_indices = None if args.all else {0, 100, 200, 300, 360, 410}
# For testing, override with a smaller set:
selected_indices = {0, 20, 40, 60, 80, 100,
                    120, 140, 160, 180, 200,
                    220, 240, 260, 280, 300,
                    325, 345, 360, 380, 400,
                    420, 445, 460, 480}

# Read bag file: record timestamps for selected point clouds and odometry poses.
with rosbag.Bag(BAG_FILE, 'r') as bag:
    count = 0
    for topic, msg, t in bag.read_messages():
        if topic == POINT_CLOUD_TOPIC:
            if selected_indices is None or count in selected_indices:
                cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
                point_clouds.append((t.to_sec(), cloud, count))
            count += 1
        elif topic == ODOMETRY_TOPIC:
            pose = msg.pose.pose
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            
            # Normalize the quaternion
            orientation /= np.linalg.norm(orientation)
            
            # Convert quaternion to rotation matrix
            rotation_matrix = tf_trans.quaternion_matrix(orientation)[:3, :3]
            
            # Construct full transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position
            poses.append((t.to_sec(), transform_matrix))

def find_closest_pose(timestamp):
    return min(poses, key=lambda x: abs(x[0] - timestamp))[1]

# Base folder where the pre-colored point clouds are stored.
# Each pair folder is assumed to be named "pair_XXXX" (with the count zero-padded)
BASE_FOLDER = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/processings/images"

merged_points_list = []
merged_colors_list = []

# Determine the reference transformation from the first selected point cloud.
if len(point_clouds) == 0:
    print("No point clouds found in the bag.")
    exit(1)
T_ref = find_closest_pose(point_clouds[0][0])
print("Reference transformation (T_ref):")
print(T_ref)

for idx, (timestamp, _, count) in enumerate(point_clouds):
    # Get the absolute pose for the current point cloud.
    T_current = find_closest_pose(timestamp)
    # Compute the relative transformation with respect to the reference.
    T_relative = np.linalg.inv(T_ref) @ T_current
    print(f"Relative transformation for Cloud {count} (T_relative):")
    print(T_relative)
    
    # Build the path to the pre-colored point cloud (expected name: "projected_colored.pcd")
    folder_path = os.path.join(BASE_FOLDER, f"pair_{count:04d}")
    pcd_file = os.path.join(folder_path, "projected_colored.pcd")
    if not os.path.exists(pcd_file):
        print(f"File {pcd_file} not found. Skipping cloud {count}.")
        continue
    loaded_cloud = o3d.io.read_point_cloud(pcd_file)
    if loaded_cloud.is_empty():
        print(f"Point cloud {pcd_file} is empty. Skipping cloud {count}.")
        continue

    # Retrieve original points and colors (preserve the RGB values).
    points = np.asarray(loaded_cloud.points)
    original_colors = np.asarray(loaded_cloud.colors)
    if original_colors.shape[0] == 0:
        print(f"No color information in {pcd_file}. Skipping cloud {count}.")
        continue
    
    # Apply the relative transformation to the points.
    homogeneous_cloud = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (T_relative @ homogeneous_cloud.T).T[:, :3]
    
    # Accumulate the transformed points and original colors.
    merged_points_list.append(transformed_points)
    merged_colors_list.append(original_colors)

if len(merged_points_list) == 0:
    print("No point clouds were merged.")
else:
    merged_points = np.vstack(merged_points_list)
    merged_colors = np.vstack(merged_colors_list)
    
    merged_cloud = o3d.geometry.PointCloud()
    merged_cloud.points = o3d.utility.Vector3dVector(merged_points)
    merged_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    
    o3d.io.write_point_cloud(OUTPUT_PCD, merged_cloud)
    print(f"Saved merged point cloud with {len(merged_points)} points as: {OUTPUT_PCD}")

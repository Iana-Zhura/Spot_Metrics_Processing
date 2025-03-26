import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process ROS bag point clouds.")
parser.add_argument("--all", action="store_true", help="Save all point clouds instead of just the first and X-th")
args = parser.parse_args()

BAG_FILE = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_asphalt_imtek/cl_imtek_asphalt.bag"

# Topics
POINT_CLOUD_TOPIC = "/ouster/points"
ODOMETRY_TOPIC = "/Odometry"

# Output file
OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_asphalt_imtek/merged_cloud_selected.pcd"

# Storage for data
point_clouds = []
poses = []

# Select which point clouds to save
X_TH = 200  # Change this value as needed
selected_indices = None if args.all else {0, X_TH}  # Save all if --all is provided

# Colors for different point clouds
colors = [np.array([1, 0, 0]), np.array([0, 0, 1])]  # Red for first, Blue for X-th

# Read bag file
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

# Function to find closest odometry for each point cloud
def find_closest_pose(timestamp):
    return min(poses, key=lambda x: abs(x[0] - timestamp))[1]

# Merge selected point clouds
merged_cloud = o3d.geometry.PointCloud()

for idx, (timestamp, cloud, count) in enumerate(point_clouds):
    transform_matrix = find_closest_pose(timestamp)
    
    print(f"Transform Matrix for Cloud {count}:")
    print(transform_matrix)
    
    homogeneous_cloud = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
    transformed_cloud = (transform_matrix @ homogeneous_cloud.T).T[:, :3]  # Correct multiplication order

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(transformed_cloud)
    
    # Assign colors based on index
    color = colors[idx % len(colors)] if selected_indices is not None else np.array([0, 1, 0])  # Green if all
    o3d_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (transformed_cloud.shape[0], 1)))
    
    merged_cloud += o3d_cloud

# Save merged point cloud
o3d.io.write_point_cloud(OUTPUT_PCD, merged_cloud)

print(f"Saved merged point cloud as: {OUTPUT_PCD}")
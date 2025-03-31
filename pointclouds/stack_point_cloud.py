import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process ROS bag point clouds.")
parser.add_argument("--all", action="store_true", help="Save all point clouds instead of just the selected indices")
args = parser.parse_args()

recording_name = "greenhouse_march_24"

if recording_name == "greenhouse_march_6":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings/march_6/v1/march_6/greenhouse_march.bag"
    # Output file
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings/march_6/v1/merged_cloud_selected.pcd"
elif recording_name == "greenhouse_march_11":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings/march_11/march11_greenhouse.bag"
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings/march_11/merged_cloud_selected.pcd"
elif recording_name == "greenhouse_march_24":
    BAG_FILE = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/recordings/march24_greenhouse.bag"
    # Output file
    OUTPUT_PCD = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_selected.pcd"

# Topics
POINT_CLOUD_TOPIC = "/ouster/points"
ODOMETRY_TOPIC = "/Odometry"

# Storage for data
point_clouds = []
poses = []

# Select which point clouds to save: 0, 200, 400, 600 if not using --all
selected_indices = None if args.all else {0, 100, 200, 300, 360, 410}
selected_indices = {0, 200, 360}

# Colors for different point clouds (Red, Green, Blue, Yellow, ...) -> 15 colors, after that it will repeat 
colors = [
    np.array([1, 0, 0]),   # Red
    np.array([0, 1, 0]),   # Green
    np.array([0, 0, 1]),   # Blue
    np.array([1, 1, 0]),    # Yellow
    np.array([1, 0, 1]),   # Magenta
    np.array([0, 1, 1]),   # Cyan
    np.array([0.5, 0.5, 0]), # Olive
    np.array([0.5, 0, 0.5]), # Purple
    np.array([0, 0.5, 0.5]), # Teal
    np.array([0.5, 0.5, 0.5]), # Gray
    np.array([1, 0.5, 0]), # Orange
    np.array([0.5, 1, 0]), # Lime
    np.array([0.5, 0, 1]), # Pink
    np.array([1, 0, 0.5]), # Coral
    np.array([0.5, 1, 1]), # Light Blue
]

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

# Function to find the closest odometry for each point cloud based on timestamp
def find_closest_pose(timestamp):
    return min(poses, key=lambda x: abs(x[0] - timestamp))[1]

# Merge selected point clouds
merged_cloud = o3d.geometry.PointCloud()

for idx, (timestamp, cloud, count) in enumerate(point_clouds):
    transform_matrix = find_closest_pose(timestamp)
    
    print(f"Transform Matrix for Cloud {count}:")
    print(transform_matrix)
    
    # Convert cloud to homogeneous coordinates and apply transformation
    homogeneous_cloud = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
    transformed_cloud = (transform_matrix @ homogeneous_cloud.T).T[:, :3]
    
    # Create an Open3D point cloud and assign points
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(transformed_cloud)
    
    # Assign one of the four colors based on the index
    color = colors[idx % len(colors)]
    o3d_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (transformed_cloud.shape[0], 1)))
    
    merged_cloud += o3d_cloud

# Save merged point cloud
o3d.io.write_point_cloud(OUTPUT_PCD, merged_cloud)

print(f"Saved merged point cloud as: {OUTPUT_PCD}")

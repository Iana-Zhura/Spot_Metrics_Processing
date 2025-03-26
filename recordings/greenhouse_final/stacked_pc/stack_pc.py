import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process ROS bag point clouds with debugging.")
parser.add_argument("--all", action="store_true", help="Save all point clouds instead of just the first and X-th")
args = parser.parse_args()

# Paths
BAG_FILE = "../greenhouse_final_2025-01-15-12-44-06_point_cloud_odometry.bag"
ODOMETRY_CSV = "odometry_iana.csv"
OUTPUT_PCD = "merged_cloud_selected_debug.pcd"

# Topics
POINT_CLOUD_TOPIC = "/ouster/points"
ODOMETRY_TOPIC = "/Odometry"

# Storage for data
odometry_data = {}
selected_point_clouds = {}

# Select which odometry readings to save
X_TH = 250  # Adjust as needed
selected_odometry_indices = None if args.all else {0, X_TH}

# Debugging Settings
ENABLE_TRANSFORMATION_DEBUG = True  # Set to True to print debug info
ENABLE_TIMESTAMP_DEBUG = True  # Set to True to check time synchronization

# Transformation matrix between IMU and LiDAR
imu2lidar = np.array([
    [0.998993, -0.044802,  0.002237,  0.1],
    [-0.044815, -0.998974,  0.006547, -0.005956],
    [0.001941,  -0.006641, -0.999976, -0.181793],
    [0.0,       0.0,       0.0,       1.0]
])

# Function to find the closest point cloud to a given odometry timestamp
def find_closest_point_cloud(odometry_timestamp, point_clouds):
    closest_time = min(point_clouds.keys(), key=lambda t: abs(t - odometry_timestamp))
    return closest_time, point_clouds[closest_time]

# Function to transform LiDAR points to the global frame using odometry
def transform_lidar_to_global(lidar_scan, position, orientation=[0, 0, 0, 1]):
    """
    Transforms LiDAR points from LiDAR frame to the global frame using odometry.
    Includes debugging outputs.
    """
    # Normalize quaternion
    orientation = np.array(orientation) / np.linalg.norm(orientation)
    rotation_matrix = R.from_quat(orientation).as_matrix()

    # Odometry transformation matrix (RT)
    RT = np.eye(4)
    RT[:3, :3] = rotation_matrix
    RT[:3, 3] = position  # Set translation

    # Compute the combined transformation (test both orders)
    combined_matrix1 = np.matmul(RT, imu2lidar)  # Direct multiplication
    combined_matrix2 = np.matmul(RT, np.linalg.inv(imu2lidar))  # Using inverse

    # Convert LiDAR points to homogeneous coordinates
    ones = np.ones((lidar_scan.shape[0], 1))
    lidar_homogeneous = np.hstack((lidar_scan, ones))  # Nx4

    # Apply transformation (Test both!)
    lidar_global1 = np.matmul(combined_matrix1, lidar_homogeneous.T).T[:, :3]
    lidar_global2 = np.matmul(combined_matrix2, lidar_homogeneous.T).T[:, :3]

    # Debugging output
    if ENABLE_TRANSFORMATION_DEBUG:
        print("\n==== Debugging Transformation ====")
        print("Odometry Position:", position)
        print("Odometry Quaternion:", orientation)
        print("Rotation Matrix:\n", rotation_matrix)
        print("RT Matrix:\n", RT)
        print("IMU-to-LiDAR Matrix:\n", imu2lidar)
        print("RT * IMU-to-LiDAR:\n", combined_matrix1)
        print("RT * inv(IMU-to-LiDAR):\n", combined_matrix2)

    return lidar_global1  # Change to lidar_global2 if testing inverse transformation

# Load odometry data from CSV
def load_odometry_data(odometry_csv_path):
    odometry_df = pd.read_csv(odometry_csv_path, header=None)
    formatted_data = {}
    for idx, row in odometry_df.iterrows():
        if selected_odometry_indices is None or idx in selected_odometry_indices:
            timestamp = int(float(row[0]) * 1e9)  # Convert seconds to nanoseconds
            position = np.array([float(row[1]), float(row[2]), float(row[3])])
            orientation = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            formatted_data[timestamp] = {"position": position, "orientation": orientation}
    return formatted_data

# Read ROS bag and store closest point clouds
point_clouds = {}

with rosbag.Bag(BAG_FILE, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        timestamp = int(t.to_sec() * 1e9)  # Convert to nanoseconds
        
        if topic == POINT_CLOUD_TOPIC:
            cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            point_clouds[timestamp] = cloud  # Store point clouds indexed by timestamp

# Load odometry data
odometry_data = load_odometry_data(ODOMETRY_CSV)

# Debugging timestamp sync
if ENABLE_TIMESTAMP_DEBUG:
    print("\n==== Debugging Timestamps ====")
    for odom_timestamp in odometry_data.keys():
        closest_pc_timestamp, _ = find_closest_point_cloud(odom_timestamp, point_clouds)
        time_diff = abs(odom_timestamp - closest_pc_timestamp)
        print(f"Odometry: {odom_timestamp}, Closest PC: {closest_pc_timestamp}, Difference: {time_diff} ns")

# Match each odometry reading to the closest point cloud
for odom_timestamp, odom_data in odometry_data.items():
    closest_pc_timestamp, closest_pc = find_closest_point_cloud(odom_timestamp, point_clouds)
    selected_point_clouds[odom_timestamp] = {"timestamp": closest_pc_timestamp, "point_cloud": closest_pc}

# Merge selected point clouds
merged_cloud = o3d.geometry.PointCloud()

for idx, (odom_timestamp, data) in enumerate(selected_point_clouds.items()):
    transformed_cloud = transform_lidar_to_global(
        data["point_cloud"],
        odometry_data[odom_timestamp]["position"],
        odometry_data[odom_timestamp]["orientation"]
    )

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(transformed_cloud)
    
    # Assign color for debugging visualization
    color = np.array([1, 0, 0]) if idx % 2 == 0 else np.array([0, 0, 1])  # Red / Blue alternation
    o3d_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (transformed_cloud.shape[0], 1)))
    
    merged_cloud += o3d_cloud

# Save merged point cloud
o3d.io.write_point_cloud(OUTPUT_PCD, merged_cloud)

print(f"\n==== Debugging Completed ====")
print(f"Saved merged point cloud as: {OUTPUT_PCD}")

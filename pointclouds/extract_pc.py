import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

# Path to your ROS bag
BAG_FILE = "/root/shared_folder/greenhouse_final/greenhouse_final_2025-01-15-12-44-06_point_cloud_odometry.bag"

# Topics
POINT_CLOUD_TOPIC = "/ouster/points"

# Output file
OUTPUT_PLY = "first_point_cloud.ply"
OUTPUT_PCD = "first_point_cloud.pcd"

# Read bag file and extract first point cloud
first_point_cloud = None
with rosbag.Bag(BAG_FILE, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == POINT_CLOUD_TOPIC:
            cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            first_point_cloud = cloud
            break

# Save first point cloud as PLY
if first_point_cloud is not None:
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(first_point_cloud)
    o3d.io.write_point_cloud(OUTPUT_PLY, o3d_cloud)
    print(f"Saved first point cloud as: {OUTPUT_PLY}")
    o3d.io.write_point_cloud(OUTPUT_PCD, o3d_cloud)
    print(f"Saved first point cloud as: {OUTPUT_PCD}")
else:
    print("No point cloud data found in the bag file.")

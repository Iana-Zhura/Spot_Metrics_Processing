# this is very ugly and opens to local host

import open3d as o3d

# Enable WebRTC visualization
o3d.visualization.webrtc_server.enable_webrtc()

# Load the point cloud
ply_path = "/media/martin/Elements/ros-recordings/pointclouds/first_point_cloud.ply"
pcd = o3d.io.read_point_cloud(ply_path)

# Visualize using WebRTC
o3d.visualization.draw([pcd])
#!/usr/bin/env python
import open3d as o3d
import numpy as np
import argparse

def smooth_point_cloud(pc, radius):
    """
    For each point in the point cloud, find all neighboring points within
    the given radius and compute the average position and color (if available).
    Return a new smoothed point cloud.
    """
    # Build a KDTree for neighborhood queries.
    kd_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    has_colors = pc.has_colors()
    colors = np.asarray(pc.colors) if has_colors else None

    # Prepare arrays for the smoothed points and colors.
    new_points = np.zeros_like(points)
    if has_colors:
        new_colors = np.zeros_like(colors)

    # For each point, search its neighbors and compute the average.
    for i in range(points.shape[0]):
        # Get indices of all points within the specified radius.
        [k, idx, _] = kd_tree.search_radius_vector_3d(pc.points[i], radius)
        if k > 0:
            new_points[i] = np.mean(points[idx, :], axis=0)
            if has_colors:
                new_colors[i] = np.mean(colors[idx, :], axis=0)
        else:
            new_points[i] = points[i]
            if has_colors:
                new_colors[i] = colors[i]

    # Create a new point cloud for the smoothed data.
    smoothed_pc = o3d.geometry.PointCloud()
    smoothed_pc.points = o3d.utility.Vector3dVector(new_points)
    if has_colors:
        smoothed_pc.colors = o3d.utility.Vector3dVector(new_colors)
    return smoothed_pc

def main():

    input_pc_pth = '/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_colored.pcd'
    output_pth =   '/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/smoothed_cloud.pcd'
    radius = 0.1  # Adjust this value based on your needs.

    # Load the point cloud.
    pc = o3d.io.read_point_cloud(input_pc_pth)
    if pc.is_empty():
        print("Loaded point cloud is empty!")
        return

    # Smooth the point cloud.
    smoothed_pc = smooth_point_cloud(pc, radius)

    # Save the smoothed point cloud.
    o3d.io.write_point_cloud(output_pth, smoothed_pc)
    print(f"Saved smoothed point cloud to {output_pth}")

if __name__ == "__main__":
    main()

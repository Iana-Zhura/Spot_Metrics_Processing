import rosbag
import cv2
import os
import argparse
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import json

def load_image_messages(bag_file, img_topic):
    """
    Load all image messages from the given image topic in the bag.
    
    Returns:
        A sorted list of tuples (timestamp, image_msg).
    """
    image_msgs = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[img_topic]):
            image_msgs.append((t.to_sec(), msg))
    image_msgs.sort(key=lambda x: x[0])
    return image_msgs

def load_odometry_messages(bag_file, odom_topic="/Odometry"):
    """
    Load all odometry messages from the given odometry topic in the bag.
    
    Returns:
        A sorted list of tuples (timestamp, odom_msg).
    """
    odom_msgs = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[odom_topic]):
            odom_msgs.append((t.to_sec(), msg))
    odom_msgs.sort(key=lambda x: x[0])
    return odom_msgs

def find_closest_image(pc_time, image_msgs, max_diff):
    """
    Find the image with the smallest absolute time difference to the point cloud timestamp.
    
    Args:
        pc_time (float): Point cloud timestamp in seconds.
        image_msgs (list): List of tuples (timestamp, image_msg).
        max_diff (float): Maximum allowed time difference in seconds.
    
    Returns:
        (img_time, img_msg) if best match is within max_diff, else None.
    """
    best_img = None
    best_diff = float('inf')
    best_img_time = None
    for img_time, img_msg in image_msgs:
        diff = abs(pc_time - img_time)
        if diff < best_diff:
            best_diff = diff
            best_img = img_msg
            best_img_time = img_time
        # Since image_msgs are sorted, if we've passed pc_time and the diff exceeds max_diff, break early.
        if img_time > pc_time and diff > max_diff:
            break

    if best_diff <= max_diff:
        return best_img_time, best_img
    else:
        return None

def find_closest_odometry(pc_time, odom_msgs):
    """
    Find the odometry message closest in time to the given point cloud timestamp.
    
    Args:
        pc_time (float): Point cloud timestamp.
        odom_msgs (list): List of tuples (timestamp, odom_msg).
    
    Returns:
        The odometry message closest to pc_time.
    """
    if not odom_msgs:
        return None
    return min(odom_msgs, key=lambda x: abs(x[0] - pc_time))[1]

def compute_transformation_matrix(odom_msg):
    """
    Compute the 4x4 transformation matrix from an odometry message.
    
    Args:
        odom_msg: The odometry message.
    
    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    pose = odom_msg.pose.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    orientation = orientation / np.linalg.norm(orientation)  # Normalize quaternion
    rotation_matrix = tf_trans.quaternion_matrix(orientation)[:3, :3]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    return transform_matrix

def process_bag(bag_file, output_dir, selected_indices, 
                pc_topic="/ouster/points",
                img_topic="/camera_front_right_rect/image_rect",
                odom_topic="/Odometry",
                max_time_diff=0.5):
    """
    Process only the selected point cloud indices, find the closest image and odometry (if within threshold),
    and save the point cloud, image, and transformation matrix in the same subfolder.
    
    Args:
        bag_file (str): Path to the ROS bag file.
        output_dir (str): Directory to save the output pairs.
        selected_indices (set): Set of point cloud indices to process.
        pc_topic (str): ROS topic for point clouds.
        img_topic (str): ROS topic for images.
        odom_topic (str): ROS topic for odometry.
        max_time_diff (float): Maximum allowed time difference in seconds for matching images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bridge = CvBridge()
    # Load all image messages once
    image_msgs = load_image_messages(bag_file, img_topic)
    print(f"Loaded {len(image_msgs)} image messages from topic {img_topic}.")

    # Load all odometry messages once
    odom_msgs = load_odometry_messages(bag_file, odom_topic)
    print(f"Loaded {len(odom_msgs)} odometry messages from topic {odom_topic}.")

    with rosbag.Bag(bag_file, 'r') as bag:
        for pc_idx, (topic, pc_msg, t) in enumerate(bag.read_messages(topics=[pc_topic])):
            if pc_idx not in selected_indices:
                continue

            pc_time = t.to_sec()
            result = find_closest_image(pc_time, image_msgs, max_time_diff)
            if result is None:
                print(f"Point cloud {pc_idx}: no matching image found within {max_time_diff} sec (pc time: {pc_time}).")
                continue

            img_time, img_msg = result
            time_diff = abs(pc_time - img_time)
            print(f"Point cloud {pc_idx}: best image match at {img_time} (diff {time_diff:.3f} sec)")

            # Find the closest odometry message for the current point cloud timestamp
            odom_msg = find_closest_odometry(pc_time, odom_msgs)
            if odom_msg is None:
                print(f"Point cloud {pc_idx}: no matching odometry found.")
                continue
            transformation_matrix = compute_transformation_matrix(odom_msg)
            print(f"Point cloud {pc_idx}: computed transformation matrix:\n{transformation_matrix}")

            # Create a subfolder named after the point cloud index (e.g., "pair_0100")
            pair_folder = os.path.join(output_dir, f"pair_{pc_idx:04d}")
            if not os.path.exists(pair_folder):
                os.makedirs(pair_folder)

            # Process and save the point cloud:
            cloud = np.array(list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)))
            if cloud.shape[0] == 0:
                print(f"Point cloud {pc_idx} is empty, skipping.")
                continue

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(cloud)
            pc_filename = os.path.join(pair_folder, f"{pc_time}.pcd")
            o3d.io.write_point_cloud(pc_filename, o3d_cloud)
            print(f"Saved point cloud to {pc_filename}")

            # Save the transformation matrix as a JSON file
            transform_filename = os.path.join(pair_folder, "transformation.json")
            with open(transform_filename, "w") as f:
                json.dump(transformation_matrix.tolist(), f, indent=4)
            print(f"Saved transformation matrix to {transform_filename}")

            # Process and save the image:
            try:
                cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"Error converting image for point cloud {pc_idx}: {e}")
                continue

            img_filename = os.path.join(pair_folder, f"{pc_time}.png")
            cv2.imwrite(img_filename, cv_image)
            print(f"Saved image to {img_filename}")

if __name__ == "__main__":
    # If needed, uncomment the argparse section below.
    if False:
        parser = argparse.ArgumentParser(
            description="Extract images, point clouds, and corresponding transformation matrices from a ROS bag.")
        parser.add_argument("--bag", required=True, help="Path to the ROS bag file.")
        parser.add_argument("--output", required=True, help="Output directory to save the data.")
        parser.add_argument("--pc_topic", default="/ouster/points", help="ROS topic for point clouds.")
        parser.add_argument("--img_topic", default="/camera_front_right_rect/image_rect",
                            help="ROS topic for images.")
        parser.add_argument("--odom_topic", default="/Odometry", help="ROS topic for odometry.")
        parser.add_argument("--indices", nargs='+', type=int, default=[100, 200, 300, 360, 420],
                            help="List of point cloud indices to extract data for.")
        parser.add_argument("--window", type=float, default=1.0,
                            help="Time window (in seconds) for matching images.")
        args = parser.parse_args()

        process_bag(
            bag_file=args.bag,
            output_dir=args.output,
            pc_topic=args.pc_topic,
            img_topic=args.img_topic,
            odom_topic=args.odom_topic,
            selected_indices=args.indices,
            max_time_diff=args.window
        )
    else:
        # Hard-coded parameters for testing.
        bag_path = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/recordings/march24_greenhouse.bag"
        output_path = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/processings/images"
        image_topic = "/camera_front_right_rect/image_rect"
        pc_topic = "/ouster/points"
        odom_topic = "/Odometry"
        pc_indexes = [0, 20, 40, 60, 80, 100,
                      120, 140, 160, 180, 200,
                      220, 240, 260, 280, 300,
                      325, 345, 360, 380, 400,
                      420, 445, 460, 480]
        
        process_bag(
            bag_file=bag_path, 
            output_dir=output_path, 
            img_topic=image_topic, 
            pc_topic=pc_topic,
            odom_topic=odom_topic,
            selected_indices=pc_indexes
        )

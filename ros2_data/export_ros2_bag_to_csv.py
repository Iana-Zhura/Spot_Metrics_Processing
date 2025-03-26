#!/usr/bin/env python3

import os
import csv
import argparse

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# Import the ROS message types we need
# from spot_msgs.msg import BatteryStateArray  # /status/battery_states
from sensor_msgs.msg import JointState       # /joint_states
from nav_msgs.msg import Odometry            # /odometry

def main(output_dir, bag_pth):

    # Initialize rclpy (for deserialization)
    rclpy.init()

    # Configure the bag reader
    storage_options = StorageOptions(uri=bag_pth, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr',
                                         output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Prepare CSV files and their writers
    # 1) battery_states.csv: one row per BatteryState in the array
    battery_csv_path = os.path.join(output_dir, "battery_states.csv")
    battery_file = open(battery_csv_path, 'w', newline='')
    battery_fieldnames = [
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
        "identifier",
        "charge_percentage",
        "estimated_runtime_sec",
        "estimated_runtime_nanosec",
        "current",
        "voltage",
        # We'll store temperatures as a semicolon-separated string
        "temperatures",
        "status",
    ]
    battery_writer = csv.DictWriter(battery_file, fieldnames=battery_fieldnames)
    battery_writer.writeheader()

    # 2) joint_states.csv: one row per JointState message
    joint_csv_path = os.path.join(output_dir, "joint_states.csv")
    joint_file = open(joint_csv_path, 'w', newline='')
    joint_fieldnames = [
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
        # name, position, velocity, effort each as semicolon-separated
        "name",
        "position",
        "velocity",
        "effort",
    ]
    joint_writer = csv.DictWriter(joint_file, fieldnames=joint_fieldnames)
    joint_writer.writeheader()

    # 3) odometry.csv: one row per Odometry message
    odom_csv_path = os.path.join(output_dir, "odometry.csv")
    odom_file = open(odom_csv_path, 'w', newline='')
    odom_fieldnames = [
        # header
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
        "child_frame_id",

        # pose.pose.position
        "pose_pose_position_x",
        "pose_pose_position_y",
        "pose_pose_position_z",

        # pose.pose.orientation
        "pose_pose_orientation_x",
        "pose_pose_orientation_y",
        "pose_pose_orientation_z",
        "pose_pose_orientation_w",

        # pose.covariance (flattened as semicolon-separated)
        "pose_covariance",

        # twist.twist.linear
        "twist_twist_linear_x",
        "twist_twist_linear_y",
        "twist_twist_linear_z",

        # twist.twist.angular
        "twist_twist_angular_x",
        "twist_twist_angular_y",
        "twist_twist_angular_z",

        # twist.covariance (flattened)
        "twist_covariance",
    ]
    odom_writer = csv.DictWriter(odom_file, fieldnames=odom_fieldnames)
    odom_writer.writeheader()

    # We only care about these three topics
    # Map them to expected types and the associated CSV writer
    topics_info = {
       # "/status/battery_states": {
       #     "type": "spot_msgs/msg/BatteryStateArray",
       #     "class": BatteryStateArray,
       # },
        "/joint_states": {
            "type": "sensor_msgs/msg/JointState",
            "class": JointState,
        },
        "/odometry": {
            "type": "nav_msgs/msg/Odometry",
            "class": Odometry,
        },
    }

    # Collect the bag’s available topics -> type
    available_topics = {
        t.name: t.type for t in reader.get_all_topics_and_types()
    }

    # Read all messages
    while reader.has_next():
        topic, raw_data, _timestamp = reader.read_next()
        if topic not in topics_info:
            continue  # skip other topics

        expected_type = topics_info[topic]["type"]
        actual_type   = available_topics.get(topic, None)
        if actual_type != expected_type:
            print(f"[WARN] For topic '{topic}', bag has type '{actual_type}', expected '{expected_type}'. Skipping.")
            continue

        msg_class = topics_info[topic]["class"]
        msg = deserialize_message(raw_data, msg_class)

        # Handle each topic separately
        if topic == "/status/battery_states":
            # Each BatteryState in 'battery_states' is its own row
            for b in msg.battery_states:
                row = {
                    "header_stamp_sec": b.header.stamp.sec,
                    "header_stamp_nanosec": b.header.stamp.nanosec,
                    "header_frame_id": b.header.frame_id,
                    "identifier": b.identifier,
                    "charge_percentage": b.charge_percentage,
                    "estimated_runtime_sec": b.estimated_runtime.sec,
                    "estimated_runtime_nanosec": b.estimated_runtime.nanosec,
                    "current": b.current,
                    "voltage": b.voltage,
                    "temperatures": ";".join(str(t) for t in b.temperatures),
                    "status": b.status,
                }
                battery_writer.writerow(row)

        elif topic == "/joint_states":
            # One row per message
            row = {
                "header_stamp_sec": msg.header.stamp.sec,
                "header_stamp_nanosec": msg.header.stamp.nanosec,
                "header_frame_id": msg.header.frame_id,
                "name": ";".join(msg.name),
                "position": ";".join(str(p) for p in msg.position),
                "velocity": ";".join(str(v) for v in msg.velocity),
                "effort": ";".join(str(e) for e in msg.effort),
            }
            joint_writer.writerow(row)

        elif topic == "/odometry":
            # One row per message
            row = {
                "header_stamp_sec": msg.header.stamp.sec,
                "header_stamp_nanosec": msg.header.stamp.nanosec,
                "header_frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,

                "pose_pose_position_x": msg.pose.pose.position.x,
                "pose_pose_position_y": msg.pose.pose.position.y,
                "pose_pose_position_z": msg.pose.pose.position.z,

                "pose_pose_orientation_x": msg.pose.pose.orientation.x,
                "pose_pose_orientation_y": msg.pose.pose.orientation.y,
                "pose_pose_orientation_z": msg.pose.pose.orientation.z,
                "pose_pose_orientation_w": msg.pose.pose.orientation.w,

                "pose_covariance": ";".join(str(c) for c in msg.pose.covariance),

                "twist_twist_linear_x": msg.twist.twist.linear.x,
                "twist_twist_linear_y": msg.twist.twist.linear.y,
                "twist_twist_linear_z": msg.twist.twist.linear.z,

                "twist_twist_angular_x": msg.twist.twist.angular.x,
                "twist_twist_angular_y": msg.twist.twist.angular.y,
                "twist_twist_angular_z": msg.twist.twist.angular.z,

                "twist_covariance": ";".join(str(c) for c in msg.twist.covariance),
            }
            odom_writer.writerow(row)

    # Close files
    battery_file.close()
    joint_file.close()
    odom_file.close()

    print(f"Done! CSVs saved to: {output_dir}")
    rclpy.shutdown()

if __name__ == '__main__':

    bag_name = "greenhouse_march"
    if bag_name == "greenhouse_march":
        save_folder = "/media/martin/Elements/ros-recordings/recordings/march_11/metrics_csvs"
        bag_path = "/media/martin/Elements/ros-recordings/recordings/march_11/ros2_bag/march11_greenhouse_0.db3"
    elif bag_name == "greenhouse_feb":
        save_folder = "/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/greenhouse_feb_csvs"
        bag_path = "/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/greenhouse_feb_ros2bag/greenhouse_feb_0.db3"

        

    main(save_folder, bag_pth=bag_path)

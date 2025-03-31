import os
import numpy as np
import rosbag
import tqdm
import pandas as pd

def process_rosbag(input_bag, odometry_output_file):
    """
    Process a ROS bag file to extract odometry data and save it to a CSV file.
    """

    print(f"the input bag is {input_bag}")
    imu_data = []

    with rosbag.Bag(input_bag, 'r') as bag:
        # Open the CSV file to append odometry data
        with open(odometry_output_file, 'a') as f_out:
            for topic, msg, t in tqdm.tqdm(bag.read_messages(topics=['/Odometry', '/ouster/imu'])):
                if topic == '/Odometry':
                    # Extract timestamp
                    timestamp = msg.header.stamp.to_sec()

                    # Extract position (x, y, z)
                    position_x = msg.pose.pose.position.x
                    position_y = msg.pose.pose.position.y
                    position_z = msg.pose.pose.position.z

                    # Extract orientation (x, y, z, w)
                    orientation_x = msg.pose.pose.orientation.x
                    orientation_y = msg.pose.pose.orientation.y
                    orientation_z = msg.pose.pose.orientation.z
                    orientation_w = msg.pose.pose.orientation.w

                    # Extract linear velocity (x, y, z)
                    linear_vel_x = msg.twist.twist.linear.x
                    linear_vel_y = msg.twist.twist.linear.y
                    linear_vel_z = msg.twist.twist.linear.z

                    # Extract angular velocity (x, y, z)
                    angular_vel_x = msg.twist.twist.angular.x
                    angular_vel_y = msg.twist.twist.angular.y
                    angular_vel_z = msg.twist.twist.angular.z

                    # Create a data row to be written to the CSV
                    data_row = [timestamp, position_x, position_y, position_z,
                                orientation_x, orientation_y, orientation_z, orientation_w,
                                linear_vel_x, linear_vel_y, linear_vel_z,
                                angular_vel_x, angular_vel_y, angular_vel_z]

                    # Write the row to the CSV file
                    np.savetxt(f_out, [data_row], delimiter=',')
        

if __name__ == "__main__":

    experiment = "greenhouse_march_final"

    if experiment == "imtek":
        odometry_csv_file = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/imtek_feb_odo.csv"
        bag_path = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/cl_imtek.bag"
    elif experiment == "imtek_asphalt":
        odometry_csv_file = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_asphalt_imtek/imtek_asphalt_odo.csv"
        bag_path = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_asphalt_imtek/cl_imtek_asphalt.bag"
    elif experiment == "greenhouse_feb":
        odometry_csv_file = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/odo_greenhouse_feb.csv"
        bag_path = "/root/shared_folder/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/cl_greenhouse.bag"
    elif experiment == "greenhouse_march":
        odometry_csv_file = "/root/shared_folder/ros-recordings/recordings/march_11/greenhouse_march_odo.csv"
        bag_path = "/root/shared_folder/ros-recordings/recordings/march_11/march11_greenhouse.bag"
    elif experiment == "greenhouse_march_final":    
        odometry_csv_file = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/processings/odometry_greenhouse.csv"
        bag_path = "/root/shared_folder/ros-recordings/recordings_final/greenhouse/recordings/march24_greenhouse.bag"
    else:
        raise ValueError("Required arguments not provided!")
    process_rosbag(bag_path, odometry_csv_file)

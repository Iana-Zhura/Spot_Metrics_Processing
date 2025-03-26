import os
import numpy as np
import rosbag
import tqdm
import pandas as pd

def process_rosbag(input_bag, odometry_output_file, imu_csv_file):
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
        
                elif topic == "/ouster/imu":
                    imu_data.append([
                        msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9,  # Timestamp in seconds
                        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
                    ])

    imu_cols = [
        "timestamp", 
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"
    ]

    df_imu = pd.DataFrame(imu_data, columns=imu_cols)
    df_imu.to_csv(imu_csv_file, index=False,) #header=False)


if __name__ == "__main__":

    imu_csv_file = "greenhouse_imu_odo.csv"
    odometry_csv_file = "greenhouse_odo.csv"

    bag_path = "/root/shared_folder/recordings/greenhouse_final/test_lc.bag"

    process_rosbag(bag_path, odometry_csv_file, imu_csv_file)
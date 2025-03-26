import os
import numpy as np
import rosbag
import tqdm

def process_rosbag(input_bag, odometry_output_file):
    """
    Process a ROS bag file to extract odometry data and save it to a CSV file.
    """

    print(f"the input bag is {input_bag}")

    with rosbag.Bag(input_bag, 'r') as bag:
        # Open the CSV file to append odometry data
        with open(odometry_output_file, 'a') as f_out:
            for topic, msg, t in tqdm.tqdm(bag.read_messages(topics=['/Odometry'])):
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
                # else:
                #     # Skip this rosbag file
                #     print("Skipping topic: {}".format(topic))

                    
def process_directory(directory, odometry_output_file):
    """
    Recursively go through all folders in the given directory and process each bag file for odometry data.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bag"):
                bag_path = os.path.join(root, file)
                print("Processing bag file: {}".format(bag_path))  # Using .format() for Python 2 compatibility
                process_rosbag(bag_path, odometry_output_file)

if __name__ == "__main__":
    # Set the top-level directory where all folders and ROS bag files are stored
    base_directory = ""
    print("Base directory: {}".format(base_directory))  # Using .format() for Python 2 compatibility
    # The single output CSV file where all odometry data will be saved
    odometry_output_file = os.path.join(base_directory, "odometry_iana_new_bag.csv")
    
#    bag_path = "/root/shared_folder/greenhouse_final/greenhouse_final_2025-01-15-12-44-06_point_cloud_odometry.bag"
    bag_path = "/root/shared_folder/greenhouse_final/greenhouse_new_odo.bag"

    process_rosbag(bag_path, odometry_output_file)
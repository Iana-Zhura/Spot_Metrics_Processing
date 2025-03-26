import rosbag
import pandas as pd

# Path to your ROS1 bag file
bag_file = "../greenhouse_final_2025-01-15-12-44-06.bag"
topic_name = "/ouster/imu"  # Update this to your actual topic

# List to store extracted data
data = []

# Open the ROS bag file
with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        data.append([
            msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9,  # Timestamp in seconds
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ])

# Define column names
columns = [
    "timestamp", 
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
csv_file = "imu_data.csv"
df.to_csv(csv_file, index=False)

print(f"IMU data successfully saved to {csv_file}")

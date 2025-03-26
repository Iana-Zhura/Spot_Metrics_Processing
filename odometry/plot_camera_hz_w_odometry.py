import os
import rosbag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

# Set file paths
bag_file = "/root/shared_folder/ros-recordings/recordings/march_17/imtek_v2/march17_imtek_v2.bag"
odom_csv = "odometry_data_v2.csv"
camera_csv = "camera_data_v2.csv"

# Check if CSV files exist; if not, process the rosbag and save the data.
if not (os.path.exists(odom_csv) and os.path.exists(camera_csv)):
    print("CSV files not found. Processing rosbag...")
    odom_data = []
    camera_data = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in tqdm.tqdm(bag.read_messages(topics=['/Odometry', '/camera_front_right_rect/image_rect'])):
            if topic == '/Odometry':
                ts = msg.header.stamp.to_sec()
                odom_data.append({
                    'timestamp': ts,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y
                })
            elif topic == '/camera_front_right_rect/image_rect':
                ts = msg.header.stamp.to_sec()
                camera_data.append({'timestamp': ts})
                
    # Save the data to CSV files
    odom_df = pd.DataFrame(odom_data)
    camera_df = pd.DataFrame(camera_data)
    odom_df.to_csv(odom_csv, index=False)
    camera_df.to_csv(camera_csv, index=False)
else:
    print("Loading data from CSV files...")
    odom_df = pd.read_csv(odom_csv)
    camera_df = pd.read_csv(camera_csv)

# Extract data as numpy arrays
odom_times = odom_df['timestamp'].values
odom_x = odom_df['x'].values
odom_y = odom_df['y'].values

camera_times = np.sort(camera_df['timestamp'].values)

# Compute camera frequency for each odometry timestamp using a sliding window
window_size = 0.5  # seconds on each side (total window = 1.0 sec)
camera_frequency = []

for t in odom_times:
    lower = np.searchsorted(camera_times, t - window_size, side='left')
    upper = np.searchsorted(camera_times, t + window_size, side='right')
    count = upper - lower
    freq = count / (2 * window_size)  # Frequency in Hz
    camera_frequency.append(freq)

camera_frequency = np.array(camera_frequency)

# Determine the color for each point: green if frequency >= 5 Hz, red otherwise.
colors = np.where(camera_frequency >= 5, 'green', 'red')

# Plot the odometry path with skewed heatmap based on camera frequency
plt.figure(figsize=(10, 8))
plt.scatter(odom_x, odom_y, c=colors, s=10)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Odometry Path with Camera "Frequency > 5 hz" Mapping')

# Save the plot to a file
output_plot_file = "odometry_camera_frequency_skewed_v2.png"
plt.savefig(output_plot_file, bbox_inches='tight')
plt.close()

print(f"Plot saved to {output_plot_file}")

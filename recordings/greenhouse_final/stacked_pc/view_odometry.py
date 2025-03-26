import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === Configurations ===
ODOMETRY_CSV = "odometry_iana_new_bag.csv"  # Path to odometry CSV
ENABLE_TIMESTAMP_DEBUG = True  # Check synchronization with LiDAR timestamps
ENABLE_MOVEMENT_DEBUG = True   # Print movement per step
ENABLE_QUATERNION_DEBUG = True # Print quaternion angles

# === Load Odometry Data ===
def load_odometry_data(odometry_csv_path):
    odometry_df = pd.read_csv(odometry_csv_path, header=None)
    formatted_data = {}
    
    for idx, row in odometry_df.iterrows():
        timestamp = int(float(row[0]) * 1e9)  # Convert seconds to nanoseconds
        position = np.array([float(row[1]), float(row[2]), float(row[3])])
        orientation = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])  # Quaternion
        formatted_data[timestamp] = {"position": position, "orientation": orientation}
    
    return formatted_data

# Load odometry data
odometry_data = load_odometry_data(ODOMETRY_CSV)
timestamps = sorted(odometry_data.keys())
positions = np.array([odometry_data[t]["position"] for t in timestamps])

# === 1️⃣ Plot Odometry Trajectory ===
plt.figure(figsize=(10, 5))

# Plot trajectory
plt.plot(positions[:, 0], positions[:, 1], label="Odometry Trajectory", marker="o", linestyle="--", color="blue", markersize=4)

# Highlight first and last points
plt.scatter(positions[0, 0], positions[0, 1], color="green", label="First Point", edgecolors="black", s=100, zorder=3)
plt.scatter(positions[-1, 0], positions[-1, 1], color="red", label="Last Point", edgecolors="black", s=100, zorder=3)

# Labels and legend
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Odometry Trajectory Debugging")
plt.legend()
plt.grid()
plt.show()

# === 2️⃣ Debug Timestamp Differences ===
if ENABLE_TIMESTAMP_DEBUG:
    print("\n==== Debugging Timestamps ====")
    for i in range(len(timestamps) - 1):
        time_diff = (timestamps[i+1] - timestamps[i]) * 1e-9  # Convert ns to seconds
        print(f"Timestamp {i} → {i+1}: Δt = {time_diff:.6f} sec")

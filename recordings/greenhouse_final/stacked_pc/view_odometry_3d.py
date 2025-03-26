import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# === Configurations ===
ODOMETRY_CSV = "../odometry.csv"  # Path to odometry CSV
ENABLE_TIMESTAMP_DEBUG = True  # Check synchronization with LiDAR timestamps
ENABLE_MOVEMENT_DEBUG = True   # Print movement per step
ENABLE_QUATERNION_DEBUG = True # Print quaternion angles

# === Load Odometry Data ===
def load_odometry_data(odometry_csv_path):
    odometry_df = pd.read_csv(odometry_csv_path)
    
    # Extract relevant columns
    timestamps = odometry_df["field.header.stamp"].values.astype(np.int64)  # Already in nanoseconds
    positions = odometry_df[["field.pose.pose.position.x", 
                             "field.pose.pose.position.y", 
                             "field.pose.pose.position.z"]].values.astype(np.float64)
    orientations = odometry_df[["field.pose.pose.orientation.x", 
                                "field.pose.pose.orientation.y", 
                                "field.pose.pose.orientation.z", 
                                "field.pose.pose.orientation.w"]].values.astype(np.float64)

    # Convert to dictionary format for easier access
    formatted_data = {timestamps[i]: {"position": positions[i], "orientation": orientations[i]} for i in range(len(timestamps))}
    
    return formatted_data

# Load odometry data
odometry_data = load_odometry_data(ODOMETRY_CSV)
timestamps = sorted(odometry_data.keys())
positions = np.array([odometry_data[t]["position"] for t in timestamps])

# === 1️⃣ Plot Odometry Trajectory in 3D ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Odometry Trajectory", marker="o", linestyle="--", color="blue", markersize=4)

# Highlight first and last points
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", label="First Point", edgecolors="black", s=100, zorder=3)
ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="red", label="Last Point", edgecolors="black", s=100, zorder=3)

# Labels and legend
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("Odometry 3D Trajectory Debugging")
ax.legend()
ax.grid()

plt.show()

# === 2️⃣ Debug Timestamp Differences ===
if ENABLE_TIMESTAMP_DEBUG:
    print("\n==== Debugging Timestamps ====")
    for i in range(len(timestamps) - 1):
        time_diff = (timestamps[i+1] - timestamps[i]) * 1e-9  # Convert ns to seconds
        print(f"Timestamp {i} → {i+1}: Δt = {time_diff:.6f} sec")

# === 3️⃣ Debug Movement Per Step ===
if ENABLE_MOVEMENT_DEBUG:
    print("\n==== Debugging Odometry Movement ====")
    for i in range(len(timestamps) - 1):
        pos1 = odometry_data[timestamps[i]]["position"]
        pos2 = odometry_data[timestamps[i+1]]["position"]
        movement = np.linalg.norm(pos2 - pos1)
        print(f"Step {i} → {i+1}: Movement = {movement:.3f} meters")

# === 4️⃣ Debug Quaternion Angles ===
if ENABLE_QUATERNION_DEBUG:
    print("\n==== Debugging Quaternion Conversions ====")
    for t in timestamps:
        quat = odometry_data[t]["orientation"]
        
        # Test wxyz → xyzw if necessary
        quat_xyzw = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert if needed
        
        angles_standard = R.from_quat(quat).as_euler('xyz', degrees=True)
        angles_converted = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)

        print(f"Time: {t}, Euler Angles (xyz) Standard: {angles_standard}")
        print(f"Time: {t}, Euler Angles (xyz) Converted: {angles_converted}")

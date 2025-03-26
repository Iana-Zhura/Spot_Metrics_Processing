import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Configurations ===
ODOMETRY_CSV = "greenhouse_imu_odo.csv"  # Path to odometry CSV
ENABLE_TIMESTAMP_DEBUG = True  # Check synchronization with LiDAR timestamps
ENABLE_MOVEMENT_DEBUG = True   # Print estimated movement per step

# === Load IMU Odometry Data ===
def load_imu_odometry_data(odometry_csv_path):
    """Load IMU odometry data from CSV."""
    odometry_df = pd.read_csv(odometry_csv_path)

    # Convert timestamps from seconds to nanoseconds
    timestamps = (odometry_df.iloc[:, 0] * 1e9).astype(int).values

    # Extract linear acceleration (assumes the first 3 columns are angular velocity)
    linear_acceleration = odometry_df.iloc[:, 4:7].values  # [ax, ay, az]

    return timestamps, linear_acceleration

# Load odometry data
timestamps, linear_acceleration = load_imu_odometry_data(ODOMETRY_CSV)

print(f"Loaded {len(timestamps)} IMU odometry data points.")

# Normalize timestamps by subtracting the first timestamp
start_time = timestamps[0] * 1e-9  # Convert ns to seconds
normalized_timestamps = (timestamps * 1e-9) - start_time  # Normalize time

# Convert timestamps to time differences (Δt)
time_deltas = np.diff(timestamps) * 1e-9  # Convert ns to seconds
time_deltas = np.insert(time_deltas, 0, 0)  # First step has no Δt

# === Integrate to Estimate Velocity and Position ===
velocity = np.zeros((len(timestamps), 2))  # [vx, vy]
position = np.zeros((len(timestamps), 2))  # [x, y]

for i in range(1, len(timestamps)):
    # Estimate velocity using integration (v = v_prev + a * dt)
    velocity[i] = velocity[i-1] + linear_acceleration[i-1, :2] * time_deltas[i]

    # Estimate position using integration (x = x_prev + v * dt)
    position[i] = position[i-1] + velocity[i] * time_deltas[i]

# === 1️⃣ Plot Estimated 2D Trajectory with Normalized Time Sections ===
num_sections = 6
section_size = len(position) // num_sections
colors = plt.cm.viridis(np.linspace(0, 1, num_sections))

plt.figure(figsize=(10, 5))

# Plot each section in a different color with normalized timestamp ranges in legend
for i in range(num_sections):
    start_idx = i * section_size
    end_idx = (i + 1) * section_size if i < num_sections - 1 else len(position)
    norm_start_time = normalized_timestamps[start_idx]
    norm_end_time = normalized_timestamps[end_idx - 1]

    plt.plot(
        position[start_idx:end_idx, 0],
        position[start_idx:end_idx, 1],
        marker="o",
        linestyle="--",
        color=colors[i],
        markersize=4,
        label=f"{norm_start_time:.2f}s → {norm_end_time:.2f}s"
    )

# Highlight first and last points
plt.scatter(position[0, 0], position[0, 1], color="green", label="Start", edgecolors="black", s=100, zorder=3)
plt.scatter(position[-1, 0], position[-1, 1], color="red", label="End", edgecolors="black", s=100, zorder=3)

# Labels and legend
plt.xlabel("Estimated X Position")
plt.ylabel("Estimated Y Position")
plt.title("Estimated 2D Trajectory with Normalized Time")
plt.legend()
plt.grid()
plt.show()

# === 2️⃣ Debug Timestamp Differences ===
if ENABLE_TIMESTAMP_DEBUG:
    print("\n==== Debugging Timestamps ====")
    for i in range(len(timestamps) - 1):
        time_diff = (timestamps[i+1] - timestamps[i]) * 1e-9  # Convert ns to seconds
        displacement = np.linalg.norm(position[i+1] - position[i])

        print(f"Timestamp {i} → {i+1}: Δt = {time_diff:.6f} sec \t  Δpos = {displacement:.4f} meters")

        if time_diff > 0.1:
            print("Large time difference detected!")
            input()

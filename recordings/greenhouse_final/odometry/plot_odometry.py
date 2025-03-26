import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load IMU CSV data
csv_file = "imu_data.csv"
df = pd.read_csv(csv_file)

# Extract necessary columns
time = df["timestamp"].values
accel_x = df["linear_acceleration_x"].values
accel_y = df["linear_acceleration_y"].values

# Compute time differences (Δt)
dt = np.diff(time, prepend=time[0])  # Prepend first value to keep array size consistent

# Initialize velocity and position arrays
velocity_x = np.zeros_like(accel_x)
velocity_y = np.zeros_like(accel_y)
position_x = np.zeros_like(accel_x)
position_y = np.zeros_like(accel_y)

# Integrate acceleration to get velocity, then integrate velocity to get position
for i in range(1, len(time)):
    velocity_x[i] = velocity_x[i-1] + accel_x[i] * dt[i]
    velocity_y[i] = velocity_y[i-1] + accel_y[i] * dt[i]
    position_x[i] = position_x[i-1] + velocity_x[i] * dt[i]
    position_y[i] = position_y[i-1] + velocity_y[i] * dt[i]

# Plot the estimated 2D path
plt.figure(figsize=(8, 6))
plt.plot(position_x, position_y, label="Estimated Path", color='b')
plt.scatter(position_x[0], position_y[0], color='g', marker='o', label="Start")
plt.scatter(position_x[-1], position_y[-1], color='r', marker='x', label="End")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Estimated 2D Path from IMU Data")
plt.legend()
plt.grid()
plt.show()

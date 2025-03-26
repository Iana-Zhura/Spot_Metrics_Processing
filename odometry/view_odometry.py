import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_odometry(odometry_path):
    # Load odometry data
    odometry_data = load_odometry_data(odometry_path)
    timestamps = sorted(odometry_data.keys())
    positions = np.array([odometry_data[t]["position"] for t in timestamps])

    # Split the data into 6 sections
    num_sections = 6
    section_size = len(positions) // num_sections
    colors = plt.cm.viridis(np.linspace(0, 1, num_sections))

    plt.figure(figsize=(10, 5))

    # Plot each section in a different color with timestamp ranges in legend
    init_time = timestamps[0]  # Convert ns to seconds

    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else len(positions)
        start_time = ( timestamps[start_idx] - init_time) * 1e-9  # Convert ns to seconds
        end_time = (timestamps[end_idx - 1]-init_time) * 1e-9  # Convert ns to seconds

        plt.plot(
            positions[start_idx:end_idx, 0],
            positions[start_idx:end_idx, 1],
            marker="o",
            linestyle="--",
            color=colors[i],
            markersize=4,
            label=f"{start_time:.2f}s → {end_time:.2f}s"
        )

    # Highlight first and last points
    plt.scatter(positions[0, 0], positions[0, 1], color="green", label="First Point", edgecolors="black", s=100, zorder=3)
    plt.scatter(positions[-1, 0], positions[-1, 1], color="red", label="Last Point", edgecolors="black", s=100, zorder=3)

    # Labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Odometry Trajectory Split into Sections")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":

    prefix = "imtek"

    ODOMETRY_CSV = f"/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/{prefix}/{prefix}_feb_odo.csv"  # Replace with actual file path
    plot_odometry(ODOMETRY_CSV)
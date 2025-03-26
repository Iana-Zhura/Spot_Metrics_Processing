import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_merged_timestamps(filename):
    """
    Load merged timestamps from a JSON file.
    
    Parameters:
      filename (str): Path to the JSON file.
      
    Returns:
      dict: Dictionary with merged time intervals per waypoint.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_odometry_csv(filename):
    """
    Load odometry data from a CSV file.
    
    Assumes that column 0 contains timestamps and columns 1-3 contain x,y,z coordinates.
    
    Parameters:
      filename (str): Path to the odometry CSV file.
      
    Returns:
      pandas.DataFrame: DataFrame containing odometry data.
    """
    df = pd.read_csv(filename, header=None)
    # Ensure the timestamp column is numeric
    df[0] = pd.to_numeric(df[0])
    return df

def load_metric_csv(filename):
    """
    Load metric data from a CSV file.
    
    Parameters:
      filename (str): Path to the metric CSV file.
      
    Returns:
      pandas.DataFrame: DataFrame containing metric data.
    """
    df = pd.read_csv(filename, header=None)
    return df

def load_and_process_joint_states(filename):
    """
    Load and process joint_states.csv to compute energy-related columns.
    
    This function:
      - Extracts joint names from the "name" column.
      - Converts semicolon-delimited 'velocity' and 'effort' strings into NumPy arrays.
      - Computes a unified timestamp from 'header_stamp_sec' and 'header_stamp_nanosec'.
      - Calculates the time difference (dt) between consecutive rows.
      - Computes per-joint power (velocity * effort) and then energy (power * dt).
    
    Parameters:
      filename (str): Path to the joint_states CSV file.
    
    Returns:
      tuple: (processed joint_states DataFrame, list of joint names)
    """
    joint_states = pd.read_csv(filename)
    joint_names = joint_states["name"].iloc[0].split(";")
    
    # Convert semicolon-delimited strings into NumPy arrays of floats
    joint_states["velocity"] = joint_states["velocity"].apply(lambda x: np.array(list(map(float, x.split(";")))))
    joint_states["effort"] = joint_states["effort"].apply(lambda x: np.array(list(map(float, x.split(";")))))
    
    # Compute timestamp in seconds
    joint_states["timestamp"] = joint_states["header_stamp_sec"] + joint_states["header_stamp_nanosec"] * 1e-9
    
    # Compute dt (time difference between rows)
    joint_states["dt"] = joint_states["timestamp"].diff().fillna(0)
    
    # Compute power per joint: torque * velocity
    joint_states["power_per_joint"] = joint_states.apply(lambda row: row["velocity"] * row["effort"], axis=1)
    
    # Compute energy per joint: power * dt
    joint_states["energy_per_joint"] = joint_states.apply(lambda row: row["power_per_joint"] * row["dt"], axis=1)
    
    return joint_states, joint_names

def calculate_joint_energy(joint_states, start_time, end_time):
    """
    Calculate the aggregated joint energy between start_time and end_time.
    
    The energy for each joint is computed as the absolute value of (torque * velocity * dt).
    This function sums these absolute energies for all joints over all rows that fall within
    the specified time window.
    
    Parameters:
      joint_states (DataFrame): Processed joint_states DataFrame with energy columns.
      start_time (float): Start time (in seconds) for the energy calculation.
      end_time (float): End time (in seconds) for the energy calculation.
    
    Returns:
      float: Aggregated joint energy (in Joules) over the given interval.
    """
    # Filter the DataFrame for the specified time window
    window_df = joint_states[(joint_states["timestamp"] >= start_time) & (joint_states["timestamp"] <= end_time)]
    
    # Sum the absolute energy over all joints for each row, then sum over the window
    total_energy = window_df["energy_per_joint"].apply(lambda energy: np.sum(np.abs(energy))).sum()
    return total_energy

def calculate_distance_for_intervals(odometry_df, intervals):
    """
    Calculate the total distance traveled from the odometry data over the given intervals.
    
    For each interval, this function filters the odometry DataFrame to find rows whose timestamp
    is within the interval. It then computes the sum of Euclidean distances between consecutive points.
    
    Parameters:
      odometry_df (DataFrame): DataFrame containing odometry data (timestamp in column 0, x,y,z in columns 1-3).
      intervals (list of dict): List of intervals, each with "start" and "end" keys.
      
    Returns:
      float: Total distance traveled.
    """
    total_distance = 0.0
    for interval in intervals:
        start = interval["start"]
        end = interval["end"]
        # Filter odometry points within the current interval
        subset = odometry_df[(odometry_df[0] >= start) & (odometry_df[0] <= end)]
        # Ensure we have at least 2 points to compute a distance
        if len(subset) < 2:
            continue
        # Extract positions as a NumPy array (columns 1-3)
        positions = subset.iloc[:, 1:4].to_numpy()
        # Compute Euclidean distances between consecutive points and sum them
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance += np.sum(distances)
    return total_distance

def print_timestamps(data):
    print("Merged Timestamps Data:")
    for waypoint, info in data.items():
        print(f"Waypoint {waypoint} (Anchor ID: {info.get('waypoint_id', 'N/A')}): Overall Time Spent: {info['overall_time']}")
        for interval in info["intervals"]:
            print(f"  Interval: Start = {interval['start']}, End = {interval['end']}")    

def timebins_metrics(merged_filename, odometry_filename, joint_states_filename, output_pth, logging=False):
    # Load data from files
    merged_data = load_merged_timestamps(merged_filename)
    odometry_data = load_odometry_csv(odometry_filename)
    joint_states, _ = load_and_process_joint_states(joint_states_filename)
    
    # Print summary of merged timestamps data
    if logging:
        print_timestamps(merged_data)
        print("\nOdometry Data Preview:")
        print(odometry_data.head())
        print("\nJoint States Data Preview:")
        print(joint_states.head())
    
    # Iterate through the merged timestamps (each waypoint) to compute joint energy, distance,
    # average effort, and total time spent in each bin.
    print("\nAggregated Joint Energy, Distance, Average Effort and Time Spent per Waypoint:")
    waypoint_results = {}
    for waypoint, info in merged_data.items():
        total_energy = 0.0
        total_effort_weighted_sum = 0.0
        total_effort_time = 0.0
        total_interval_time = 0.0
        
        # Process each interval (bin) in the current waypoint
        for interval in info["intervals"]:
            start = interval["start"]
            end = interval["end"]
            duration = end - start
            total_interval_time += duration
            
            # Calculate energy for this interval
            energy = calculate_joint_energy(joint_states, start, end)
            total_energy += energy
            
            # Calculate average effort within this interval from joint_states data
            subset = joint_states[(joint_states["timestamp"] >= start) & (joint_states["timestamp"] <= end)]
            if not subset.empty:
                subset_avg_effort = subset["effort"].apply(lambda x: np.mean(x))
                dt_sum = subset["dt"].sum()
                if dt_sum > 0:
                    avg_effort_interval = (subset_avg_effort * subset["dt"]).sum() / dt_sum
                else:
                    avg_effort_interval = subset_avg_effort.mean()
                total_effort_weighted_sum += avg_effort_interval * duration
                total_effort_time += duration
            
        overall_avg_effort = total_effort_weighted_sum / total_effort_time if total_effort_time > 0 else 0
        distance = calculate_distance_for_intervals(odometry_data, info["intervals"])
        
        # Keep the waypoint_id from the merged JSON data
        waypoint_id = info.get("waypoint_id", None)
        
        waypoint_results[waypoint] = {
            "waypoint_id": waypoint_id,
            "energy": total_energy,
            "distance": distance,
            "average_effort": overall_avg_effort,
            "time_spent": total_interval_time
        }
        
        if logging:
            print(f"Waypoint {waypoint} (Waypoint ID: {waypoint_id}): Total Energy = {total_energy} Joules, "
                  f"Distance = {distance}, Average Effort = {overall_avg_effort}, "
                  f"Time Spent = {total_interval_time} seconds")
    
    # Export the results to a JSON file for use by other code
    with open(output_pth, "w") as f:
        json.dump(waypoint_results, f, indent=2)
    print(f"\nExported waypoint energy, distance, average effort and time spent results to '{output_pth}'")

if __name__ == "__main__":
  # Filenames - update these paths as necessary.
  prefix = "greenhouse_final"

  merged_filename = "../fit_sdk_odometry/merged_time_intervals.json"
  odometry_filename = "../odometry/greenhouse_odo.csv"
  joint_states_filename = "../recordings/greenhouse_final/greenhouse_final_csvs/joint_states.csv"
  save_filename = f"{prefix}_metrics.json"
  
  timebins_metrics(merged_filename=merged_filename,
                    odometry_filename=odometry_filename,
                    joint_states_filename=joint_states_filename,
                    output_pth="waypoint_energy_distance_results.json",
                    logging=True)

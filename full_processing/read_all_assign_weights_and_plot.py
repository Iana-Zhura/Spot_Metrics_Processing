import subprocess
import argparse

from odometry.view_odometry import plot_odometry
from fit_sdk_odometry.martin_fit_graph_odo import fit_graph2odo, save_transformation_params
from metrics_assigned_to_time.assign_metrics import timebins_metrics
from plot_with_weights.plot_weighted_anchors import plot_anchors_weights_2d
import os

def load_transformation_params(trans_json_path):
    """
    Load transformation parameters from a JSON file."
    """
    import json
    # Check if the file exists
    
    with open(trans_json_path, 'r') as f:
        loaded_json = json.load(f)
    
    # Check if the loaded JSON has the expected keys
    required_keys = ["rotation_z", "rotation_y", "translation"]
    for key in required_keys:
        if key not in loaded_json:
            raise KeyError(f"Missing key '{key}' in the loaded JSON.")
    
    return loaded_json

def run_pipeline(odometry_csv, sdk_graph_path, point_cloud_path, metrics_csv, path_fit="fit_output.json", path_metric="metric_output.json", transformation={}):
    
    # Step 1: Run odometry/view_odometry.py with the odometry CSV path.
    print("Step 1: Running odometry/view_odometry.py ...")
    plot_odometry(odometry_csv)

    # Step 2: Run fit_sdk_odometry/martin_fit_graph_odo.py with SDK graph, odometry, point cloud, and prefix.
    # this requires to run an export in the terminal
    if True:
        os.environ['XDG_SESSION_TYPE'] = 'x11'

    print("\nStep 2: Running fit_sdk_odometry/martin_fit_graph_odo.py ...")
    # We assume that the script will output a file with this name.
    if len(transformation.keys()):
        rotation_z = transformation["rotation_z"]
        rotation_y = transformation["rotation_y"]
        translation = transformation["translation"]
        fit_graph2odo(sdk_graph_path, odometry_csv, point_cloud_path, export_pth=path_fit, rotation_z=rotation_z, rotation_y=rotation_y, translation=translation)
    else:
        raise ValueError("Transformation parameters are required for fitting the graph to odometry.")
        # fit_graph2odo(sdk_graph_path, odometry_csv, point_cloud_path, export_pth=path_fit)
    
    # Step 4: Run metrics_assigned_to_time/assign_metrics.py with metrics CSV, output from step 4, odometry CSV, and prefix.
    print("\nStep 3: Running metrics_assigned_to_time/assign_metrics.py ...")
    # Assume that this script outputs a JSON file.

    timebins_metrics(merged_filename=path_fit, odometry_filename=odometry_csv, joint_states_filename=metrics_csv, output_pth=path_metric)
    
    # Step 4: Run plot_with_weights/plot_weighted_anchors.py with odometry CSV, SDK graph path, and the metrics JSON output.
    print("\nStep 4: Running plot_with_weights/plot_weighted_anchors.py ...")
    if len(transformation.keys()):
        plot_anchors_weights_2d(map_path=sdk_graph_path, odometry_file=odometry_csv, weights_file=path_metric,
                                rotation_z=transformation["rotation_z"], rotation_y=transformation["rotation_y"],
                                translation=transformation["translation"])
    else:
        plot_anchors_weights_2d(map_path=sdk_graph_path, odometry_file=odometry_csv, weights_file=path_metric)
    
    
    print("Pipeline execution complete.")


if __name__ == "__main__":

    data = "greenhouse_very_final"

    # TODO: for all other files, you need to update the fit and metric paths (depending on where you were saving them)
    if False:
        # to save the params from here, you can just run by hand the following function, regardless, you also need to load from here
        # so it's fair game to just be updating from here too
        save_transformation_params(otation_z, rotation_y, translation, output_filename)

        # also, you can just run the `fit_graph2odo` from here, no need to run it separately
        fit_graph2odo(sdk_graph_path, odometry_csv, point_cloud_path, export_pth=path_fit,
                      rotation_z=rotation_z, rotation_y=rotation_y, translation=translation)

    transformation = {}
    if data=="greenhouse_march":
        default_odo = "/media/martin/Elements/ros-recordings/recordings/march_11/greenhouse_march_odo.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings/march_11/downloaded_graph"
        default_pc = "/media/martin/Elements/ros-recordings/recordings/march_11/merged_cloud_selected.pcd"
        default_metrics = "/media/martin/Elements/ros-recordings/recordings/march_11/metrics_csvs/joint_states.csv"
        default_prefix = "greenhouse_march"

        transformation["rotation_z"] = 205
        transformation["rotation_y"] = -1.5
        transformation["translation"] = [2.25, -0.8, -0.45]

    elif data=="greenhouse_final":
        default_odo =  "/media/martin/Elements/ros-recordings/odometry/greenhouse_odo.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/downloaded_graph/"
        default_pc = '/media/martin/Elements/ros-recordings/pointclouds/merged_cloud_selected.pcd'
        default_metrics = "/media/martin/Elements/ros-recordings/recordings/greenhouse_final/greenhouse_final_csvs/joint_states.csv"
        default_prefix = "greenhouse_final"
    elif data=="imtek":
        default_odo = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/imtek_feb_odo.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings/feb_27/campus_imtek/downloaded_graph"
        default_pc = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_imtek/merged_cloud_selected.pcd"
        default_metrics = "/media/martin/Elements/ros-recordings/recordings/feb_27/campus_imtek/campus_imtek_csvs/joint_states.csv"
        default_prefix = "imtek"

        transformation["rotation_z"] = 162
        transformation["rotation_y"] = 1
        transformation["translation"] = [2.7, -0.9, -0.4]

    elif data == "greenhouse_feb":
        default_odo = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/odo_greenhouse_feb.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/downloaded_graph"
        default_pc = "/media/martin/Elements/ros-recordings/recordings/feb_27/RERECORDED/rerecorded_greeenhouse_feb/merged_cloud_selected.pcd"
        default_metrics = "/media/martin/Elements/ros-recordings/recordings/feb_27/greenhouse_feb/greenhouse_feb_csvs/joint_states.csv"
        default_prefix = "greenhouse_feb"

        transformation["rotation_z"] = 171
        transformation["rotation_y"] = -5
        transformation["translation"] = [2.4, -1.5, -0.5]
    
    elif data == "greenhouse_very_final":
        default_odo = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/odometry_greenhouse.csv"
        default_sdk = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/recordings/downloaded_graph"
        default_pc = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/merged_cloud_selected.pcd"
        default_metrics = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/metrics_csvs/joint_states.csv"
        default_prefix = "greenhouse_final_very_final"

        fit_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/greenhouse_final_fit_output.json"
        metric_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/metrics_to_time/greenhouse_time_metrics.json"
        trans_path = "/media/martin/Elements/ros-recordings/recordings_final/greenhouse/processings/fit_odometry/transformation_params.json"

        # this is assuming I ran the martin_fit_graph_odo.py before and saved the transormation params 
        # (that is kind of required anyway, so I say it's a fair assumption)
        transformation = load_transformation_params(trans_path)
        
    if False:
        parser = argparse.ArgumentParser(description="Run full odometry and metrics processing pipeline.")
        parser.add_argument("--odometry", default=default_odo, help="Path to the odometry CSV file")
        parser.add_argument("--sdk_graph", default=default_sdk, help="Path to the SDK graph file (or folder with the graph file)")
        parser.add_argument("--point_cloud", default=default_pc, help="Path to the saved point cloud file")
        parser.add_argument("--metrics_csv", default=default_metrics, help="Path to the metrics CSV file (e.g., joint efforts)")
        parser.add_argument("--prefix", default=default_prefix, help="Prefix for all output files")
        
        args = parser.parse_args()
        run_pipeline(args.odometry, args.sdk_graph, args.point_cloud, args.metrics_csv, transformation, path_fit=fit_path, path_metric=metric_path)

    else:
        run_pipeline(odometry_csv=default_odo, sdk_graph_path=default_sdk, point_cloud_path=default_pc,
                     metrics_csv=default_metrics, transformation=transformation,
                     path_fit=fit_path, path_metric=metric_path)
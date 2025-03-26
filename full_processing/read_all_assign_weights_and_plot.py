import subprocess
import argparse

from odometry.view_odometry import plot_odometry
from fit_sdk_odometry.martin_fit_graph_odo import fit_graph2odo
from metrics_assigned_to_time.assign_metrics import timebins_metrics
from plot_with_weights.plot_weighted_anchors import plot_anchors_weights_2d
import os

BASE_PTH = os.getcwd()

def run_pipeline(odometry_csv, sdk_graph_path, point_cloud_path, metrics_csv, prefix="", transformation={}):
    
    # Step 3: Run odometry/view_odometry.py with the odometry CSV path.
    print("Step 3: Running odometry/view_odometry.py ...")
    plot_odometry(odometry_csv)

    # Step 4: Run fit_sdk_odometry/martin_fit_graph_odo.py with SDK graph, odometry, point cloud, and prefix.
    # this requires to run an export in the terminal
    if True:
        os.environ['XDG_SESSION_TYPE'] = 'x11'

    print("Step 4: Running fit_sdk_odometry/martin_fit_graph_odo.py ...")
    # We assume that the script will output a file with this name.
    fit_output = f"{BASE_PTH}/fit_sdk_odometry/{prefix}_fit_output.json"
    if len(transformation.keys()):
        rotation_z = transformation["rotation_z"]
        rotation_y = transformation["rotation_y"]
        translation = transformation["translation"]
        fit_graph2odo(sdk_graph_path, odometry_csv, point_cloud_path, export_pth=fit_output, rotation_z=rotation_z, rotation_y=rotation_y, translation=translation)
    else:
        fit_graph2odo(sdk_graph_path, odometry_csv, point_cloud_path, export_pth=fit_output)
    
    # Step 5: Run metrics_assigned_to_time/assign_metrics.py with metrics CSV, output from step 4, odometry CSV, and prefix.
    print("Step 5: Running metrics_assigned_to_time/assign_metrics.py ...")
    # Assume that this script outputs a JSON file.
    metrics_output = f"{BASE_PTH}/metrics_assigned_to_time/{prefix}_metrics.json"
    timebins_metrics(merged_filename=fit_output, odometry_filename=odometry_csv, joint_states_filename=metrics_csv, output_pth=metrics_output)
    
    # Step 6: Run plot_with_weights/plot_weighted_anchors.py with odometry CSV, SDK graph path, and the metrics JSON output.
    print("Step 6: Running plot_with_weights/plot_weighted_anchors.py ...")
    if len(transformation.keys()):
        plot_anchors_weights_2d(map_path=sdk_graph_path, odometry_file=odometry_csv, weights_file=metrics_output,
                                rotation_z=transformation["rotation_z"], rotation_y=transformation["rotation_y"],
                                translation=transformation["translation"])
    else:
        plot_anchors_weights_2d(map_path=sdk_graph_path, odometry_file=odometry_csv, weights_file=metrics_output)
    
    
    print("Pipeline execution complete.")

if __name__ == "__main__":

    data = "greenhouse_march"

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
        

    parser = argparse.ArgumentParser(description="Run full odometry and metrics processing pipeline.")
    parser.add_argument("--odometry", default=default_odo, help="Path to the odometry CSV file")
    parser.add_argument("--sdk_graph", default=default_sdk, help="Path to the SDK graph file (or folder with the graph file)")
    parser.add_argument("--point_cloud", default=default_pc, help="Path to the saved point cloud file")
    parser.add_argument("--metrics_csv", default=default_metrics, help="Path to the metrics CSV file (e.g., joint efforts)")
    parser.add_argument("--prefix", default=default_prefix, help="Prefix for all output files")
    
    args = parser.parse_args()
    
    run_pipeline(args.odometry, args.sdk_graph, args.point_cloud, args.metrics_csv, args.prefix, transformation)

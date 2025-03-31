# How to actually use any of the code

### Workflow
1. record ros1 with odometry from fastlio and the pointclouds
2. record ros2 bag with the metrics
3. stack point clouds from ros1 bag = run `pointclouds/stack_point_odo.py`
  - accumulate the point clouds into a `.pcd` file
4. read odometry from ros1 bag = `odometry/odometry_bag2csv.py`
  -  reads the odometry and save it as prefix_odo.csv
5. read the metrics from ros2 bag = `ros2_data/export_ros2bag_to_csv.py`
6. add the outputs (their path) to `fir_sdk_odometry/martin_fit_graph_odo.py` -> checkout already existing scenes
  - now the transformation needs to be updated to fit the odometry to the waypoints
7. with the odometry fitting the waypoints layout, run `full_processing/read_all_assign_weights_and_plot.py`
8. now is time to calculate the weights
  - run `edge_weights/assign_edge_weights.py` (needs data paths for the scene edits in the code)
9. last you need to load the weights back to the graph and save it
  - run `edge_weights/load_edge_weights.py` (needs updates with the data paths)
10. finally, we can view in 3d the stacked point clouds with the the updated graph weight by running
    `edge_weights/updated_graphs/plot_graph.py`

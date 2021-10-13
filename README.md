# Camera Projection Node for Drone Mapping

Combine Lidar scans, Mask-R-CNN predictions, and GPS data to create 3D landmarks for mapping and localization!

![Visualization](https://user-images.githubusercontent.com/35245591/137205129-ef81c576-67c0-449e-9c62-129139faaf1e.png)

## Table of Contents
- [Details](#Details)
- [Pipeline](#Pipeline)
- [Usage](#Usage)
- [Acknowledgements](#Acknowledgements)

## Details
This node listens to 4 ROS topics:
- `/dji_sdk/gps_position` for GPS latitude and longitude
- `/dji_sdk/imu` for IMU (orientation only)
- `/velodyne_aggregated` or `/velodyne_points` for Lidar scans, depending on whether you use aggregated (described below) or individual scans. Using aggregated scans is slower but acheives better accuracy.
- `/cnn_predictions` for Mask-R-CNN predictions, ask described below

and publishes to 2 ROS topics (or more for debugging):
- `/projected_predictions` for predictions of 3D landmarks (of type predictions.msg found in this repo)
- `/vis_full` for live visualizations of the projection and estimated trunk depths

In summary, this node (1) transforms the Lidar point cloud to the image frame, (2) matches image pixels to Lidar points, (3) estimates the depth of the detected trunks in the image using surrounding Lidar points, (4) converts these 3D locations to GPS coordinates, and (5) publishes these for use in localization and mapping.

## Pipeline
This node is part of a larger drone mapping pipeline with separate ROS nodes, each receiving and publishing relevant data.

Code locations for the other nodes are listed below:
- [__MASK-R-CNN__](https://github.com/aaronzberger/CMU_Mask-R-CNN_ROS) - Predict locations of trunks (and other objects) in images
- [__AGGREGATION__](https://github.com/aaronzberger/CMU_Aggregation_Node) - Aggregate multiple Lidar scans together using IMU and GPS data

## Usage
This node has been tested in combination with the other two nodes listed above performs well offline.

For information on usage, please contact Aaron (aaronzberger@gmail.com).

## Acknowledgements
- Francisco Yandun for assistance throughout the creation of the pipeline

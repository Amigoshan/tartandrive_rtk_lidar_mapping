## rtk_lidar_mapping.py
The `accumulate_pointclouds_using_rtk` function reads 
- pointcloud with per-point timestamp
- RTK odometry
- RGB images
, and returns a accumulated pointcloud or a colorized pointcloud. 

Two sample input folders are located at: 
- AirLab_Storage_Offroad/wenshanw/06_warehouse_calib_6
- AirLab_Storage_Offroad/wenshanw/00_warehouse_calib_1

Know issues: 
- This is a sample script, not quite a general implementation. There are hard-coded directories and parameters all over the place. 


## voxel_rate.py
This script implements a metric to evaluate the quality of pointcloud registration. Good registration results in thinner ground surfaces and object surfaces, which results in less voxel numbers if the points are insert to a voxel grid. 

voxel_rate = #voxel / #points
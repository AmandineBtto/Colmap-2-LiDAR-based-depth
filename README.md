# Improved depth from mono camera images fusing Colmap and LiDAR data


## Description

The goal of this project is to create improved depth images using Colmap SfM and a LiDAR scan of the environement. 

Using camera images, Colmap computes camera intriniscs, extrinsics and poses (sparse reconstruction) but also depth (depth reconstruction). In some case the depth quality is not good enough. LiDAR scan of the environement can therefore be a way to get a better depth image for each camera image. 

Before using this code, Colmap sparse reconstruction and optionnaly Colmap undistorter must be runned. 
You need a "images.txt" file (containing extrinsics) and "cameras.txt" file (containing intrinsics). 
If you also want to use the undistortion of Colmap, you also need the "cameras.txt" given after running the Colmap undistorter. 

The 4x4 transformation matrix between Colmap point cloud from the sparse reconstruction and the LiDAR have to be given in argument. For example, you can find it using CloudCompare point registration between the two point clouds.

Warning:
- This code works only for images using the same camera.
- If the Colmap reconstruction is of bad quality, the camera poses won't be accurate resulting in bad LiDAR depth images.
- Using a LiDAR point cloud with a lot of points will improve the depth image quality but can result in a longer time of execution and it will take more memory space.


## Steps of the project

1. Read "images.txt" and compute extrinsics and poses in Colmap Space.
2. Transform the camera poses in the LiDAR space using the given transformation matrix.
3. Compute the extrinsics in LiDAR space from the poses obtained in 2.
4. Project the LiDAR point cloud in each camera space.
5. Filter the projected LiDAR point cloud to keep only points in the FOV (saved in .ply format). The FOV is calculated using Colmap "cameras.txt". If the Colmap undistorter is used, the "cameras.txt" should be the output of the Colmap undistorter.
6. For each filtered point cloud, compute the distance of each point to the camera.
7. Project the filtered point cloud in the 2D space.
8. Save the depth image in distance in a npy file.


## Run 

This code has been developped and tested on Ubuntu VERSION="20.04.5 LTS". 
The .YAML file can be used to duplicate the conda environement. 
The main packages necessary are pandas, numpy, matplotlib, plyfile and opencv-python.

### To get parameters description:
```
python main.py -h
```

### To run without the Colmap undistorter:
```
python main.py \
--colmap_camera_extrinsics_file path/to/images.txt \
--colmap_camera_intrinsics_file path/to/cameras.txt \
--saving_dir path/to/saving_dir/ \
--path_lidar_pc path/to/lidar_pc.ply \
--transformation_mat LINE 1  \
--transformation_mat LINE 2  \
--transformation_mat LINE 3  \
--transformation_mat 0 0 0 1  \
--save True \
--visualization True \
--orig_image_id 10 \
--data_augment False \
--size WIDTH HEIGHT \
--undistorter None \
--colmap_dense_undistorter_intrinsics_file None
```

### To run with Colmap undistorter:
```
python main.py \
--colmap_camera_extrinsics_file path/to/images.txt \
--colmap_camera_intrinsics_file path/to/cameras.txt \
--saving_dir path/to/saving_dir/ \
--path_lidar_pc path/to/lidar_pc.ply \
--transformation_mat LINE 1  \
--transformation_mat LINE 2  \
--transformation_mat LINE 3  \
--transformation_mat 0 0 0 1  \
--save True \
--visualization True \
--orig_image_id 10 \
--data_augment False \
--size WIDTH HEIGHT \
--undistorter colmap \
--colmap_dense_undistorter_intrinsics_file path/to/colmap_undistorter_result/cameras.txt
```

### Parameters description:

- ```--colmap_camera_intrinsics_file```
Path to your "cameras.txt" file.

- ```--saving_dir```
Path to the directory you want to save outputs of the code.

- ```--path_lidar_pc ```
Path to the LiDAR point cloud in .ply.

- ```--transformation_mat```
4x4 Transformation matrix between Colmap space and LiDAR space. 
Each line must be given as float, with space between numbers.

- ```--save```
Can be True or False.
If True will save in the saving_dir, two .pkl files containing extrinsics and poses in Colmap and LiDAR space. 

- ```--visualization```
Can be True or False.
If True will save two .ply files containing the cameras centers in Colmap and LiDAR space. It can be use to check the tran sformation matrix.

- ```--orig_image_id```
Can be an int or None. 
If None, will run on all images. If an int in specified, will compute the depth for the corresponding image.

- ```--data_augment ```
Can be True or False. 
If True, will do a naive data augmentation on the depth. It means it will duplicate the distance to the directly adjacent pixels. Can be used in the case of very sparce LiDAR point clouds.

- ```--size ```
Can be WIDTH HEIGHT or None. 
If None, will return automatic size depending of the 2D projection. 
If a size is given, it can crop the original 2D projection or add 0.

- ```--undistorter ```
Can be "colmap" or None.
If Colmap undistorter is chosen, you need to specify the path to the new intrinsics with ```--colmap_dense_undistorter_intrinsics_file```. 
The advice here is also to fix the size of the depth to match Colmap undistorted RGB images.
If None is chosen, the 2D projection of the points is done using the PINHOLE matrix. It means that the distortion is not used. In the case your original RGB images are distorted, you can undistort them to PINHOLE using openCV.

- ```--colmap_dense_undistorter_intrinsics_file```
Path to your "cameras.txt" file given after Colmap undistorter. Can also be None if ```--undistorter``` is None.

### Output

```

├── saving_dir
│   ├── projected_pc
│   │   ├── projected_lidar_pc_in_camera_space_fov_0.ply
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   ├── projected_lidar_pc_in_camera_space_fov_N.ply
│   ├── depth
│   │   ├── depth_camera_image_0.npy
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   ├── depth_camera_image_N.npy
│   ├── colmap_camera_ext_and_pose.pkl
│   ├── lidar_camera_ext_and_pose.pkl
│   ├── viz_camera_centers_colmap.ply
│   ├── viz_camera_centers_lidar.ply

```

## Terms and Conditions

The code can be used freely if you agree with all the following terms.
- The code is used only for non-commercial purposes, such as teaching and research. You do not use the code or any of its modified versions for any purposes of commercial advantage or private financial gain.
- In case you use this code within your research papers, you refer to this repository. 
- I reserve all rights that are not explicitly granted to you. The code is provided as is, and you take full responsibility for any risk of using it. There may be inaccuracies although I tried, and will try my best to rectify any inaccuracy once found.
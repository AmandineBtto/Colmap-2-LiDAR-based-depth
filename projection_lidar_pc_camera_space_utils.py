import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import os
import glob
import PIL
from PIL import Image

from scipy.spatial.transform import Rotation as R

from plyfile import PlyData, PlyElement

import cv2 as cv
import math

import sys
global EPSILON 
EPSILON = sys.float_info.epsilon

from viz_utils import *
from depth_utils import *


# Projection of lidar point cloud in camera space + filter FOV -------
def get_intrinsic_and_fov(colmap_camera_intrinsic_file):
    """
    Parameters:
        colmap_camera_intrinsic_file: str
                                      Path to colamp "cameras.txt" output file.
    
    Output:
        intrinsics_mat: Numpy matrix
                        Camera intrinsics matrix (pinhole model) w/o distorsion.
        dist_coef: float
                   Distorsion coefficient of the camera (Radial Simple Fisheye model)
        fov_x: float 
               Angular field of view on the x axis, in degree
        fov_y: float 
               Angular field of view on the y axis, in degree
    """
    
    line_list = []
    with open(colmap_camera_intrinsic_file) as f:
        line_list = f.readlines()
    
    cam_model = line_list[3].strip().split(" ")[1]
    
    if cam_model == "PINHOLE":
        CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy = line_list[3].strip().split(" ") 
        # pinhole structure 
        intrinsics = [[float(fx), 0, float(cx)], 
                       [0,float(fy), float(cy)], 
                       [0, 0, 1]]
        intrinsics_mat = np.mat(intrinsics)
        dist_coef = None
        # retrieve fov
        w = float(WIDTH)
        h = float(HEIGHT)
        fov_x = np.rad2deg(2 * np.arctan2(w, 2 * float(fx)))
        fov_y = np.rad2deg(2 * np.arctan2(h, 2 * float(fy)))
    
    else:
        CAMERA_ID, MODEL, WIDTH, HEIGHT, f, cx, cy, k = line_list[3].strip().split(" ") 
        # pinhole structure 
        intrinsics = [[float(f), 0, float(cx)], 
                       [0,float(f), float(cy)], 
                       [0, 0, 1]]
        intrinsics_mat = np.mat(intrinsics)
        dist_coef = float(k)
        # retrieve fov
        w = float(WIDTH)
        h = float(HEIGHT)
        focal_length = float(f)
        fov_x = np.rad2deg(2 * np.arctan2(w, 2 * focal_length))
        fov_y = np.rad2deg(2 * np.arctan2(h, 2 * focal_length))

        #print("Field of View (degrees):")
        #print(f"  {fov_x = :.1f}\N{DEGREE SIGN}")
        #print(f"  {fov_y = :.1f}\N{DEGREE SIGN}")
    
    return intrinsics_mat, dist_coef, fov_x, fov_y


def filter_pc_in_fov(fov_x, fov_y, pc_in_cam_space):
    """
    Parameters: 
        fov_x: float 
               Angular field of view on the x axis, in degree
        fov_y: float 
               Angular field of view on the x axis, in degree
        pc_in_cam_space: list
                         LiDAR point cloud projected in camera space
    Output:
        pc_in_fov_for_ply: Numpy array
                           FOV point cloud projected in camera space ready for .ply saving
    """
    
    pc_in_fov = []
    for elt in pc_in_cam_space:
        x = elt[0]
        y = elt[1]
        z = elt[2]
        # keep front of the cam in fov angle
        x_lim = math.tan((fov_x/2)*math.pi/180)
        y_lim = math.tan((fov_y/2)*math.pi/180)
        if z>0 and x <= x_lim*z and x >= - x_lim*z:
            if y <= y_lim*z and y >= - y_lim*z:
                pc_in_fov.append(tuple(elt))
                   
    pc_in_fov_for_ply = np.array(pc_in_fov,
                        dtype=[('x', 'f4'), ('y', 'f4'),
                                 ('z', 'f4')])
    return pc_in_fov_for_ply

def proj_lidar_pc_in_cam_space(lidar_camera_info_df, path_lidar_pc, colmap_camera_intrinsic_file, saving_dir, orig_image_id = None):
    """
    Parameters:
        lidar_camera_info_df: Pandas df
                              Dataframe containing the correspondance between colmap image id and the actual image, projected extrinsics and poses in lidar space.
        path_lidar_pc: str
                       Path to the lidar point cloud in ply
        colmap_camera_intrinsic_file: str
                                      Path to colamp "cameras.txt" output file.
        saving_dir: str
                    path to save projected point cloud in camera space. Create a 'projected_pc' directory.
        orig_image_id: int
                       Chosen image, the user wants the depth, if None will compute the depth for all images.
    
    """
    # read lidar pc
    plydata = PlyData.read(path_lidar_pc)
    d_lidar = {'x': plydata.elements[0].data['x'], 'y': plydata.elements[0].data['y'], 'z': plydata.elements[0].data['z']}
    df_lidar = pd.DataFrame(data=d_lidar)
    
    # get fov values
    intrinsics_mat, dist_coef, fov_x, fov_y = get_intrinsic_and_fov(colmap_camera_intrinsic_file)
    
    # create a folder to store projected point cloud
    file_path = os.path.join(saving_dir, 'projected_pc')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # project pc in all camera spaces
    if orig_image_id == None:
        for i, row in lidar_camera_info_df.iterrows():
            # get extrinsic
            cam_ext = np.matrix(row['lidar_extrinsic'])
            pc_in_cam_space = []
            for idx in range(0, len(df_lidar)):
                point_mat = np.mat([df_lidar['x'] [idx], df_lidar['y'][idx], df_lidar['z'][idx], 1]).transpose()
                n_point = cam_ext*point_mat
                pc_in_cam_space.append(n_point[:3].transpose().tolist()[0])
            pc_in_fov = filter_pc_in_fov(fov_x, fov_y, pc_in_cam_space)
            # retrieve original image id and save
            orig_img_name = row['name']
            real_idx = orig_img_name.split('_')[-1].split('.')[-2]
            file_name = 'projected_lidar_pc_in_camera_space_fov_' + real_idx + '.ply'
            el_col = PlyElement.describe(pc_in_fov, 'projected_lidar_pc_in_camera_space')
            PlyData([el_col]).write(os.path.join(file_path, file_name))
              
    # project pc in only one camera space
    else:
        pc_in_cam_space = []
        # get original camera image type of name
        orig_name = lidar_camera_info_df['name'][0]
        nb = len(str(orig_image_id))
        first_part_name = orig_name.split('.')[-2][:-nb]
        extension_name = orig_name.split('.')[-1]
        # get colmap id 
        colmap_id = lidar_camera_info_df[lidar_camera_info_df['name'] == first_part_name + str(orig_image_id) + '.' +  extension_name]['image_id'].item()
        # get extrinsic
        cam_ext = np.matrix(lidar_camera_info_df['lidar_extrinsic'][colmap_id]) 
        for idx in range(0, len(df_lidar)):
            point_mat = np.mat([df_lidar['x'] [idx], df_lidar['y'][idx], df_lidar['z'][idx], 1]).transpose()
            n_point = cam_ext*point_mat
            pc_in_cam_space.append(n_point[:3].transpose().tolist()[0])
        pc_in_fov = filter_pc_in_fov(fov_x, fov_y, pc_in_cam_space)
        # save
        file_name = 'projected_lidar_pc_in_camera_space_fov_' + str(orig_image_id) + '.ply'
        el_col = PlyElement.describe(pc_in_fov, 'projected_lidar_pc_in_camera_space')
        PlyData([el_col]).write(os.path.join(file_path, file_name))

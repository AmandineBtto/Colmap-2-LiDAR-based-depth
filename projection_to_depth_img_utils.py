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
from projection_lidar_pc_camera_space_utils import *
from projection_colmap_lidar_utils import *


# Projection from 3D pc in camera space to 2D ------
def proj_2D_space(proj_pc_3D, intrinsics_mat): 
    """
    Parameters:
        proj_pc_3D: .ply file. 
                    LiDAR 3D point cloud projected in a camera space. 
            
        intrinsics_mat: Numpy matrix
                        Camera intrinsics matrix as estimated by Colmap.
    Output:
        pt_2D: Pandas dataframe
               Contains the projection of the 3D points in the image space (2D).
    """
    
    x_in_img = []
    y_in_img = []
    for i, row in proj_pc_3D.iterrows():
        pt_3D = [row['x'], row['y'], row['z']]
        proj_pt = intrinsics_mat*np.transpose(np.mat(pt_3D)) 
        x = proj_pt[0,0].tolist()
        y = proj_pt[1,0].tolist()
        z = proj_pt[2,0].tolist()
        x_in_img.append(round(x/z))
        y_in_img.append(round(y/z))
    x_y_in_img = {'x': x_in_img, 'y': y_in_img}
    pt_2D = pd.DataFrame(data=x_y_in_img)
    
    return pt_2D


def dist_to_cam_mat(proj_pc_3D):
    """
    Parameters:
        proj_pc_3D: .ply file. 
                    LiDAR 3D point cloud projected in a camera space. 
    Output:
        dist: list of the euclidian distance between FOV 3D points and the camera.
    """
    
    squared_dist = proj_pc_3D['x']**2 + proj_pc_3D['y']**2 + proj_pc_3D['z']**2
    dist = np.sqrt(squared_dist)

    return dist


def depth_img_in_dist(pt_2D, dist, data_augm, size = None):
    """
    Parameters:
        pt_2D: Pandas dataframe
               Contains the projection of the 3D points in the image space (2D).
        dist: list of the euclidian distance between FOV 3D points and the camera.
        data_augm: Boolean
                   If True, will duplicate each value in adjacent pixels.
                   It is a very "basic" data augmentation.
        size: tuple 
              If None, will create image depending of the 2D projection, can vary between image (rounding consequence)
    Output:
        dist_depth: Numpy array
                    Return depth image in distance.
    """
    
    x_max = pt_2D['x'].max()
    y_max = pt_2D['y'].max()
    
    if size == None: 
        if data_augm == True:
            dist_depth = np.zeros((x_max+2,y_max+2), np.float32)
        else: 
            dist_depth = np.zeros((x_max+1,y_max+1), np.float32)
    else:
        dist_depth = np.zeros(size, np.float32)
        if size[0] <= x_max or size[1] <= y_max:
            print('Warning: the given size for depth image crop original FOV')

    for i, row in pt_2D.iterrows():
        x = row['x']
        y = row['y']
        
        if size != None and (x > size[0]-1 or y > size[1]-1):
            # skip px out of the chosen size (value stays at 0)
            continue
            
        if data_augm == True and size != None:
            if x >= size[0]-1 or y >= size[1]-1:
                # no augmentation on side px to keep the chosen size
                dist_depth[x, y] = dist[i]
            else:
                data_aug_dist(dist_depth, x, y, dist[i])
        elif data_augm == True:
            data_aug_dist(dist_depth, x, y, dist[i])
        else:
            dist_depth[x, y] = dist[i]
            
    return dist_depth


def point_cloud_to_depth_image(saving_dir, colmap_camera_intrinsic_file, orig_image_id = None, data_augm = False, size = None):
    """
    Parameters:
        saving_dir: str
                    path to save depth. Create a 'depth' directory.
        colmap_camera_intrinsic_file: str
                                      Path to colamp "cameras.txt" output file.
        orig_image_id: int
                       Chosen image, the user wants the depth, if None will compute the depth for all images.
        data_augm: bool
                   Depth image with naive data augmentation
        size: tuple
              Size of the depth 
    """
    
    # check that the folder containing the projected LiDAR pc in camera space exists and is not empty
    proj_pc_path = os.path.join(saving_dir, 'projected_pc')
    if not os.path.exists(proj_pc_path) or not os.listdir(proj_pc_path):
        raise Exception('Projected point cloud in camera space not saved, please run proj_lidar_pc_in_cam_space')
    
    # create a folder to save the depth images
    saved_depth_path = os.path.join(saving_dir, 'depth')
    if not os.path.exists(saved_depth_path):
        os.makedirs(saved_depth_path)
    
    # get intrinsic matrix
    intrinsics_mat, _, _, _ = get_intrinsic_and_fov(colmap_camera_intrinsic_file)
    
    if orig_image_id == None:
        
        # list all .ply files in directory
        projected_pc_saved = sorted([f for f in os.listdir(proj_pc_path) if f.endswith('.ply')])
        
        for pc_file_name in projected_pc_saved:
            
            # load ply file
            plydata = PlyData.read(os.path.join(proj_pc_path, pc_file_name))
            d_pc = {'x': plydata.elements[0].data['x'], 'y': plydata.elements[0].data['y'], 'z': plydata.elements[0].data['z']}
            df_pc = pd.DataFrame(data=d_pc)
            
            # compute distance to camera
            dist_mat = dist_to_cam_mat(df_pc)
            
            # project points in 2D space
            pt_2D = proj_2D_space(df_pc, intrinsics_mat)
            
            # create depth img 
            dist_depth = depth_img_in_dist(pt_2D, dist_mat, data_augm, size)
            
            # save depth img 
            depth_name = 'depth_camera_image_' + pc_file_name.split('.')[-2].split('_')[-1] + '.npy'
            np.save(os.path.join(saved_depth_path, depth_name), dist_depth)     
            
    else:
        pc_file_name = 'projected_lidar_pc_in_camera_space_fov_' + str(orig_image_id) + '.ply'
        if not os.path.exists(os.path.join(proj_pc_path, pc_file_name)):
            raise Exception('Projected point cloud for given image id is not saved')
        
        # load ply file
        plydata = PlyData.read(os.path.join(proj_pc_path, pc_file_name))
        d_pc = {'x': plydata.elements[0].data['x'], 'y': plydata.elements[0].data['y'], 'z': plydata.elements[0].data['z']}
        df_pc = pd.DataFrame(data=d_pc)
            
        # compute distance to camera
        dist_mat = dist_to_cam_mat(df_pc)
            
        # project points in 2D space
        pt_2D = proj_2D_space(df_pc, intrinsics_mat)
            
        # create depth img 
        dist_depth = depth_img_in_dist(pt_2D, dist_mat, data_augm, size)
            
        # save depth img 
        depth_name = 'depth_camera_image_' + str(orig_image_id) + '.npy'
        np.save(os.path.join(saved_depth_path, depth_name), dist_depth)      
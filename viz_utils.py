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


def viz_camera_centers(space, camera_info_df, saving_dir):
    """
    Parameters: 
        space: str
               Lidar or Colmap space
        camera_info_df: Pandas df
                        Dataframe containing the correspondance between colmap image id and the actual image, extrinsics and poses, for lidar or colmap.
        saving_dir: str
                    path to save visualization file
    """

    if space == 'lidar':
        list_pose_matrix = camera_info_df['lidar_pose']
    elif space == 'colmap':
        list_pose_matrix = camera_info_df['colmap_pose']
    else:
        raise Error("Chosen space not available. Please specify colmap or lidar")
    
    list_camera_centers = []
    for elt in list_pose_matrix:
        list_camera_centers.append(np.squeeze(np.asarray(elt[:3,3:]).tolist()))

    # viz camera centers
    list_camera_centers_ply = [tuple(cc) for cc in list_camera_centers]
    vertex_col = np.array(list_camera_centers_ply,
                          dtype=[('x', 'f4'), ('y', 'f4'),
                                 ('z', 'f4')])
    el_col = PlyElement.describe(vertex_col, 'camera_centers_' + str(space))
    PlyData([el_col]).write(os.path.join(saving_dir, 'viz_camera_centers_' + str(space) + '.ply'))
    
    

def viz_camera_pose_vector_base(space, camera_info_df, saving_dir, orig_image_id = None):
    """
    Parameters: 
        space: str
               Lidar or Colmap space
        camera_info_df: Pandas df
                        Dataframe containing the correspondance between colmap image id and the actual image, extrinsics and poses, for lidar or colmap.
        saving_dir: str
                    path to save visualization file
        orig_image_id: int
                       Chosen image, the user wants the depth, if None will compute the depth for all images.
                       
    Function not used in the main but can be used to visualize camera pose vector base.
    """
    
    if space == 'lidar':
        list_pose_matrix = camera_info_df['lidar_pose']
    elif space == 'colmap':
        list_pose_matrix = camera_info_df['colmap_pose']
    else:
        raise Error("Chosen space not available. Please specify colmap or lidar")
    
    # when no specific image_id is given
    if image_id == None:
        xaxis = []
        yaxis = []
        zaxis = []
        
        for elt in list_pose_matrix:
            rvec = elt[:3,:3]
            tvec = elt[:3,3]
            xaxis.extend([tvec + rvec[:,0]*i*0.1 for i in range(10)])
            yaxis.extend([tvec + rvec[:,1]*i*0.1 for i in range(10)])
            zaxis.extend([tvec + rvec[:,2]*i*0.1 for i in range(10)])
        
        file_name = 'viz_pose_vector_base_' + str(space) + '.ply'
    
    # when image_id is given
    else:
        rvec = list_pose_matrix[orig_image_id][:3,:3]
        ccvec = list_pose_matrix[orig_image_id][:3,3]

        xaxis = [ccvec + rvec[:,0]*i*0.1 for i in range(10)]
        yaxis = [ccvec + rvec[:,1]*i*0.1 for i in range(10)]
        zaxis = [ccvec + rvec[:,2]*i*0.1 for i in range(10)]
        
        file_name = 'viz_cam_id' + str(orig_image_id) + 'pose_vector_base_' + str(space) + '.ply'

    tab_repere = [tuple(np.transpose(elt).tolist()[0] + [255,0,0]) for elt in xaxis] + [tuple(np.transpose(elt).tolist()[0] + [0,255,0]) for elt in yaxis] + [tuple(np.transpose(elt).tolist()[0] + [0,0,255]) for elt in zaxis]
    vertex = np.array(tab_repere,
                      dtype=[('x', 'f4'), ('y', 'f4'),
                                 ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'),
                               ('blue', 'u1')])

    el = PlyElement.describe(vertex,  str(space) + 'camera_pose_vector_base')
    PlyData([el]).write(os.path.join(saving_dir, file_name))
    
    

def fov_viz(saving_dir, orig_image_id, angle = 80):
    """
    Parameters: 
        saving_dir: str
                    path to save visualization file
        orig_image_id: int
                       Chosen image, the user wants the depth, if None will compute the depth for all images.
        fov_angle: int
                   FOV angle in degree
                   
    Function not used in the main but can be used to visualize FOV angle.
    """
    
    FOV_test = []
    for z_tmp in range(100):
        z = z_tmp/10
        x = math.tan((angle/2)*math.pi/180)*z
        FOV_test.append((x,0,z))
        FOV_test.append((-x,0,z))
    vertex_col = np.array(FOV_test,
                          dtype=[('x', 'f4'), ('y', 'f4'),
                                 ('z', 'f4')])
    el_col = PlyElement.describe(vertex_col, 'lidar_pc_in_cam_space_fov')
    PlyData([el_col]).write(os.path.join(saving_dir, 'lidar_pc_in_cam_id_' + str(orig_image_id) + '_space_fov_viz.ply'))
    

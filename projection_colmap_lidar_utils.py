import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import os
import glob
import PIL
from PIL import Image

from plyfile import PlyData, PlyElement

import cv2 as cv
import math

import sys
global EPSILON 
EPSILON = sys.float_info.epsilon

from viz_utils import *
from depth_utils import *


def quaternion_rotation_matrix(Q):
    """ 
    Parameters:
        Q: List of quaternions
    Output:
        rot_matrix: Numpy array
                    Rotation matrix corresponding to the given quaternions
                    
    found here: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    """

    # Extract the values from Q
    q0 = Q[0] # angle of rotation
    q1 = Q[1] # axis of rotation about which the angle of rotation is performed
    q2 = Q[2] #same
    q3 = Q[3] # same
     
    # First row of the rotation matrix
    r00 = 2 * (q0 ** 2 + q1 ** 2 ) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 **2 + q2 **2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


# Colmap functions -------------

def get_colamp_extrinsics_and_poses(colmap_camera_extrinsics_path, save = True, viz = False, saving_dir = None):
    """
    Parameters:
        colmap_camera_extrinsics_path: str
                                       path to the 'images.txt' file output by colmap sparse reconstruction
        save: bool
              save colmap camera poses and extrinsics in a pkl file
        viz: bool
             create a ply file containing cameras centers to check it match with colmap 
        saving_dir: str
                    path to save pkl and visualization files
    Output:
        colmap_camera_info_df: pandas df
                               Dataframe containing the correspondance between colmap image id and the actual image, extrinsics and poses.
    """
    
    # read .txt file
    line_list = []
    with open(colmap_camera_extrinsics_path) as f:
        line_list = f.readlines()
    
    list_image_id = []
    list_name = []
    list_pose_matrix = []
    list_extrinisic_matrix = []
    
    for i in range(4, len(line_list)):
        if i%2==0:
            # read and store colmap information
            IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = line_list[i].strip().split(" ")
            list_image_id.append(int(IMAGE_ID))
            list_name.append(str(NAME))
    
            qvec = [float(QW), float(QX), float(QY), float(QZ)]
            tvec = [float(TX), float(TY), float(TZ)]
            
            # extrinsic matrix from quaternion and translation                                                                    
            tvec_i = np.mat(tvec)
            rot_i = quaternion_rotation_matrix(qvec)
            ext_matrix_i = np.eye(4)
            ext_matrix_i[:3,:3] = rot_i
            ext_matrix_i[:3,3:] = tvec_i.transpose()
            list_extrinisic_matrix.append(np.matrix(ext_matrix_i))
            
            # camera pose matrix from extrinsic
            pose_rvec_i = np.transpose(rot_i)
            camera_center_i =  - np.transpose(rot_i) * tvec_i.transpose()
            pose_matrix_i = np.eye(4)
            pose_matrix_i[:3,:3] = pose_rvec_i
            pose_matrix_i[:3,3:] = camera_center_i
            list_pose_matrix.append(np.matrix(pose_matrix_i)) 
            
    colmap_camera_info_df = pd.DataFrame()        
    colmap_camera_info_df['image_id'] = list_image_id
    colmap_camera_info_df['name'] = list_name
    colmap_camera_info_df['colmap_extrinsic'] = list_extrinisic_matrix
    colmap_camera_info_df['colmap_pose'] = list_pose_matrix
    
    if save == True:
        assert saving_dir != None
        # save camera extrinsics and pose
        colmap_camera_info_df.to_pickle(os.path.join(saving_dir, 'colmap_camera_ext_and_pose.pkl'))
        
    if viz == True:
        assert saving_dir != None
        viz_camera_centers('colmap', colmap_camera_info_df, saving_dir)
        
    return colmap_camera_info_df


# Colmap to lidar space functions -------------

def colmap_to_lidar_space(colmap_camera_info_df, transformation_mat, save = True, viz = False, saving_dir = None):
    """
    Parameters:
        colmap_camera_info_df: Pandas df
                               Dataframe containing the correspondance between colmap image id and the actual image, extrinsics and poses.
                               Output of 'get_colamp_extrinsics_and_poses'
        transformation_mat: Numpy matrix
                            Transformation matrix from colmap space to lidar space.
        save: bool
              save colmap camera poses and extrinsics in a pkl file
        viz: bool
             create a ply file containing cameras centers to check it match with colmap 
        saving_dir: str
                    path to save pkl and visualization files
             
    Output:
        lidar_camera_info_df: Pandas df
                              Dataframe containing the correspondance between colmap image id and the actual image, projected extrinsics and poses in lidar space.
    
    """
    
    list_pose_matrix = colmap_camera_info_df['colmap_pose']
    
    camera_pose_in_lidar_space = []
    camera_extrinsic_in_lidar_space = []
    
    for pose_matrix in list_pose_matrix:
        pose_inlidspace = transformation_mat * pose_matrix
        camera_pose_in_lidar_space.append(pose_inlidspace)
        
        ext_inlidspace = np.eye(4)
        ext_inlidspace[:3,:3] = np.transpose(pose_inlidspace[:3,:3])
        ext_inlidspace[:3,3:] = - np.transpose(pose_inlidspace[:3,:3]) * pose_inlidspace[:3,3:]
        camera_extrinsic_in_lidar_space.append(np.mat(ext_inlidspace))
    
    lidar_camera_info_df = pd.DataFrame()
    lidar_camera_info_df['image_id'] = colmap_camera_info_df['image_id'] 
    lidar_camera_info_df['name'] = colmap_camera_info_df['name'] 
    lidar_camera_info_df['lidar_extrinsic'] = camera_extrinsic_in_lidar_space
    lidar_camera_info_df['lidar_pose'] = camera_pose_in_lidar_space
    
    if save == True:
        assert saving_dir != None
        lidar_camera_info_df.to_pickle(os.path.join(saving_dir, 'lidar_camera_ext_and_pose.pkl'))
        
    if viz == True:
        assert saving_dir != None
        # extract camera centers
        viz_camera_centers('lidar', lidar_camera_info_df, saving_dir)
    
    return lidar_camera_info_df


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
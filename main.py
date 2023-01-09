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
import argparse
import sys

from projection_colmap_lidar_utils import *
from projection_lidar_pc_camera_space_utils import *
from projection_to_depth_img_utils import *
from depth_utils import *
from viz_utils import *

def main(args):

    Tmat = np.matrix(args.transformation_mat)
    print(Tmat)
    
    if args.undistorter == 'colmap':
        assert args.colmap_dense_undistorter_intrinsics_file != None 
        intrinsics = args.colmap_dense_undistorter_intrinsics_file
    else:
        intrinsics = args.colmap_camera_intrinsics_file
    
    if args.size != None:
        size = (args.size[0], args.size[1])
        print(size)
    else:
        size = args.size
        
    # get extrinsincs and poses matrix
    colmap_camera_info_df = get_colamp_extrinsics_and_poses(args.colmap_camera_extrinsics_file, save = args.save, viz = args.visualization, saving_dir = args.saving_dir)
    
    # get extrinsics and poses matrix in lidar space
    lidar_camera_info_df = colmap_to_lidar_space(colmap_camera_info_df, transformation_mat = Tmat, save = args.save, viz = args.visualization, saving_dir = args.saving_dir)
    
    # project lidar point cloud in each camera space and filter to keep only FOV
    proj_lidar_pc_in_cam_space(lidar_camera_info_df, path_lidar_pc = args.path_lidar_pc, colmap_camera_intrinsic_file = intrinsics, saving_dir = args.saving_dir, orig_image_id = args.orig_image_id)
    
    # project 3D points in 2D and get depth image in distance
    point_cloud_to_depth_image(saving_dir = args.saving_dir, colmap_camera_intrinsic_file = intrinsics, orig_image_id = args.orig_image_id, data_augm = args.data_augment, size = size)

    
    
    
    
if __name__ == '__main__':
    
    # Create the parser
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--colmap_camera_extrinsics_file', type=str, required=True, help = "path to colmap 'images.txt' file")
    parser.add_argument('--colmap_camera_intrinsics_file', type=str, required=True, help = "path to colmap 'cameras.txt' file")
    parser.add_argument('--saving_dir', type=str, required=True, help = "path to the directory where output files will be save")
    parser.add_argument('--path_lidar_pc', type=str, required=True, help = "path to lidar point cloud to use for depth image creation")
    
    # colmap space to lidar space transformation matrix
    parser.add_argument('--transformation_mat', '-tmat', type=float, nargs='+', action='append', required=True, help = "transformation matrix to go colmap sparse point cloud to lidar point cloud in list")
    
    # debugging arg
    parser.add_argument('--save', type=bool, required=True, default=True, help = "saving intermediate files")
    parser.add_argument('--visualization', type=bool, required=True, default=False, help = "saving intermediate ply visualization files")
    
    # image choice
    parser.add_argument('--orig_image_id', type=int, required=False, default=None, help = "image id to get the depth image, if None will compute all depth image")
    
    # depth parameters
    parser.add_argument('--data_augment', type=bool, required=False, default=False, help = "chose data augmentation for depth")
    parser.add_argument('--size', type=int, nargs='+', required=False, default=None, help = "size of the desired depth image, if None return original size")
    
    # if undistorted rgb
    parser.add_argument('--undistorter', type=str, required=True, default=None, help = "rgb image undistorter chosen. 'colmap' use colmap undistorter")
    parser.add_argument('--colmap_dense_undistorter_intrinsics_file', type=str, required=False, help = "path to colmap undistorter 'cameras.txt' file")
    
    args = parser.parse_args()
    
    main(args)
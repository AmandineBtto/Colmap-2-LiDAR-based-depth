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

"""
Distance depth utils
"""
# naive augmentation for distance depth
def data_aug_dist(dist_depth,x,y,v):
    dist_depth[x,y] = v
    dist_depth[x+1,y] = v
    dist_depth[x-1,y] = v
    dist_depth[x,y+1] = v
    dist_depth[x,y-1] = v
    dist_depth[x+1,y+1] = v
    dist_depth[x-1,y+1] = v
    dist_depth[x-1,y-1] = v
    dist_depth[x+1,y-1] = v

# run but results not very good, either improve later or remove
def interpolate_dist_depth(dist_depth):
    mask = np.zeros((dist_depth.shape[0],dist_depth.shape[1]), 'uint8')
    for x in range(dist_depth.shape[0]):
        for y in range(dist_depth.shape[1]):
            mask[x,y] = 1 if dist_depth[x,y]==0 else 0
    dist_depth_test = np.expand_dims(dist_depth, axis =2)
    interpolate_depth = cv.inpaint(dist_depth_test,mask,3,cv.INPAINT_TELEA)
    return interpolate_depth


"""
RGB mapped depth utils
WARNING: not in main
"""

# naive augmentation for rgb depth
def set_chan(rgb_depth,x,y,c,v):
    rgb_depth[x,y,c] = v
    rgb_depth[x+1,y,c] = v
    rgb_depth[x-1,y,c] = v
    rgb_depth[x,y+1,c] = v
    rgb_depth[x,y-1,c] = v
    rgb_depth[x+1,y+1,c] = v
    rgb_depth[x-1,y+1,c] = v
    rgb_depth[x-1,y-1,c] = v
    rgb_depth[x+1,y-1,c] = v

def set_pix(rgb_depth,x,y,r,g,b):
    set_chan(rgb_depth, x,y,0,r)
    set_chan(rgb_depth, x,y,1,g)
    set_chan(rgb_depth, x,y,2,b)
    

def interpolate_rgb_depth(rgb_depth):
    mask = np.zeros((rgb_depth.shape[0],rgb_depth.shape[1]),'uint8')
    for x in range(rgb_depth.shape[0]):
        for y in range(rgb_depth.shape[1]):
            mask[x,y] = 1 if rgb_depth[x,y,0]==0 and rgb_depth[x,y,1]==0 and rgb_depth[x,y,2]==0 else 0
    interpolate_depth = cv.inpaint(rgb_depth,mask,3,cv.INPAINT_TELEA)
    return interpolate_depth

# RGB mapping 
def convert_to_rgb(minval, maxval, val, colors):
    # `colors` is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.

    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the `colors` palette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)

    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.

    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else: # Return a color linearly interpolated in the range between it and 
          # the following one.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def depth_rgb_mapping(depth_dist_name, overall_min_dist, overall_max_dist, saving_dir, data_augm = False):
    # min and max should be global and not relative to one camera or one room configuration (consistency)
    depth_dist = np.load(os.path.join(saving_dir, 'depth', depth_dist_name))
    
    saved_depth_rgb_path = os.path.join(saving_dir, 'depth_rgb')
    if not os.path.exists(saved_depth_rgb_path):
        os.makedirs(saved_depth_rgb_path)
        
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
    
    x_max = depth_dist.shape[0]
    y_max = depth_dist.shape[1]
    
    if data_augm == True:
        rgb_depth = np.zeros((xmax+1,ymax+1,3), 'uint8')
    else:
        rgb_depth = np.zeros((x_max,y_max,3), 'uint8')
        
            
    for x in range(0, x_max):
        for y in range(0, y_max):
            d = depth_dist[x, y]
            r,g,b = convert_to_rgb(overall_min_dist, overall_max_dist, d, colors)
            if data_augm == True:
                set_pix(rgb_depth,x,y,r,g,b)
            else:
                rgb_depth[x,y,0] = r
                rgb_depth[x,y,1] = g
                rgb_depth[x,y,2] = b
        
    # save 
    rgb_depth_name = 'rgb_depth_image_' + depth_dist_name.split('_')[-1].split('.')[-2] + '.npy'
    np.save(os.path.join(saved_depth_rgb_path, rgb_depth_name), rgb_depth)  



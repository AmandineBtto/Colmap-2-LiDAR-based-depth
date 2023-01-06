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


# run but results not very good, either improve later or remove
""""
def interpolate_dist_depth(dist_depth):
    mask = np.zeros((dist_depth.shape[0],dist_depth.shape[1]), 'uint8')
    for x in range(dist_depth.shape[0]):
        for y in range(dist_depth.shape[1]):
            mask[x,y] = 1 if dist_depth[x,y]==0 else 0
    dist_depth_test = np.expand_dims(dist_depth, axis =2)
    interpolate_depth = cv.inpaint(dist_depth_test,mask,3,cv.INPAINT_TELEA)
    return interpolate_depth
"""

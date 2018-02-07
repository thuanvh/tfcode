import cv2
#import dlib
import h5py
import glob
from os import walk
# Read all files
import scipy.io as sio
import numpy as np
import os
import sys
#import thread
import threading
import argparse

from data_sample import TrainSample

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):    
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def read_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    pt2d = mat['pt2d']
    return pose_params,pt2d

def read_sample_list(datapath, bins):
    file_list = []

    for(dirpath, dirnames, filenames) in walk(datapath):
        for f in filenames:
            if f.endswith(".jpg"):
                file_list.append(dirpath + "/" + f)
    print(len(file_list))

    #file_list2 = glob.glob("/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP/**/*.jpg", recursive=True)
    #print(len(file_list2))

    sample_list = []
    sample_number = 6
    file_idx = 0
    for filepath in file_list:  
        #print(file_idx, len(file_list))
        sys.stdout.write("Reading mat: %d   \r" % (file_idx) )
        sys.stdout.flush()
        file_idx+=1
        src = cv2.imread(filepath)
        #print(src.shape)
        if src.shape[0] == 0 or src.shape[1] == 0:
            print("Error in reading image", filepath)
        mat_path = filepath[:-3] + "mat"
        #pt2d = get_pt2d_from_mat(mat_path)
        #pose = get_ypr_from_mat(mat_path) # We get the pose in radians
        pose, pt2d = read_mat(mat_path)
        #print(pose)
        for i in range(0,3):
            pose[i] = pose[i] * 180 / np.pi
        pitch = pose[0]
        yaw = pose[1]
        roll = pose[2]
        # Bin values    
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1  
        for i in range(sample_number):  
            sample = TrainSample()
            sample.img_path = filepath        
            sample.face_pts = pt2d        
            
            sample.pitch = pose[0]
            sample.yaw = pose[1]
            sample.roll = pose[2]
              
            sample.yaw_bin = binned_pose[0]
            sample.pitch_bin = binned_pose[1]
            sample.roll_bin = binned_pose[2]
            
            sample.flip = np.random.random_sample() < 0.5
            sample.blur = np.random.random_sample() < 0.05
            sample.trans = np.random.random_sample()
            sample.alpha = 1.0 + 0.1 * np.random.random_sample() - 0.05
            sample.beta = 10 * np.random.random_sample()

            k =  sample.trans * 0.2 + 0.2
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])
            x_min -= int(0.6 * k * abs(x_max - x_min))
            y_min -= int(2 * k * abs(y_max - y_min))
            x_max += int(0.6 * k * abs(x_max - x_min))
            y_max += int(0.6 * k * abs(y_max - y_min))
            x_min = int(max(round(x_min),0))
            y_min = int(max(round(y_min),0))
            x_max = int(min(round(x_max),src.shape[1]-1))
            y_max = int(min(round(y_max),src.shape[0]-1))
            
            sample.box=[x_min,x_max,y_min,y_max]
            sample_list.append(sample)
            #break
        #break
    return sample_list

sample_list_file = "sample_list.npz"
if not os.path.exists(sample_list_file):
    #datapath = "/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP"
    datapath = "D:/sandbox/images/300W-LP/300W_LP"
    print("Create samples", datapath)
    sample_list = read_sample_list(datapath, bins)
    np.savez(sample_list_file,sample_list=sample_list)

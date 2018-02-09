import cv2
import dlib
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

from data_sample import FpSample
def read_pts(ptsfile):
  try:
    line_list = []
    with open(ptsfile, "r") as ins:
      for line in ins:
        line_list.append(line.rstrip("\n "))
    start_line = 3
    pts_list=[]
    for i in range(start_line, len(line_list)-1):
      xstr,ystr = line_list[i].split(" ")
      x=float(xstr)
      y=float(ystr)
      pts_list.append([x,y])
    return pts_list
  except:
    print(ptsfile, sys.exc_info()[0])
    raise

def f300W_gen_sample_list(file_list):  
  sample_list = []
  sample_number = 20
  file_idx = 0
  with open(file_list,"r") as ins:
    for line in ins:
      filepath, pts_path = line.rstrip("\n").split(",")
      pt2d = read_pts(pts_path)
      sys.stdout.write("Reading pose: %d %d %s \r" % (file_idx, len(pt2d), filepath) )
      sys.stdout.flush()
      file_idx+=1
      #src = cv2.imread(filepath)
      #print(src.shape)
      #if src.shape[0] == 0 or src.shape[1] == 0:
      #    print("Error in reading image", filepath)
      
      #binned_pose = np.digitize([yaw, pitch, roll], bins) - 1  
      for i in range(sample_number):  
        sample = FpSample()
        sample.img_path = filepath        
        sample.face_pts = pt2d        
        
        sample.flip = np.random.random_sample() < 0.5
        sample.blur = np.random.random_sample() < 0.05
        
        k = i % 4
        t = np.random.random_sample()
        sample.trans = [t for i in range(4)]
        if k > 0:
          idx_array=[0, 1, 2, 3]
          np.random.shuffle(idx_array)
          for i in range(k):
            sample.trans[idx_array[i]] = np.random.random_sample()

        sample.alpha = 1.0 + 0.1 * np.random.random_sample() - 0.05
        sample.beta = 10 * np.random.random_sample()
        if i == 0 :
          sample.rot = 0.0 
        else:
          sample.rot = -20 + np.random.random_sample() * (20 - (-20))
        
        sample_list.append(sample)          
      
  return sample_list

# 300w sample list
if True:
    sample_file = "tmp/sample_300w.npz"
    sample_list = []
    if not os.path.exists(sample_file):
        file_list = "tmp/filelist_300w.txt"
        print("Create samples")
        sample_list = f300W_gen_sample_list(file_list)
        np.savez(sample_file,sample_list=sample_list)

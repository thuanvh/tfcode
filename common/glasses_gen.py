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

from data_sample import TrainSample

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
  coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
  for i in range(0, 68):
    coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
  return coords

def prima_gen_sample_list(prima_file_list, dlib_data, bins):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(dlib_data)
  file_list = []

  sample_list = []
  sample_number = 10
  file_idx = 0
  for file_ele in prima_file_list:
    with open(file_ele,"r") as ins:
      for line in ins:
        #print(line)
        if line.rstrip("\n") == "":
          continue
        filepath = line.rstrip("\n")
        if not (os.path.exists(filepath) :
          continue
        #print(filepath)
        #print(file_idx, len(file_list), filepath)
        sys.stdout.write("Reading image: %d %s \r" % (file_idx, filepath) )
        sys.stdout.flush()
        file_idx+=1
        src = cv2.imread(filepath)
        #print(src.shape)
        if src.shape[0] == 0 or src.shape[1] == 0:
            print("Error in reading image", filepath)
        
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for k, d in enumerate(rects):
          #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
          pt2d = predictor(gray, d)
          pt2d = shape_to_np(pt2d)
          #print(pose)

          for i in range(sample_number):            
            sample.face_pts = pt2d        
                      
          break
  return sample_list

bins = np.array(range(-99, 102, 3))
print(bins, len(bins))

# parser = argparse.ArgumentParser()
# parser.add_argument('--size', type=int,
#                     default=100,
#                     help='Size of output data')
# parser.add_argument('--combineall', type=int,
#                     default=1,
#                     help='Output all label in one file')
# parser.add_argument('--sample_file', type=str,
#                     default="sample_list.npz",
#                     help='sample list file')
# parser.add_argument('--thread', type=int,
#                     default=8,
#                     help='thread')

# FLAGS, unparsed = parser.parse_known_args()

# prima sample list
if True:
    prima_sample_file = "tmp/custom_sample_list.npz"
    prima_sample_list = []
    if not os.path.exists(prima_sample_file):
        #datapath = "/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP"
        #prima_datapath = "C:/out/kinec_head_pose_image_train/"
        #prima_label_path = "C:/out/kinec_head_pose_label/"
        prima_file_list = ["data/2/image_label.1.txt","data/3/image_label.1.txt","data/4/image_label.1.txt",
          "data/5/image_label.1.txt","data/6/image_label.1.txt","data/7/image_label.1.txt","data/8/image_label.1.txt",]
        dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/shape_predictor_68_face_landmarks.dat"
        print("Create samples")
        prima_sample_list = prima_gen_sample_list(prima_file_list, dlib_data, bins)
        np.savez(prima_sample_file,sample_list=prima_sample_list)

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

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/common')

import imgutil
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
  #file_list = []

  sample_list = []
  sample_number = 10
  file_idx = 0
  for filepath in prima_file_list:
    #filepath, rotation_path = line.rstrip("\n").split(",")
    #if not (os.path.exists(filepath) and os.path.exists(rotation_path)) :
    #  continue
    #print(filepath)
    #print(file_idx, len(file_list), filepath)
    sys.stdout.write("Reading pose: %d %s \r" % (file_idx, filepath) )
    sys.stdout.flush()
    file_idx+=1
    src4 = cv2.imread(filepath,-1)
    src = imgutil.blend_alpha(src4, (255,255,255))
    #cv2.imwrite("D:/alpha.png", src)
    
    #print(src.shape)
    if src.shape[0] == 0 or src.shape[1] == 0:
        print("Error in reading image", filepath)
    
    name_eles = os.path.basename(filepath)[:-4].split("_")
    #pose = np.genfromtxt(rotation_path,delimiter=' ')
    yaw = -float(name_eles[1])
    pitch = -float(name_eles[2])
    roll = 0.0
    #print(pitch, yaw, roll)
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for k, d in enumerate(rects):
      #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
      pt2d = predictor(gray, d)
      pt2d = shape_to_np(pt2d)
      #print(pose)

      # Bin values    
      binned_pose = np.digitize([yaw, pitch, roll], bins) - 1  
      for i in range(sample_number):  
        sample = TrainSample()
        sample.img_path = filepath        
        sample.face_pts = pt2d        
        
        sample.pitch = pitch
        sample.yaw = yaw
        sample.roll = roll
          
        sample.yaw_bin = binned_pose[0]
        sample.pitch_bin = binned_pose[1]
        sample.roll_bin = binned_pose[2]
        
        sample.flip = np.random.random_sample() < 0.5
        sample.blur = np.random.random_sample() < 0.05
        sample.trans = np.random.random_sample()
        sample.alpha = 1.0 + 0.4 * np.random.random_sample() - 0.2
        sample.beta = 10 * np.random.random_sample() -5
        
        if i % 2 == 0:
          color = 180 + (255 - 180) * np.random.random_sample()
          sample.bgcolor = [color, color, color]
        else:
          sample.bgcolor = [255 * np.random.random_sample(), 255 * np.random.random_sample(), 255 * np.random.random_sample()]

        k =  sample.trans * 0.2 + 0.2
        x_min = min(pt2d[:,0])
        y_min = min(pt2d[:,1])
        x_max = max(pt2d[:,0])
        y_max = max(pt2d[:,1])
        #cv2.rectangle(src, (x_min, y_min), (x_max,y_max), (30,0,255))
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
      #cv2.rectangle(src, (x_min, y_min), (x_max,y_max), (0,0,255))
      #cv2.rectangle(src, (d.left(), d.top()), (d.right(), d.bottom()), (0,255,0))
      #cv2.polylines(src, [pt2d], True, (0, 255, 255))
      #cv2.imwrite("tmp/"+str(file_idx)+".jpg", src)
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
    prima_sample_file = "tmp/hm3d_sample_list.npz"
    prima_sample_list = []
    if not os.path.exists(prima_sample_file):
      datapath = "D:/sandbox/images/headpose_3d"
      file_list=[]
      for(dirpath, dirnames, filenames) in walk(datapath):
        for f in filenames:
          if f.endswith(".png"):
            file_list.append(dirpath + "/" + f)
      
      dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/shape_predictor_68_face_landmarks.dat"
      print("Create samples")
      prima_sample_list = prima_gen_sample_list(file_list, dlib_data, bins)
      np.savez(prima_sample_file,sample_list=prima_sample_list)

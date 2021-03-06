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
import math

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

# dataset_folder = "C:/Users/Thuan/Downloads/dataset-for-smile-detection-from-face-images"
# lfw_none_smile_file = dataset_folder + "//" + "NON-SMILE_list.txt"
# lfw_smile_file = dataset_folder + "//" + "SMILE_list.txt"
# image_folder = "C:/Users/Thuan/Downloads/lfw/lfw"
# line_list = []
#line_idx = 0
# fsmile=open("data/smile/smile_3.txt","w")
# fnon_smile=open("data/smile/non_smile_3.txt","w")
# with open(lfw_none_smile_file, "r") as ins:
#   for line in ins:
#     fname = line.rstrip("\n ")
#     line_idx += 1
#     if fname != "":
#       name = fname[:-9]
#       filepath = image_folder + "/" + name + "/" + fname
#       #line_list.append((filepath, 0, "non_smile_lfw_" + str(line_idx)))
#       fnon_smile.write(filepath+"\n")
# line_idx = 0
# with open(lfw_smile_file, "r") as ins:
#   for line in ins:
#     fname = line.rstrip("\n ")
#     line_idx += 1
#     if fname != "":
#       name = fname[:-9]
#       filepath = image_folder + "/" + name + "/" + fname
#       fsmile.write(filepath+"\n")
#       #line_list.append((filepath, 1, "smile_lfw_" + str(line_idx)))

# exit(1)

smile_file_list = ["D:/sandbox/utility/tfcode/icao/data/smile/smile_1.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/smile_2.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/smile_3.txt"]
none_smile_file_list = ["D:/sandbox/utility/tfcode/icao/data/smile/non_smile_1.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/non_smile_2.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/non_smile_3.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/non_smile_4.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/non_smile_5.txt",
  "D:/sandbox/utility/tfcode/icao/data/smile/non_smile_6.txt"]
# file_idx = 0
# for smile_file in smile_file_list:
#   file_idx += 1
#   with open(smile_file, "r") as ins:
#     line_idx = 0
#     for line in ins:
#       filepath = line.rstrip("\n ")
#       line_idx += 1
#       if filepath != "":
#         line_list.append((filepath, 1, "smile_" + str(file_idx) + "_" + str(line_idx)))
# file_idx = 0
# for none_smile_file in none_smile_file_list:
#   file_idx += 1
#   with open(none_smile_file, "r") as ins:
#     line_idx = 0
#     for line in ins:
#       filepath = line.rstrip("\n ")
#       line_idx += 1
#       if filepath != "":
#         line_list.append((filepath, 0, "non_smile_" + str(file_idx) + "_" + str(line_idx)))


def append_file(line_list, file_list, label):
  file_idx = 0
  prefix = "non_smile" if label == 0 else "smile"
  for file_list_name in file_list:
    file_idx += 1
    with open(file_list_name, "r") as ins:
      line_idx = 0
      for line in ins:
        filepath = line.rstrip("\n ")
        line_idx += 1
        if filepath != "":
          line_list.append((filepath, label, prefix + "_" + str(file_idx) + "_" + str(line_idx)))

line_list = []
append_file(line_list, smile_file_list, 1)
append_file(line_list, none_smile_file_list, 0)

test_smile_file_list=["D:/sandbox/utility/tfcode/icao/data/smile/test_smile_1.txt"]
test_none_smile_file_list=["D:/sandbox/utility/tfcode/icao/data/smile/test_none_smile_1.txt"]
test_line_list = []
append_file(test_line_list, test_smile_file_list, 1)
append_file(test_line_list, test_none_smile_file_list, 0)

dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)

def gen_sample_list(line_list, sample_number):
  file_idx = 0
  sample_list=[]
  for line in line_list:
    filepath = line[0]
    #print(line, filepath)

    sys.stdout.write("Reading sample: %d %s \r" % (file_idx, filepath) )
    sys.stdout.flush()
    file_idx+=1
    src = cv2.imread(filepath)
    #print(src.shape)
    if src is None or src.shape[0] == 0 or src.shape[1] == 0:
      print("Error in reading image", filepath)
      print(line)
      #print(name)
      continue
    
    label = line[1]

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    imgcx = src.shape[1] / 2
    imgcy = src.shape[0] / 2
    distance = src.shape[1] * src.shape[2]
    one_rect = None
    for k, d in enumerate(rects):
      fx = (d.left() + d.right()) / 2.0
      fy = (d.top() + d.bottom()) / 2.0
      new_distance = math.sqrt((fx - imgcx) * (fx- imgcx) + (fy -imgcy) * (fy-imgcy))
      #print(new_distance, distance)
      if new_distance < distance:
        distance = new_distance
        one_rect = d      
    if one_rect != None:
      #print("Select best rect")
      pt2d = predictor(gray, one_rect)
      pt2d = shape_to_np(pt2d)
      #print(pose)
      roi = src[one_rect.top():one_rect.bottom(), one_rect.left():one_rect.right()]
      cv2.imwrite("tmp/images/smile/" + line[2] + ".jpg", roi)

      for i in range(sample_number):  
        sample = TrainSample()
        sample.img_path = filepath        
        sample.face_pts = pt2d        
        
        sample.label = label
        
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

        sample.alpha = 1.0 + 0.2 * np.random.random_sample() - 0.1
        sample.beta = 10 * np.random.random_sample()
        if i == 0 :
          sample.rot = 0.0 
        else:
          sample.rot = -20 + np.random.random_sample() * (20 - (-20))
          
        sample_list.append(sample)
  return sample_list


biwi_sample_file = "tmp/smile_lfw_sample_list.npz"
if not os.path.exists(biwi_sample_file):
  sample_number = 30
  sample_list = gen_sample_list(line_list, sample_number)
  np.savez(biwi_sample_file,sample_list=sample_list)

test_sample_file = "tmp/smile_test_smile.npz"
if not os.path.exists(test_sample_file):
  test_sample_list = gen_sample_list(test_line_list, 1)
  np.savez(test_sample_file,sample_list=test_sample_list)

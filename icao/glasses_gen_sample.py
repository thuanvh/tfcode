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

line_list = []

glassestinted_file_list = [
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-2.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-3.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-4.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-5.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-6.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-7.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-8.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-9.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-10.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/tinted-11.txt"]
none_glassestinted_file_list = [
  #"D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted.txt",
  #"D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted2.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted3.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted4.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted5.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted7.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted8.txt",
  "D:/sandbox/utility/tfcode/icao/data/glasses-tinted/non_tinted9.txt"]

def append_file(line_list, file_list, label):
  file_idx = 0
  prefix = "non_glassestinted" if label == 0 else "glassestinted"
  for file_list_name in file_list:
    file_idx += 1
    with open(file_list_name, "r") as ins:
      line_idx = 0
      for line in ins:
        filepath = line.rstrip("\n ")
        line_idx += 1
        if filepath != "":
          line_list.append((filepath, label, prefix + "_" + str(file_idx) + "_" + str(line_idx)))
append_file(line_list, glassestinted_file_list, 1)
append_file(line_list, none_glassestinted_file_list, 0)

# test_glassestinted_file_list=["D:/sandbox/utility/tfcode/icao/data/glassestinted/test_glassestinted_1.txt"]
# test_none_glassestinted_file_list=["D:/sandbox/utility/tfcode/icao/data/glassestinted/test_none_glassestinted_1.txt"]
# test_line_list = []
# append_file(test_line_list, test_glassestinted_file_list, 1)
# append_file(test_line_list, test_none_glassestinted_file_list, 0)

dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)

def gen_sample_list(line_list, sample_number, name, startIdx, endIdx):
  file_idx = 0
  sample_list=[]
  #slice_list = line_list
  for lineidx in range(startIdx, endIdx):
    line = line_list[lineidx]
    filepath = line[0]
    #print(line, filepath)

    sys.stdout.write("Reading sample: %d %s \r" % (file_idx, line[2]) )
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
    if len(rects) > 1:
      continue
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
      cv2.imwrite("tmp/images/glasses/" + line[2] + ".jpg", roi)

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
          sample.rot = -10 + np.random.random_sample() * (10 - (-10))
          
        sample_list.append(sample)
  return sample_list


# biwi_sample_file = "tmp/glasses_tinted.npz"
# if not os.path.exists(biwi_sample_file):
#   sample_number = 10
#   sample_list = gen_sample_list(line_list, sample_number)
#   np.savez(biwi_sample_file,sample_list=sample_list)

# test_sample_file = "tmp/test_glassestinted.npz"
# if not os.path.exists(test_sample_file):
#   test_sample_list = gen_sample_list(test_line_list, 1)
#   np.savez(test_sample_file,sample_list=test_sample_list)

class myThread(threading.Thread):
    def __init__(self, threadID, name, input_list, startIdx, endIdx, sample_number):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.name = name
        self.output_list = list()
        self.input_list = input_list
        self.sample_number = sample_number
    def run(self):
        print("starting", self.name, self.startIdx, self.endIdx)
        self.output_list = gen_sample_list(self.input_list, self.sample_number, self.name, self.startIdx, self.endIdx)
        print("exiting", self.name)

thread_num = 8
data_size=int(len(line_list)/thread_num)
thread_list = list()
sample_number = 10
for tid in range(thread_num):
    t = myThread(tid,"a"+str(tid), line_list, data_size * tid, data_size * (tid+1), sample_number)
    thread_list.append(t)
    t.start()
for i in thread_list:
  t.join()
sample_list = list()
for t in thread_list:
  for s in t.output_list:
    sample_list.append(s)

test_sample_file = "tmp/glassestinted.npz"
#if not os.path.exists(test_sample_file):
#   test_sample_list = gen_sample_list(test_line_list, 1)
np.savez(test_sample_file,sample_list=sample_list)



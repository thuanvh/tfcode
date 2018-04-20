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

gaze0_file_list = [
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze0_1.txt",
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze0_2.txt",
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze0_3.txt",
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze0_4.txt",
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze0_5.txt",]
gaze1_file_list = [
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze1_1.txt",
  "D:/sandbox/utility/tfcode/icao/data/gaze/gaze1_2.txt",]

def append_file(line_list, file_list, label):
  file_idx = 0
  prefix = "gaze0" if label == 0 else "gaze1"
  for file_list_name in file_list:
    file_idx += 1
    with open(file_list_name, "r") as ins:
      line_idx = 0
      for line in ins:
        filepath = line.rstrip("\n ")
        line_idx += 1
        if filepath != "":
          line_list.append((filepath, label, prefix + "_" + str(file_idx) + "_" + str(line_idx)))
append_file(line_list, gaze1_file_list, 1)
append_file(line_list, gaze0_file_list, 0)

# test_glassestinted_file_list=["D:/sandbox/utility/tfcode/icao/data/glassestinted/test_glassestinted_1.txt"]
# test_none_glassestinted_file_list=["D:/sandbox/utility/tfcode/icao/data/glassestinted/test_none_glassestinted_1.txt"]
# test_line_list = []
# append_file(test_line_list, test_glassestinted_file_list, 1)
# append_file(test_line_list, test_none_glassestinted_file_list, 0)

dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)

status_dict={}
def print_status(name, idx):
  status_dict[name] = idx
  sys.stdout.write("Reading sample: ")
  #for i in status_dict.keys:
  #    print(status_dict[i])
  #print(status_dict)
  #print(name, status_dict[name])
  for name1 in status_dict.keys():
      sys.stdout.write("%s:%d   " % (name1,status_dict[name1]) )
  sys.stdout.write("\r")
  #sys.stdout.flush()

def gen_sample_list(line_list, sample_number, name, startIdx, endIdx):
  file_idx = 0
  sample_list=[]
  #slice_list = line_list
  for lineidx in range(startIdx, endIdx):
    line = line_list[lineidx]
    filepath = line[0]
    #print(line, filepath)

    #sys.stdout.write("Reading sample: %d %s \r" % (file_idx, line[2]) )
    #sys.stdout.flush()
    print_status(name, file_idx)
    file_idx+=1
    #print("imread")
    src = cv2.imread(filepath)
    #print(src.shape)
    if src is None or src.shape[0] == 0 or src.shape[1] == 0:
      print("Error in reading image", filepath)
      print(line)
      #print(name)
      continue
    
    label = line[1]
    #print("cvtgray")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #print("detector")
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
      #print("predictor")
      pt2d = predictor(gray, one_rect)
      #print("shape_to_np")
      pt2d = shape_to_np(pt2d)
      #print(pose)
      #roi = src[one_rect.top():one_rect.bottom(), one_rect.left():one_rect.right()]
      #cv2.imwrite("tmp/images/" + line[2] + ".jpg", roi)

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

        sample.alpha = 1.0 + 0.1 * np.random.random_sample() - 0.05
        sample.beta = 10 * np.random.random_sample()
        if i == 0 :
          sample.rot = 0.0 
        else:
          sample.rot = -10 + np.random.random_sample() * (10 - (-10))
          
        sample_list.append(sample)
  return sample_list


# biwi_sample_file = "tmp/gaze.npz"
# if not os.path.exists(biwi_sample_file):
#   sample_number = 10
#   sample_list = gen_sample_list(line_list, sample_number)
#   np.savez(biwi_sample_file,sample_list=sample_list)

# test_sample_file = "tmp/test_glassestinted.npz"
# if not os.path.exists(test_sample_file):
#   test_sample_list = gen_sample_list(test_line_list, 1)
#   np.savez(test_sample_file,sample_list=test_sample_list)

class myThread(threading.Thread):
    def __init__(self, threadID, name, input_list, startIdx, endIdx, sample_number, output_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.name = name
        self.output_list = output_list
        self.input_list = input_list
        self.sample_number = sample_number
    def run(self):
        print("starting", self.name, self.startIdx, self.endIdx)
        self.output_list.extend(gen_sample_list(self.input_list, self.sample_number, self.name, self.startIdx, self.endIdx))
        print("exiting", self.name)

thread_num = 8
data_size=int(len(line_list)/thread_num)
sample_list_list = list()
tlist = list()
for tid in range(thread_num):
  sample_ele = list()
  sample_list_list.append(sample_ele)
  t = myThread(tid,"a"+str(tid), line_list, data_size * tid, data_size * (tid+1), 10, sample_ele)
  tlist.append(t)
  t.start()  

for t in tlist:
  t.join()
sample_list = list()
for l in sample_list_list:
  sample_list.extend(l)
np.savez("tmp/gaze.npz", sample_list=sample_list)
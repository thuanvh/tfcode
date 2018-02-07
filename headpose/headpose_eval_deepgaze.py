#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import tensorflow as tf
import cv2
import dlib
import numpy as  np
from thirdparty.deepgaze.src.deepgaze.head_pose_estimation import CnnHeadPoseEstimator

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
  coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
  for i in range(0, 68):
    coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
  return coords

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("thirdparty/deepgaze/src/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("thirdparty/deepgaze/src/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("thirdparty/deepgaze/src/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))

# for i in range(1,9):
#     file_name = "examples\\ex_cnn_head_pose_estimation_images\\" + str(i) + ".jpg"
#     print("Processing image ..... " + file_name)
#     image = cv2.imread(file_name) #Read the image with OpenCV
#     # Get the angles for roll, pitch and yaw
#     roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
#     pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
#     yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
#     print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
#     print("")

dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)

txtfile = "biwi-all.txt"
file_idx = 0
with open(txtfile,"r") as ins:
  for line in ins:
    imgfile, labelfile = line.rstrip("\n").split(",")
    file_idx +=1
    image = cv2.imread(imgfile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
  
    for k, d in enumerate(rects):
      pt2d = predictor(gray, d)
      pt2d = shape_to_np(pt2d)

      x_min = min(pt2d[:,0])
      y_min = min(pt2d[:,1])
      x_max = max(pt2d[:,0])
      y_max = max(pt2d[:,1])

      bbox_width = abs(x_max - x_min) * 1.3
      bbox_height = abs(y_max - y_min) * 1.3

      bbox_width = max(bbox_height, bbox_width)
      bbox_height = bbox_width
      cx = int((x_max + x_min) / 2)
      cy = int((y_max + y_min) / 2)
      d2 = int(bbox_width / 2)
      x_min = cx - d2
      x_max = cx + d2
      y_min = cy - d2
      y_max = cy + d2
      
      roi=image[int(y_min+0.5):int(y_max+0.5),int(x_min+0.5):int(x_max+0.5)]
      cv2.imwrite("tmp"+ str(file_idx) + ".jpg", roi)
      #print(roi.shape)
      roll = my_head_pose_estimator.return_roll(roi).reshape(1)[0]  # Evaluate the roll angle using a CNN
      pitch = my_head_pose_estimator.return_pitch(roi).reshape(1)[0]  # Evaluate the pitch angle using a CNN
      yaw = my_head_pose_estimator.return_yaw(roi).reshape(1)[0]  # Evaluate the yaw angle using a CNN
      
      true_pose = np.genfromtxt(labelfile,delimiter=' ')
      true_pitch = -true_pose[0] #Biwi pose invert
      true_yaw = true_pose[1]
      true_roll = true_pose[2]
      
      print(1,true_yaw, true_pitch, true_roll, yaw, pitch, roll)
      #print(imgfile, labelfile)
      break



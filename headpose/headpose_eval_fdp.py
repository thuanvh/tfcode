import sys
sys.path.insert(0,'thirdparty/faceposenet/faceposenet')
import os
import csv
import numpy as np
import cv2
import math
import pose_utils
import os
import myparse
#import renderer_fpn
import dlib
from data_sample import shape_to_np
## To make tensorflow print less (this can be useful for debug though)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import ctypes; 
print('> loading getRts')
import get_Rts as getRts
######## TMP FOLDER #####################
_tmpdir = './tmp/'#os.environ['TMPDIR'] + '/'
print('> make dir')
if not os.path.exists( _tmpdir):
    os.makedirs( _tmpdir )
#########################################
##INPUT/OUTPUT
input_file = 'input.csv'#str(sys.argv[1]) #'input.csv'
outpu_proc = 'output_preproc.csv'
output_pose_db =  './output_pose.lmdb'
output_render = './output_render'
#################################################
print('> network')
_alexNetSize = 227
_factor = 0.25 #0.1

# ***** please download the model in https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0 ***** #
model_folder = 'thirdparty/faceposenet/faceposenet/fpn_new_model/'
model_used = 'model_0_1.0_1.0_1e-07_1_16000.ckpt' #'model_0_1.0_1.0_1e-05_0_6000.ckpt'
lr_rate_scalar = 1.0
if_dropout = 0
keep_rate = 1
################################

dlib_data = "D:/sandbox/vmakeup/VirtualMakeover/Binaries/Data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)


file_idx = 0
if os.path.exists(input_file):
  data_dict = myparse.parse_input(input_file)
else:
  data_dict = dict()
  txtfile = "biwi-all.txt"
  f = open(input_file, 'w')
  f.write("ID,FILE,FACE_X,FACE_Y,FACE_WIDTH,FACE_HEIGHT,YAW,PITCH,ROLL\n")
  with open(txtfile,"r") as ins:
    for line in ins:
      imgfile, labelfile = line.rstrip("\n").split(",")
      true_pose = np.genfromtxt(labelfile,delimiter=' ')
      true_pitch = -true_pose[0] #Biwi pose invert
      true_yaw = true_pose[1]
      true_roll = true_pose[2]

      sys.stdout.write("%s   " % (imgfile) )
      sys.stdout.write("\r")
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
        
        key = base_name = os.path.basename(imgfile)[:-3]
        data_dict[key] = {'file':imgfile ,\
          'x' : float(x_min),\
          'y' : float(y_min),\
          'width' : float(x_max - x_min),\
          'height' : float(y_max - y_min),\
          'yaw' : true_yaw,
          'pitch' : true_pitch,
          'roll' : true_roll,
          }
        f.write(key + "," + imgfile + "," + str(x_min) + "," + str(y_min) + "," + str(x_max - x_min) + "," + str(y_max - y_min) + "," + str(true_yaw) +  "," + str(true_pitch) + "," + str(true_roll) + "\n")
  f.close()


## Pre-processing the images 
if not os.path.exists(outpu_proc):
  print('> preproc')
  pose_utils.preProcessImage( _tmpdir, data_dict, '.',\
                            _factor, _alexNetSize, outpu_proc )
## Runnin FacePoseNet
print('> run')
## Running the pose estimation
result_dict = getRts.esimatePose( model_folder, outpu_proc, output_pose_db, model_used, lr_rate_scalar, if_dropout, keep_rate, use_gpu=False )
f = open("output.txt", 'w')
keylist = list(data_dict.keys())
keylist.sort()
for k in keylist:
  true_yaw = data_dict[k]["yaw"]
  true_pitch = data_dict[k]["pitch"]
  true_roll = data_dict[k]["roll"]
  yaw, pitch, roll = result_dict[k]
  f.write("%s %f %f %f %f %f %f\n" %(k, true_yaw, true_pitch, true_roll, yaw, pitch, roll))
f.close()
  
#renderer_fpn.render_fpn(outpu_proc, output_pose_db, output_render)

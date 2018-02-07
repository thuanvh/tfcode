import cv2
import dlib
import sys, os, argparse
from os import walk
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import hopenet, utils

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
  coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
  for i in range(0, 68):
    coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
  return coords

snapshot_path = "/home/dev/sandbox/train/fp/hopenet/hopenet_robust_alpha1.pkl"
datapath = "/home/dev/sandbox/train/fp/out/kinec_head_pose_image/"
labelpath = "/home/dev/sandbox/train/fp/out/kinec_head_pose_label/"
dlib_data = "/home/dev/sandbox/train/fp/hopenet/shape_predictor_68_face_landmarks.dat"
# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

#print 'Loading snapshot.'
# Load snapshot
saved_state_dict = torch.load(snapshot_path,map_location=lambda storage, loc: storage)
model.load_state_dict(saved_state_dict)
model.eval()

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

idx_tensor = [idx for idx in range(0,66)]
idx_tensor = torch.FloatTensor(idx_tensor)#.cuda(gpu)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_data)

#predictor = dlib.shape_predictor(dlib_data)
file_list=[]
for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".png"):
            file_list.append(dirpath + "/" + f)
file_list.sort()
#print(len(file_list))

#file_list2 = glob.glob("/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP/**/*.jpg", recursive=True)
#print(len(file_list2))

sample_list = []
sample_number = 10
file_idx = 0

for filepath in file_list:

  file_idx += 1
  src = cv2.imread(filepath)
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  #print("read image")
  rects = detector(gray, 1)
  
  for k, d in enumerate(rects):
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    base_name = os.path.basename(filepath)[:-3]
    rotation_path = labelpath + base_name +"txt"
    true_pose = np.genfromtxt(rotation_path,delimiter=' ')
    true_pitch = -true_pose[0] #Biwi pose invert
    true_yaw = true_pose[1]
    true_roll = true_pose[2]

    pt2d = predictor(gray, d)
    pt2d = shape_to_np(pt2d)

    x_min = min(pt2d[:,0])
    y_min = min(pt2d[:,1])
    x_max = max(pt2d[:,0])
    y_max = max(pt2d[:,1])

    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    bbox_width = max(bbox_height, bbox_width)
    bbox_height = bbox_width
    cx = (x_max + x_min) / 2
    cy = (y_max + y_min) / 2
    x_min = cx - bbox_width / 2
    x_max = cx + bbox_width / 2
    y_min = cy - bbox_height / 2
    y_max = cy + bbox_height / 2

    x_min -= 2 * bbox_width / 4
    x_max += 2 * bbox_width / 4
    y_min -= 3 * bbox_height / 4
    y_max += bbox_height / 4
    img = src[int(y_min):int(y_max),int(x_min):int(x_max)]
    cv2.imwrite("tmp/" + str(file_idx) + ".jpg", img)
    img = Image.fromarray(img)
    # Transform
    img = transformations(img)
    img_shape = img.size()
    #print(img_shape[0], img_shape[1], img_shape[2])
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img)#.cuda(gpu)

    yaw, pitch, roll = model(img)

    # # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)
    #print(yaw_bpred[0], pitch_bpred[0], roll_bpred[0])
    #yaw_bpred_val = yaw_bpred[0] * 3 - 99
    #pitch_bpred_val = pitch_bpred[0] * 3 - 99
    #roll_bpred_val = roll_bpred[0] * 3 - 99

    # # Continuous predictions
    yaw_predicted2 = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted2 = utils.softmax_temperature(pitch.data, 1)
    roll_predicted2 = utils.softmax_temperature(roll.data, 1)

    yaw_predicted2 = torch.sum(yaw_predicted2 * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted2 = torch.sum(pitch_predicted2 * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted2 = torch.sum(roll_predicted2 * idx_tensor, 1).cpu() * 3 - 99

    # print(base_name, true_yaw, true_pitch, true_roll, yaw_predicted[0], pitch_predicted[0], roll_predicted[0])
    
    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)

    #print(yaw_predicted.data[0])
    #print(pitch_predicted.data[0])
    #print(roll_predicted.data[0])
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    # Print new frame with cube and axis
    #txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
    print(base_name, true_yaw, true_pitch, true_roll, yaw_predicted, pitch_predicted, roll_predicted, yaw_predicted2[0], pitch_predicted2[0], roll_predicted2[0])
    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
    #utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
    # Plot expanded bounding box
    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
    break

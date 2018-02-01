import cv2
import dlib
import sys, os, argparse

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
import hopenet

snapshot_path = "/home/dev/sandbox/train/fp/hopenet/hopenet_alpha1.pkl"
datapath = ""
labelpath = ""
# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

#print 'Loading snapshot.'
# Load snapshot
saved_state_dict = torch.load(snapshot_path,map_location=lambda storage, loc: storage)
model.load_state_dict(saved_state_dict)

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

idx_tensor = [idx for idx in range(0,66)]
idx_tensor = torch.FloatTensor(idx_tensor)#.cuda(gpu)

detector = dlib.get_frontal_face_detector()

#predictor = dlib.shape_predictor(dlib_data)

for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".png"):
            file_list.append(dirpath + "/" + f)
print(len(file_list))

#file_list2 = glob.glob("/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP/**/*.jpg", recursive=True)
#print(len(file_list2))

sample_list = []
sample_number = 10
file_idx = 0
for filepath in file_list:
  rotation_path = labelpath + os.path.basename(filepath)[:-3]+"txt"
  pose = np.genfromtxt(rotation_path,delimiter=' ')
  pitch = -pose[0] #Biwi pose invert
  yaw = pose[1]
  roll = pose[2]

  src = cv2.imread(filepath)
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  #print("read image")
  rects = detector(gray, 1)
  
  for k, d in enumerate(rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    #pt2d = predictor(gray, d)
    #pt2d = shape_to_np(pt2d)
    x_min = d.left()
    x_max = d.right()
    y_min = d.top()
    y_max = d.bottom()
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    x_min -= 2 * bbox_width / 4
    x_max += 2 * bbox_width / 4
    y_min -= 3 * bbox_height / 4
    y_max += bbox_height / 4
    img = src[int(y_min):int(y_max),int(x_min):int(x_max)]
    img = Image.fromarray(img)
    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img)#.cuda(gpu)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    # Print new frame with cube and axis
    #txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
    print(yaw_predicted, pitch_predicted, roll_predicted)
    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
    #utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
    # Plot expanded bounding box
    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
    break

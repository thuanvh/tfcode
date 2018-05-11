import os
from os import walk
import numpy as np
# datapath = "D:/sandbox/images/IMAGES/data/images/"
# labelpath = "C:/out/kinec_head_pose_label/"
# file_list=[]
# for(dirpath, dirnames, filenames) in walk(datapath):
#     for f in filenames:
#         if f.endswith(".jpg"):
#             file_list.append(dirpath + "/" + f)
# file_list.sort()
# for filepath in file_list:
#   rotation_path = labelpath + os.path.basename(filepath)[:-3]+"txt"
#   #print(filepath + "," + rotation_path)
#   print(filepath)

# # Gen file list containing _fa or _fb .jpg
# datapath = "D:/sandbox/images/IMAGES/images/"
# for(dirpath, dirnames, filenames) in walk(datapath):
#     for f in filenames:
#         if f.endswith(".jpg") and (("_fa" in f) or ("_fb" in f)):
#             print(dirpath + "/" + f)

datapath = "D:/sandbox/utility/tfcode/headpose/data/8/label"
for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".txt"):
            filepath = dirpath + "/" + f
            pose = np.genfromtxt(filepath,delimiter=' ')
            yaw = pose[0]
            pitch = pose[1]
            roll = pose[2]
            pitch = pitch - int(pitch)
            yaw = yaw - int(yaw)
            datapath_output = "D:/sandbox/utility/tfcode/headpose/data/8/label/"
            fout = open(datapath_output + "/" + f, "w")
            fout.write("%f %f %f" % (yaw, pitch, roll))
            fout.close()





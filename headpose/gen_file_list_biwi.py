import os
from os import walk
datapath = "C:/out/kinec_head_pose_image/"
labelpath = "C:/out/kinec_head_pose_label/"
file_list=[]
for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".png"):
            file_list.append(dirpath + "/" + f)
file_list.sort()
for filepath in file_list:
  rotation_path = labelpath + os.path.basename(filepath)[:-3]+"txt"
  print(filepath + "," + rotation_path)
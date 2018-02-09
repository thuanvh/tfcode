import os
from os import walk
datapath = "D:/sandbox/vmakeup/repos/src/learncnn/traindata/face/train5"
labelpath = "D:/sandbox/vmakeup/repos/src/learncnn/traindata/face/train5"
file_list=[]
for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".jpg") or f.endswith(".png"):
            file_list.append(dirpath + "/" + f)
file_list.sort()
for filepath in file_list:
  rotation_path = labelpath + "/" + os.path.basename(filepath)[:-3]+"pts"
  print(filepath + "," + rotation_path)
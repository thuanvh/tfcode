import os
from os import walk
#datapath = "D:/sandbox/vmakeup/repos/src/learncnn/traindata/face/train5"
#labelpath = "D:/sandbox/vmakeup/repos/src/learncnn/traindata/face/train5"
datapath = ["D:/sandbox/images/300W/01_Indoor","D:/sandbox/images/300W/02_Outdoor"]
labelpath = ["D:/sandbox/images/300W/01_Indoor","D:/sandbox/images/300W/02_Outdoor"]
file_list=[]
for i in range(len(datapath)):
  for(dirpath, dirnames, filenames) in walk(datapath[i]):
    for f in filenames:
      if f.endswith(".jpg") or f.endswith(".png"):
        filepath = dirpath + "/" + f
        ptspath = labelpath[i] + "/" + os.path.basename(filepath)[:-3]+"pts"
        print(filepath + "," + ptspath)
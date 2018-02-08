import os
from os import walk
datapath = "C:/out/headposejpg/"
labelpath = "C:/out/headposejpg_txt/"
file_list=[]
for(dirpath, dirnames, filenames) in walk(datapath):
    for f in filenames:
        if f.endswith(".jpg"):
            file_list.append(dirpath + "/" + f)
file_list.sort()
for filepath in file_list:
  rotation_path = labelpath + os.path.basename(filepath)[:-3]+"txt"
  print(filepath + "," + rotation_path)
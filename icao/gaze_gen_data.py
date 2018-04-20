import numpy as np
import os.path
import sys

gaze_folder = "C:/Users/Thuan/Downloads/columbia_gaze_data_set/Columbia Gaze Data Set/"
# Generate resize command
with open(gaze_folder + "resize.bat",'w') as f:
  for i in range(1,57):
    idname = '{num:04d}'.format(num=i)
    cmd = "cd " + idname
    print(cmd, file=f) #f.write(cmd)
    cmd = "for %%f in (*.jpg) do (magick %%f -resize 1000x1000 %%f_resize.jpg)"
    print(cmd, file=f) #f.write(cmd)
    cmd = "cd .."
    print(cmd, file=f) #f.write(cmd)
  #sys.exit()

file_list0=list()
file_list1=list()
for i in range(1,57):
  idname = '{num:04d}'.format(num=i)
  for v in ["0","5","-5"] :
    for h in ["0","5","-5"] :
      fname = gaze_folder + idname + "/" + idname +"_2m_0P_"+ v + "V_" + h + "H.jpg" + "_resize.jpg"
      if os.path.exists(fname):
        file_list0.append(fname)
  for v in ["10", "-10"]:
    for h in ["0","5","-5","10","-10"] :
      fname = gaze_folder + idname + "/" + idname + "_2m_0P_"+ v + "V_" + h + "H.jpg" + "_resize.jpg"
      if os.path.exists(fname):
        file_list1.append(fname)
  for h in ["10", "-10"]:
    for v in ["0","5","-5","10","-10"] :
      fname = gaze_folder + idname + "/" + idname + "_2m_0P_"+ v + "V_" + h + "H.jpg" + "_resize.jpg"
      if os.path.exists(fname):
        file_list1.append(fname)

file_list0 = list(set(file_list0))
file_list1 = list(set(file_list1))
with open("data/gaze/gaze0_1.txt",'w') as f:
  for i in file_list0:
    print(i, file=f) #f.writelines(i)

with open("data/gaze/gaze1_1.txt",'w') as f:
  for i in file_list1:
    print(i, file=f) #f.writelines(i)

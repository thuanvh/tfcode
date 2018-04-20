import shutil
import os

def readfileresult(icao_file, store_dict, label):
  with open(icao_file, "r") as ins:
    for line in ins:
      fname = line.rstrip("\n ")
      store_dict[fname]=label


icao_file="D:/sandbox/images/PhotoDatabaseValid/valid.txt"
not_icao_file="D:/sandbox/images/PhotoDatabaseValid/invalid.txt"
detect_icaofile="D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/icao_all/valid_glasses.txt"
detect_not_icaofile="D:/sandbox/vmakeup/VirtualMakeover/Binaries/Bin/icao_all/invalid_glasses.txt"

truth_dict = dict()
readfileresult(icao_file, truth_dict, 1)
readfileresult(not_icao_file, truth_dict, 0)

detect_dict = dict()
readfileresult(detect_icaofile, detect_dict, 1)
readfileresult(detect_not_icaofile, detect_dict, 0)

result_list = [list(), list(), list(), list()]
for k in truth_dict.keys():
  if truth_dict[k]==1 and detect_dict[k] == 1 :
    result_list[0].append(k)
  if truth_dict[k]==1 and detect_dict[k] == 0 :
    result_list[1].append(k)
  if truth_dict[k]==0 and detect_dict[k] == 1 :
    result_list[2].append(k)
  if truth_dict[k]==0 and detect_dict[k] == 0 :
    result_list[3].append(k)

print("0 - Truth ICAO - Detect ICAO : ",len(result_list[0]))
print("1 - Truth ICAO - Detect None ICAO : ",len(result_list[1]))
print("2 - Truth None ICAO - Detect ICAO : ",len(result_list[2]))
print("3 - Truth None ICAO - Detect None ICAO : ",len(result_list[3]))

image_folder = "D:/sandbox/images/PhotosDatabase/"
output_folder = "tmp/images/icao/0/"
folder_names = ["icao_detect_icao","icao_detect_none_icao","none_icao_detect_icao","none_icao_detect_none_icao"]
for i in [1,2]:
  folder = output_folder + "/" + folder_names[i]
  if not os.path.exists(folder):
    os.makedirs(folder)
  for f in result_list[i]:
    shutil.copy2(image_folder + "/" + f, folder)
  
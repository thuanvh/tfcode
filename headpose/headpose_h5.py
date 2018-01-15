import cv2
#import dlib
import h5py
import glob
from os import walk
# Read all files
import scipy.io as sio
import numpy as np
import os
import sys
#import thread
import threading

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):    
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def read_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    pt2d = mat['pt2d']
    return pose_params,pt2d

class TrainSample:
    img_path = ""
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    yaw_bin = 0
    pitch_bin = 0
    roll_bin = 0
    face_pts =[]
    blur = False
    flip = False
    trans = 0.0
    alpha = 1.0
    beta = 0
    box=[]#xmin,xmax,ymin,ymax

def create_h5_file(folder,train_file_idx, sample_per_file, channel, height, width):
    h5file = h5py.File(folder + "/train" + str(train_file_idx) + ".h5", "w")
    #print(folder,train_file_idx, sample_per_file, channel, height, width)
    h5file.create_dataset("data", (sample_per_file, channel, height, width), dtype='f4')
    h5file.create_dataset("label", (sample_per_file, 1), dtype='f4')
    return h5file

def create_h5_file_list(suffix, train_file_idx, sample_per_file, channel, height, width):
    h5_file_list = []
    for i in range(len(suffix)):
        h5_file_list.append(create_h5_file(suffix[i],train_file_idx, sample_per_file, channel, height, width))
    return h5_file_list

def add_sample(data_idx, data, label_list, h5_file_list):
    for i in range(len(label_list)):
        h5_file_list[i]['data'][data_idx] = data
        h5_file_list[i]['label'][data_idx] = label_list[i]

def save_h5_file_list(data_h5, label_h5, suffix, train_file_idx):
    print("Save h5 file ", train_file_idx)
    sample_per_file = len(data_h5)
    #print(sample_per_file, data_h5[0].shape)
    channel = data_h5[0].shape[1]
    height = data_h5[0].shape[2]
    width = data_h5[0].shape[3]
    h5_file_list = create_h5_file_list(suffix, train_file_idx, sample_per_file, channel, height, width)
    for i in range(sample_per_file):
        add_sample(i, data_h5[i], label_h5[i], h5_file_list)
    for h5f in h5_file_list:
        h5f.close()


def read_sample_list(datapath, bins):
    file_list = []

    for(dirpath, dirnames, filenames) in walk(datapath):
        for f in filenames:
            if f.endswith(".jpg"):
                file_list.append(dirpath + "/" + f)
    print(len(file_list))

    #file_list2 = glob.glob("/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP/**/*.jpg", recursive=True)
    #print(len(file_list2))

    sample_list = []
    sample_number = 1
    file_idx = 0
    for filepath in file_list:  
        #print(file_idx, len(file_list))
        sys.stdout.write("Reading mat: %d   \r" % (file_idx) )
        sys.stdout.flush()
        file_idx+=1
        src = cv2.imread(filepath)
        #print(src.shape)
        if src.shape[0] == 0 or src.shape[1] == 0:
            print("Error in reading image", filepath)
        mat_path = filepath[:-3] + "mat"
        #pt2d = get_pt2d_from_mat(mat_path)
        #pose = get_ypr_from_mat(mat_path) # We get the pose in radians
        pose, pt2d = read_mat(mat_path)
        #print(pose)
        for i in range(0,3):
            pose[i] = pose[i] * 180 / np.pi
        pitch = pose[0]
        yaw = pose[1]
        roll = pose[2]
        # Bin values    
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1  
        for i in range(sample_number):  
            sample = TrainSample()
            sample.img_path = filepath        
            sample.face_pts = pt2d        
            
            sample.pitch = pose[0]
            sample.yaw = pose[1]
            sample.roll = pose[2]
              
            sample.yaw_bin = binned_pose[0]
            sample.pitch_bin = binned_pose[1]
            sample.roll_bin = binned_pose[2]
            
            sample.flip = np.random.random_sample() < 0.5
            sample.blur = np.random.random_sample() < 0.05
            sample.trans = np.random.random_sample()
            sample.alpha = 1.0 + 0.1 * np.random.random_sample() - 0.05
            sample.beta = 10 * np.random.random_sample()

            k =  sample.trans * 0.2 + 0.2
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])
            x_min -= int(0.6 * k * abs(x_max - x_min))
            y_min -= int(2 * k * abs(y_max - y_min))
            x_max += int(0.6 * k * abs(x_max - x_min))
            y_max += int(0.6 * k * abs(y_max - y_min))
            x_min = int(max(round(x_min),0))
            y_min = int(max(round(y_min),0))
            x_max = int(min(round(x_max),src.shape[1]-1))
            y_max = int(min(round(y_max),src.shape[0]-1))
            
            sample.box=[x_min,x_max,y_min,y_max]
            sample_list.append(sample)
            #break
        #break
    return sample_list

sample_list_file = "sample_list.npz"
sample_list=[]
bins = np.array(range(-99, 102, 3))
print(bins, len(bins))
if not os.path.exists(sample_list_file):
    #datapath = "/media/sf_D_DRIVE/sandbox/images/300W-LP/300W_LP"
    datapath = "D:/sandbox/images/300W-LP/300W_LP"
    print("Create samples", datapath)
    sample_list = read_sample_list(datapath, bins)
    np.savez(sample_list_file,sample_list=sample_list)
else:
    print("Load samples ", sample_list_file)
    data = np.load(sample_list_file)    
    sample_list = data["sample_list"]
np.random.shuffle(sample_list)
#for s in sample_list:
#    print(s.yaw, s.pitch, s.roll)
#print(train_data)
print("Sample list size: ", len(sample_list))

#file_idx = 0

#output_folder = "/media/sf_D_DRIVE/sandbox/vmakeup/repos/src/learncnn/model_face/model29_headpose/_data/headpose_100_6/"
inputsize = 40
height = inputsize #100
width = inputsize #100
output_folder = "D:/sandbox/vmakeup/repos/src/learncnn/model_face/model29_headpose/_data/headpose_" + str(inputsize) + "/"
h5_name = ["yaw_cont", "pitch_cont", "roll_cont"]#, "yaw_bin", "pitch_bin", "roll_bin"]
for i in range(len(h5_name)):
    h5_name[i] = output_folder + h5_name[i]    
    if not os.path.exists(h5_name[i]):
        os.makedirs(h5_name[i])
if not os.path.exists(output_folder + "images/"):
    os.makedirs(output_folder + "images/")
status_dict={}
def print_status(name, idx):
    status_dict[name] = idx
    sys.stdout.write("Reading sample: ")
    #for i in status_dict.keys:
    #    print(status_dict[i])
    #print(status_dict)
    #print(name, status_dict[name])
    for name1 in status_dict.keys():
        sys.stdout.write("%s:%d   " % (name1,status_dict[name1]) )
    sys.stdout.write("\r")
    #sys.stdout.flush()

def sample_gen(rangename,start_idx, end_idx):
    channel = 3
    data_h5=[]
    label_h5=[]
    train_file_idx = 0
    sample_per_file = 1000
    for cur_idx in range(start_idx, end_idx):
        file_idx = cur_idx
        sample = sample_list[cur_idx]
        prefix = output_folder+"images/"+str(file_idx)
        print_status(rangename, cur_idx)
        src = cv2.imread(sample.img_path)

        pt2d = sample.face_pts

        img = src[sample.box[2]:sample.box[3], sample.box[0]:sample.box[1]]#(int(x_min), int(y_min), int(x_max), int(y_max)))
        
        if sample.flip:
            #print(sample.pitch, sample.yaw, sample.roll, sample.pitch_bin, sample.yaw_bin, sample.roll_bin)
            sample.yaw = -sample.yaw
            sample.roll = -sample.roll
            sample.yaw_bin = len(bins)-1-sample.yaw_bin
            sample.roll_bin = len(bins)-1-sample.roll_bin
            img = cv2.flip(img,1)
            #cv2.imwrite(prefix + "_flip.jpg", img)
            #print(sample.pitch, sample.yaw, sample.roll, sample.pitch_bin, sample.yaw_bin, sample.roll_bin)

        if sample.blur:
            img = cv2.blur(img, (3,3))
            #cv2.imwrite(prefix + "_blur.jpg", img)    
        img = cv2.add(cv2.multiply(img, np.array([sample.alpha])), np.array([sample.beta]))
        
        #cv2.imwrite(prefix + ".jpg", img)
        # write to h5
        if len(label_h5) % sample_per_file == 0 :        
            if len(data_h5) > 0 :
                save_h5_file_list(data_h5, label_h5, h5_name, rangename + str(int(cur_idx/sample_per_file)))
                data_h5=[]
                label_h5=[]                

        img = cv2.resize(img, (width, height)).astype(np.float32)
        img *= 2/255.0
        img -= 1
        #cv2.imwrite(prefix + "_norm.jpg",img)
        b, g, r = cv2.split(img)
        data = [b, g, r]
        data = np.reshape(data,(1,channel, height, width))
        label_list = [sample.yaw, sample.pitch, sample.roll]#, sample.yaw_bin, sample.pitch_bin, sample.roll_bin]
        #add_sample(file_idx % sample_per_file, data, label_list, h5_file_list)
        data_h5.append(data)
        label_h5.append(label_list)
        file_idx += 1

    if len(data_h5) > 0 :
        save_h5_file_list(data_h5, label_h5, h5_name, rangename + str(int(cur_idx/sample_per_file)) + "end")            

class myThread(threading.Thread):
    def __init__(self, threadID, name, startIdx, endIdx):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.name = name
    def run(self):
        print("starting", self.name, self.startIdx, self.endIdx)
        sample_gen(self.name, self.startIdx, self.endIdx)
        print("exiting", self.name)
#try:
#    thread.start_new_thread(sample_gen, (0, 2000))
#    thread.start_new_thread(sample_gen, (2001, 3000))
#except:
#    print("Error in start thread")

#sample_gen("t1",0, 100)
#thread1 = myThread(1,"a",0,100)
#thread2 = myThread(2,"b",1000, 1100)
#thread1.start()
#thread2.start()

thread_num = 8
data_size=int(len(sample_list)/thread_num)
for tid in range(thread_num):
    t = myThread(tid,"a"+str(tid), data_size * tid, data_size * (tid+1))
    t.start()



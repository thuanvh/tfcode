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
import argparse

from data_sample import TrainSample

def create_h5_file(folder,train_file_idx, sample_per_file, sample_shape, label_len):
    h5file = h5py.File(folder + "/train" + str(train_file_idx) + ".h5", "w")
    #print(folder,train_file_idx, sample_per_file, channel, height, width)
    #h5file.create_dataset("data", (sample_per_file, channel, height, width), dtype='f4')
    h5file.create_dataset("data", (sample_per_file, sample_shape[0], sample_shape[1], sample_shape[2]), dtype='f4')
    h5file.create_dataset("label", (sample_per_file, label_len), dtype='f4')
    return h5file

def create_h5_file_list(suffix, train_file_idx, sample_per_file, channel, height, width):
    h5_file_list = []
    for i in range(len(suffix)):
        h5_file_list.append(create_h5_file(suffix[i],train_file_idx, sample_per_file, channel, height, width, 1))
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

def save_h5_file_combine(data_h5, label_h5, suffix, train_file_idx):
    print("Save h5 file ", train_file_idx)
    sample_per_file = len(data_h5)
    #print(sample_per_file, data_h5[0].shape)
    sample_shape = [data_h5[0].shape[1], data_h5[0].shape[2], data_h5[0].shape[3]]
    label_len = len(label_h5[0])
    label_h5 = np.reshape(label_h5,(sample_per_file, label_len))
    h5_file = create_h5_file(suffix[0], train_file_idx, sample_per_file, sample_shape, label_len)
    #h5_file['data']=data_h5
    #h5_file['label'] = label_h5
    for i in range(sample_per_file):
        h5_file['data'][i] = data_h5[i]
        h5_file['label'][i] = label_h5[i]
    h5_file.close()

def write_sample(img, pts, name):
  drawing = img.copy()
  for pt in pts:
    cv2.circle(drawing, (int(pt[0]),int(pt[1])), 3, (0,0,255), -1)
  cv2.imwrite(name, drawing)

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int,
                    default=100,
                    help='Size of output data')
parser.add_argument('--combineall', type=int,
                    default=1,
                    help='Output all label in one file')
parser.add_argument('--sample_file', type=str,
                    default="sample_list.npz",
                    help='sample list file')
parser.add_argument('--thread', type=int,
                    default=8,
                    help='thread')

FLAGS, unparsed = parser.parse_known_args()


bins = np.array(range(-99, 102, 3))
print(bins, len(bins))

sample_list_file = FLAGS.sample_file #"sample_list.npz"
sample_list=[]
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
inputsize =  FLAGS.size #40
height = inputsize #100
width = inputsize #100
base_name = os.path.basename(sample_list_file)[:-4]
output_folder = "D:/sandbox/vmakeup/src/learncnn/model_face/model29_headpose/_data/headpose_" + str(inputsize) + "_" + str(base_name) + "/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(output_folder)
if FLAGS.combineall == 0 :
    h5_name = ["yaw_cont", "pitch_cont", "roll_cont", "yaw_bin", "pitch_bin", "roll_bin"]
else:
    h5_name = ["combine"]
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
        if(pt2d.shape[0]==2):
            pt2d = np.zeros((pt2d.shape[1],2))
            for i in range(sample.face_pts.shape[1]):
                pt2d[i][0] = sample.face_pts[0][i]
                pt2d[i][1] = sample.face_pts[1][i]
        #write_sample(src, pt2d, prefix +"_raw.jpg")

        # box_w = sample.box[1] - sample.box[0]
        # box_h = sample.box[3] - sample.box[2]
        # dx = (box_h-box_w)/2
        # startx = max(int(sample.box[0]-dx),0)
        # endx = min(int(sample.box[1]+dx), src.shape[1]-1)
        # #img = src[startx:endx, sample.box[0]:sample.box[1]]#(int(x_min), int(y_min), int(x_max), int(y_max)))
        # #img = src[sample.box[2]:sample.box[3], sample.box[0]:sample.box[1]]#(int(x_min), int(y_min), int(x_max), int(y_max)))
        # img = src[sample.box[2]:sample.box[3], startx:endx]#(int(x_min), int(y_min), int(x_max), int(y_max)))

        eye_leftx, eye_lefty = np.mean(pt2d[36:42], axis=0)
        eye_rightx, eye_righty = np.mean(pt2d[42:48], axis=0)
        eye_cx = (eye_leftx + eye_rightx) * 0.5
        eye_cy = (eye_lefty + eye_righty) * 0.5        
        nose_cx, nose_cy = np.mean(pt2d[31:36], axis=0)
        eye_nose_y = nose_cy - eye_cy
        box_h = eye_nose_y * 3
        box_w = box_h
        box_cx = eye_cx
        box_cy = (eye_cy * 0.3 + nose_cy * 0.7)
        starty = int(box_cy - box_h/2)
        endy = int(box_cy + box_h/2)+1
        startx = int(box_cx - box_w/2)
        endx = int(box_cx + box_w/2)+1
        if(startx < 0 or endx > src.shape[1] or startx >= endx or starty < 0 or endy > src.shape[0] or starty >= endy):
            continue
        img = src[starty:endy,startx:endx]
        
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
        
        #img_suffix = "_" + str(int(sample.yaw))+"_"+str(int(sample.pitch))+"_"+str(int(sample.roll))
        #cv2.imwrite(prefix + img_suffix + ".jpg", img)
                

        img = cv2.resize(img, (width, height)).astype(np.float32)
        #cv2.imwrite(prefix + "_norm.jpg",img)
        img *= 2/255.0
        img -= 1
        
        sample.yaw *= 1/180.0
        sample.pitch *= 1/180.0
        sample.roll *= 1/180.0
        
        # Format of caffe 3 x h x w
        #b, g, r = cv2.split(img)
        #data = [b, g, r]
        #data = np.reshape(data,(1,channel, height, width))

        # Format of tf h x w x 3
        data = img
        data = np.reshape(data, (1, height, width, channel))

        label_list = [sample.yaw, sample.pitch, sample.roll, sample.yaw_bin, sample.pitch_bin, sample.roll_bin]
        #add_sample(file_idx % sample_per_file, data, label_list, h5_file_list)
        data_h5.append(data)
        label_h5.append(label_list)
                # write to h5
        if len(label_h5) % sample_per_file == 0 :        
            if len(data_h5) > 0 :
                if FLAGS.combineall == 0 :
                    save_h5_file_list(data_h5, label_h5, h5_name, rangename + "_" + str(int(cur_idx/sample_per_file)))
                else:
                    save_h5_file_combine(data_h5, label_h5, h5_name, rangename + "_" + str(int(cur_idx/sample_per_file)))
                data_h5=[]
                label_h5=[]
        

    if len(data_h5) > 0 :
        if FLAGS.combineall == 0 :
            save_h5_file_list(data_h5, label_h5, h5_name, rangename + "_" + str(int(cur_idx/sample_per_file)) + "_end")    
        else:
            save_h5_file_combine(data_h5, label_h5, h5_name, rangename + "_" + str(int(cur_idx/sample_per_file)) + "_end")        

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



#tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

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

thread_num = FLAGS.thread #8
data_size=int(len(sample_list)/thread_num)
for tid in range(thread_num):
    t = myThread(tid,"a"+str(tid), data_size * tid, data_size * (tid+1))
    t.start()



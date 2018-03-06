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


#bins = np.array(range(-99, 102, 3))
#print(bins, len(bins))

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
output_folder = "D:/sandbox/vmakeup/src/learncnn/model_face/model29_headpose/_data/glasses_" + str(inputsize) + "_" + str(base_name) + "/"
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
def write_sample(img, pts, name):
  drawing = img.copy()
  for pt in pts:
    cv2.circle(drawing, (int(pt[0]),int(pt[1])), 3, (0,0,255), -1)
  cv2.imwrite(name, drawing)

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
        cx, cy = np.mean(pt2d, axis=0)
        #rotation
        M = cv2.getRotationMatrix2D((cx,cy), sample.rot, 1)
        img = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))        
        pt2d1 = np.c_[pt2d, np.ones(len(pt2d))]
        pt2d = np.dot(pt2d1,M.transpose())        
        #write_sample(img, pt2d, prefix + "_rotation" + ".jpg")

        #translation
        pt_range = slice(17, 48)
        x_min = min(pt2d[pt_range,0])
        y_min = min(pt2d[pt_range,1])
        x_max = max(pt2d[pt_range,0])
        y_max = max(pt2d[pt_range,1])
        w = x_max - x_min
        h = y_max - y_min
        
        #print(sample.trans)
        dx = w * 0.1
        dy = h * 0.1
        x_min -= dx * sample.trans[0]
        x_max += dx * sample.trans[1]
        y_min -= dy * sample.trans[2]
        y_max += dy * sample.trans[3]
        w = x_max - x_min
        h = y_max - y_min
        cx = int((x_max + x_min) * 0.5)
        cy = int((y_max + y_min) * 0.5)

        h = w = max(w,h)
        h2 = int(h/2)
        w2 = int(w/2)
        x_min = cx - w2
        x_max = cx + w2
        y_min = cy - h2
        y_max = cy + h2

        left = top = bottom = right = 0
        if x_min < 0 : left = -x_min        
        if y_min < 0 : top = -y_min
        if x_max > img.shape[1] : right = x_max - img.shape[1]
        if y_max > img.shape[0] : bottom = y_max - img.shape[0]
        img = cv2.copyMakeBorder(img, top=top, bottom=bottom, left=left, right=right, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
        y_min+=top
        y_max+=top
        x_min+=left
        x_max+=left
        img = img[y_min:y_max, x_min:x_max]
        for p in pt2d:
          p[0] = p[0] + left - x_min
          p[1] = p[1] + top - y_min
        #write_sample(img, pt2d, prefix + "_translation" + ".jpg")
        
        #Flip
        if sample.flip:
          img = cv2.flip(img,1)
          for p in pt2d:
            p[0] = img.shape[1] - p[0]
          fromidx = [
            1, 2, 3, 4, 5, 6, 7, 8, 
            10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 
            23, 24, 25, 26, 27, 
            37, 38, 39, 40, 41, 42,
            43, 44, 45, 46, 47, 48,
            32, 33, 35, 36,
            49, 50, 51, 53, 54, 55, 56, 57, 59, 60,
            61, 62, 64, 65, 66, 68
            ]
          toidx = [
            17, 16, 15, 14, 13, 12, 11, 10, 
            8,  7,  6, 5, 4, 3, 2, 1, 
            27, 26, 25, 24, 23, 
            22, 21, 20, 19, 18, 
            46, 45, 44, 43, 48, 47,
            40, 39, 38, 37, 42, 41,
            36, 35, 33, 32, 
            55, 54, 53, 51, 50, 49, 60, 59, 57, 56,
            65, 64, 62, 61, 68, 66
            ]
          pt2d_flip = pt2d.copy()
          for i in range(len(fromidx)):
            pt2d_flip[fromidx[i]-1] = pt2d[toidx[i]-1]
          pt2d = pt2d_flip.copy()
          #write_sample(img, pt2d, prefix + "_flip" + ".jpg")

        #Blur
        if sample.blur:
          img = cv2.blur(img, (3,3))
            #cv2.imwrite(prefix + "_blur.jpg", img)    
        img = cv2.add(cv2.multiply(img, np.array([sample.alpha])), np.array([sample.beta]))
        
        for p in pt2d:
          p[0] = p[0] / img.shape[1]
          p[1] = p[1] / img.shape[0]
        #print(pt2d)
        img = cv2.resize(img, (width, height)).astype(np.float32)
        #cv2.imwrite(prefix + "_translation" + ".jpg", img)
        img *= 2/255.0
        img -= 1
        
        # Format of caffe 3 x h x w
        #b, g, r = cv2.split(img)
        #data = [b, g, r]
        #data = np.reshape(data,(1,channel, height, width))

        # Format of tf h x w x 3
        data = img
        data = np.reshape(data, (1, height, width, channel))
        label_list = np.reshape([sample.label], (1))
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



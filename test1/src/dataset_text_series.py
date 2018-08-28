import argparse
import sys
import tempfile
import numpy as np

import tensorflow as tf
import pickle as pkl
import process
import random

class TextSeriesDataFileList:
  def __init__(self, listfilename, wdict, batch_size, series_len, normalize_data, loop = True, epoch_changed=None):
    #threading.Thread.__init__(self)
    if isinstance(listfilename, str) : 
      with open(listfilename) as f: self.file_list = [line.rstrip('\n') for line in f]
    if isinstance(listfilename, list) : self.file_list = listfilename
    self.file_idx = 0
    self.batch_size = batch_size
    #self.cur_data = []
    #self.cur_label = []
    self.cur_text = []
    self.cur_idx = 0
    #self.y_slice = label_slice
    self.epoch = 0
    self.epoch_changed_event = epoch_changed
    self.wdict = wdict
    self.loop = loop
    #self.reading = False
    self.series_len = series_len
    self.normalize_data = normalize_data

  def get_next_batch(self):
    batch_x = []
    batch_y = []
    while len(batch_x) < self.batch_size :
      if len(self.cur_text) == self.cur_idx :
        self.file_idx += 1
        if self.file_idx >= len(self.file_list) : 
          if (not self.loop) and self.epoch > 0:
            print("End of file")
            return batch_x, batch_y

          #if self.loop or self.epoch == 0:

          self.file_idx = 0
          self.epoch += 1
          print("Epoch ", self.epoch)
          self.reading = True
          if self.epoch_changed_event != None:
            self.epoch_changed_event(self.epoch)
          # else :
          #   print("End of file")
          #   return None, None
        #print("Reading", self.file_list[self.file_idx])
        #f = #h5py.File(self.file_list[self.file_idx],'r')
        #data = f.get('data')
        #label,data = process.open_sample(self.file_list[self.file_idx])
        #self.cur_data = data #np.array(data)
        # Swapaxes for caffe h5
        #self.cur_data = np.swapaxes(np.swapaxes(self.cur_data, 1, 2), 2, 3)
        #img = self.cur_data[0]
        #label = f.get('label')
        #self.cur_label = label #np.array(label)
        self.cur_text = [line.rstrip('\n') for line in open(self.file_list[self.file_idx])]
        self.cur_idx = 0
      start = self.cur_idx
      end = min(self.cur_idx + self.batch_size - len(batch_x), len(self.cur_text))
      #print("Get data from", start, end)
      if len(batch_x) == 0:
        label, data = process.series_samples_lines(self.cur_text[start:end], self.wdict, self.series_len, self.normalize_data)
        batch_x = np.array(data) #self.cur_data[start:end,:]
        batch_y = np.array(label) #self.cur_label[start:end,self.y_slice]
      else:
        #print("Append batch")
        label, data = process.series_samples_lines(self.cur_text[start:end], self.wdict, self.series_len, self.normalize_data)
        batch_x = np.concatenate((batch_x, np.array(data)),axis=0)
        batch_y = np.concatenate((batch_y, np.array(label)),axis=0)
      self.cur_idx = end
    return batch_x, batch_y
  def get_valid_set(self, valid_count):
    all_text = [line.rstrip('\n') for line in open(self.file_list[0])]
    rand_text = random.sample(all_text, valid_count)
    label, data = process.series_samples_lines(rand_text, self.wdict, self.series_len, self.normalize_data)
    batch_x = np.array(data) #self.cur_data[start:end,:]
    batch_y = np.array(label) #self.cur_label[start:end,self.y_slice]
    return batch_x, batch_y

  def get_all(self):
    x = np.array([])
    y = np.array([])
    for fname in self.file_list:
      print("read " + fname)
      #f = h5py.File(h5file,'r')
      #data = f.get('data')
      label,data = process.series_samples_files(fname, self.wdict, self.series_len, self.normalize_data)
      x_data = np.array(data)
      #label = f.get('label')
      y_data = np.array(label)
      print(x_data.shape, y_data.shape)
      if len(x) == 0:
        x = x_data#[:,:]
        y = y_data#[:,self.y_slice]
      else:
        x = np.concatenate((x, x_data), axis=0)
        y = np.concatenate((y, y_data), axis=0)    
      print(x.shape, y.shape)
    return x,y

      
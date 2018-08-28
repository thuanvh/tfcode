
import numpy as np
import pickle
import os

def save_dict(dict, name):
  np.save(name, dict)

def open_dict(name):
  return np.load(name).item()

# def save_sample(label, features, name):
#   data = (label, features)
#   np.save(name, data)

# def open_sample(name):
#   return np.load(name)

def save_sample(label, features, name):
  data = (label, features)
  pickle.dump(data, open(name,"wb"))

def open_sample(name):
  return pickle.load(open(name,"rb"))

def compare_result(ytrue, ypredict, result = None):
  class_number = 2
  if result is None:
    result = np.zeros((class_number, class_number))
  for i in range(len(ytrue)):
    result[ytrue[i]][ypredict[i]] += 1
  return result

#read data
def process_dict(line, wdict):
  label_features = line.split('\t')
  #print(label_features)
  label = label_features[0]
  features = label_features[1]
  features = features.split(',')
  for f in features:
    if f not in wdict :
      wdict[f]=1 #len(wdict)
    else:
      wdict[f] += 1

def binarize_feature(ftext, wdict):
  eles = ftext.split(',')
  features = np.zeros(len(wdict))
  for e in eles:
    if e in wdict :
      features[wdict[e]]=1
  return features

def binarize_samples_files(fname, wdict):
  lines = [line.rstrip('\n') for line in open(fname)]
  return binarize_samples_lines(lines, wdict)

def binarize_samples_lines(lines, wdict):  
  #sample_num = len(lines)
  features = list()
  labels = list()
  feature_only = False
  if len(lines) > 0 :
    if len(lines[0].split('\t')) == 1 :
      feature_only = True
  if feature_only :
    labels = None
    for line in lines:
      features.append(binarize_feature(line, wdict))
  else :
    for line in lines:
      label_features = line.split('\t')
      #print(label_features)
      labels.append(int(label_features[0]))
      features.append(binarize_feature(label_features[1], wdict))
  return (labels, features)

def series_samples_files(fname, wdict, series_len, normalize_data):
  lines = [line.rstrip('\n') for line in open(fname)]
  return series_samples_lines(lines, wdict, series_len, normalize_data)

def series_samples_lines(lines, wdict, series_len, normalize_data):  
  #sample_num = len(lines)
  features = list()
  labels = list()
  feature_only = False
  if len(lines) > 0 :
    if len(lines[0].split('\t')) == 1 :
      feature_only = True
  if feature_only :
    labels = None
    for line in lines:
      features.append(series_feature(line, wdict, series_len, normalize_data))
  else :
    for line in lines:
      label_features = line.split('\t')
      #print(label_features)
      labels.append(int(label_features[0]))
      features.append(series_feature(label_features[1], wdict, series_len, normalize_data))
  return (labels, features)

def series_feature(ftext, wdict, series_len, normalize_data):
  eles = ftext.split(',')
  #print(eles)
  features = np.zeros(series_len)
  index = 0
  scale = 1
  if normalize_data:
    scale = len(wdict) + 1
  for e in eles:
    if e in wdict :
      features[index] = (wdict[e] + 1) / float(scale)
      index += 1
      if index >= series_len :
        break
  #print(features)
  return features

def binarize_samples_files_part(fname, wdict, sample_per_file, output_name):
  lines = [line.rstrip('\n') for line in open(fname)]
  sample_num = len(lines)
  features = list()
  labels = list()
  fileidx = 0
  filelist = open(output_name + ".list.txt", "w")
  for line in lines:
    label_features = line.split('\t')
    #print(label_features)
    labels.append(int(label_features[0]))
    features.append(binarize_feature(label_features[1], wdict))
    if len(labels) == sample_per_file:
      outfile = output_name + str(fileidx)
      save_sample(labels, features, outfile)
      filelist.write(outfile + os.linesep)
      fileidx += 1
      labels = list()
      features = list()
  if len(labels) > 0:
    outfile = output_name + str(fileidx)
    save_sample(labels, features, outfile)
    filelist.write(outfile + os.linesep)
  filelist.close()
    


from sklearn import svm
import pickle
import sys
import process
import numpy as np

if __name__ == "__main__":
  model_name = sys.argv[1]
  eval_name = sys.argv[2]
  label,features = process.open_sample(eval_name)
  #print(label.shape)
  X=np.asarray(features)
  Y=np.asarray(label)
  #np.reshape(Y,(1,len(label)))
  print(X.shape)
  print(Y.shape)
  clf = pickle.load(open(model_name + ".pkl", 'rb'))
  
  result = clf.predict(X)
  print(Y)
  print(result)
  table = process.compare_result(Y, result)
  print(table)
  


from sklearn import svm
import pickle
import sys
import process
import numpy as np

if __name__ == "__main__":
  train_file = sys.argv[1]
  model_name = sys.argv[2]
  label,features = process.open_sample(train_file)
  #print(label.shape)
  X=np.asarray(features)
  Y=np.asarray(label)
  #np.reshape(Y,(1,len(label)))
  print(X.shape)
  print(Y.shape)
  clf = svm.SVC()
  clf.fit(X,Y)
  #result = clf.predict([[2,2]])
  #print(result)
  tuple_objects = (clf)
  pickle.dump(tuple_objects, open(model_name + ".pkl", 'wb'))


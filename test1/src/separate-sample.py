import os
import random

#separate training set to training data and validation data

fname = '../data/training-data-small.txt'
#fname = '../data/training-data-large.txt'

lines = [line for line in open(fname)]
print("Training samples:", len(lines))


fname_train = fname + ".train.txt"
fname_test = fname + ".test.txt"
fname_valid = fname + ".valid.txt"
with open(fname_train, 'w') as f1, open(fname_test, 'w') as f2:
  for line in lines:
    if random.random() > 0.1 :
      f1.write(line)
    else:
      f2.write(line)
  f1.close()
  f2.close()

lines = [line for line in open(fname_train)]
with open(fname_valid, 'w') as f3 :
  for line in lines:
    if random.random() < 0.1 :
      f3.write(line)  
  f3.close()    

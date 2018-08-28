import process
import matplotlib.pyplot as plt
import os
import sys

#  python dict_generate.py ../data/training-data-large.txt ../model/dict-large-all.npy 0
#  python dict_generate.py ../data/training-data-large.txt ../model/dict-large-all.npy 0

if __name__ == "__main__":
  wdict = dict()
  fname = sys.argv[1] #'training-data-small.txt'
  dict_name = sys.argv[2] #'dict-small.npy'
  minoccur = int(sys.argv[3]) #'threshold

  #fname = 'training-data-large.txt'
  #dict_name = 'dict-large.npy'
  lines = [line.rstrip('\n') for line in open(fname)]
  print("Training samples:", len(lines))

  #build dictionary
  for l in lines:
    process.process_dict(l, wdict)
  print("Dictionary size:", len(wdict))

  dict_count = dict()
  for k in wdict:
    c = wdict[k]
    if c in dict_count:
      dict_count[c] += 1
    else:
      dict_count[c] = 1

  maxcount = 10
  sum = 0
  #total = 0
  for i in range(maxcount):
    if i in dict_count:
      sum += dict_count[i]
      print(i, sum, " percent : ", sum / float(len(wdict)))
  #for i in dict_count:
  #  total += dict_count[i]
  
  dictlist = dict()
  for i in wdict:
    if wdict[i] > minoccur :
      dictlist[i] = len(dictlist) #.append(i)
  #print(sum, " percent : ", sum / float(len(wdict)))

  keylist = list(dict_count.keys())
  keylist.sort()
  for i in range(len(keylist)-10, len(keylist)):
    print(keylist[i], dict_count[keylist[i]])

  print("Save dictionary:", dict_name)
  process.save_dict(dictlist, dict_name)


  #plt.plot(xyaw,y1yaw,'r,',xyaw,xyaw,'b-')
  #plt.legend(["yaw1","truth"])
  #plt.savefig(folder+"y1yaw.png")

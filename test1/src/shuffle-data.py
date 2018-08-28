import sys
import os
from random import shuffle

if __name__ == "__main__":
  fname = sys.argv[1]
  fout = sys.argv[2]
  lines = [line for line in open(fname)]
  print("Training samples:", len(lines))
  shuffle(lines)

  with open(fout, 'w') as f1:
    for line in lines:
      f1.write(line)      
    f1.close()
    

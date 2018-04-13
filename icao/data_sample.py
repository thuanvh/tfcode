import numpy as np

class TrainSample:
  img_path = ""  
  face_pts =[]
  blur = False
  flip = False
  trans = []  
  alpha = 1.0
  beta = 0
  box=[]#xmin,xmax,ymin,ymax
  rot = 0
  label = 0
  haze = 0


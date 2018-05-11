import numpy as np

class TrainSample:
  img_path = ""
  yaw = 0.0
  pitch = 0.0
  roll = 0.0
  yaw_bin = 0
  pitch_bin = 0
  roll_bin = 0
  face_pts =[]
  blur = False
  flip = False
  trans = 0.0
  alpha = 1.0
  beta = 0
  box=[]#xmin,xmax,ymin,ymax
  bgcolor=[255,255,255]

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
  coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
  for i in range(0, 68):
    coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
  return coords
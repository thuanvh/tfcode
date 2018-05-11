import cv2
import numpy as np

def blend_alpha(src4, color):
  src3 = src4[:,:,0:3]
  alpha = src4[:,:,3]
  #print(src4.shape)
  #print(src3.shape)
  #print(alpha.shape)
  alpha = alpha.astype(float)/255
  alpha = cv2.merge((alpha,alpha,alpha))
  bgimg = np.zeros(src3.shape, np.uint8)
  bgimg[:,:]=color
  src3 = src3.astype(float)
  bgimg = bgimg.astype(float)
  src3 = cv2.multiply(alpha, src3)
  bgimg = cv2.multiply(1.0 - alpha, bgimg)
  src = cv2.add(src3, bgimg)
  src = np.asarray(src, dtype='uint8')
  return src

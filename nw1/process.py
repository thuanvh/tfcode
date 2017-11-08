import cv2
import glob
import os
import numpy as np
from random import *
def getimages(folder):
    jpglist = glob.glob(folder+"/*.jpg")
    pnglist = glob.glob(folder + "/*.png")
    imglist = jpglist + pnglist
    return imglist

def readpts(file):
    arrlist = np.loadtxt(file)
    return arrlist

class transform_info:
    image = ""
    pts=""
    delta = 0
    dleft = 0
    dright = 0
    dtop = 0
    dbottom = 0
    anglerot = 0
    alpha = 0
    beta = 0
    flipcode = 0

def generate_transform(imgfolder, lblfolder):
    imglist = getimages(imgfolder)
    sample_number = 50
    dmax = 0.01
    dmin = 0.15
    anglemin = -10
    anglemax = 10
    alphamin = 0.8
    alphamax = 1.2
    betamin =-10
    betamax = 10
    trans_list =[]
    for imgfile in imglist:        
        fname = os.path.basename(imgfile)
        fname_no_ext, ext = os.path.splitext(fname)
        lblfile = lblfolder + "/" + fname_no_ext + ".pts"
        print(imgfile)
        print(lblfile)
        if(os.path.exists(lblfile)):
            for i in range(0, sample_number):
                tf = transform_info()
                tf.image = imgfile
                tf.pts = lblfile
                tf.flipcode = i % 4                
                tf.delta = i * (dmax - dmin)/(sample_number -1) + dmin
                tf.dleft = random()
                tf.dright = random()
                tf.dtop = random()
                tf.dbottom = random()
                tf.anglerot = i * (anglemax - anglemin)/(sample_number - 1) + anglemin
                tf.alpha = i * (alphamax - alphamin) / (sample_number - 1) + alphamin
                tf.beta = i * (betamax - betamin) / (sample_number - 1) + betamin
                trans_list.append(tf)

    return trans_list
def swap_point_map(origin, map, pv):
    pv2 = np.copy(pv)
    #print("swap" + str(len(pv)))
    for i in range(0, len(pv)):        
        pv[origin[i]-1] = pv2[map[i]-1]
    

def flip_img_pts(tmat, tvec, flipcode):
    #print("flipcode" + str(flipcode))
    rows = tmat.shape[0]
    cols = tmat.shape[1]
    #print(rows)
    #print(cols)
    origin_map = [
        1, 2, 5, 6,
        4, 3, 8, 7,
        13, 14, 9, 10,
        16, 15, 12, 11,
    ]
    if flipcode == 0:
        pass
    elif flipcode == 1:
        map = [
            16, 15, 12, 11,
            13, 14, 9, 10,
            4, 3, 8, 7,
            1, 2, 5, 6
        ]
        #print(tvec)
        for p in tvec:
            p[1] = rows - p[1] - 1
        #print(tvec)
        swap_point_map(origin_map, map, tvec)
        #print(tvec)
        cv2.flip(tmat, 0, tmat) # flip x-axis
    elif flipcode == 2:
        map = [
            6, 5, 2, 1,
            7, 8, 3, 4,
            10, 9, 14, 13,
            11, 12, 15, 16
        ]
        for p in tvec:
            p[0] = cols - p[0] - 1
        swap_point_map(origin_map, map, tvec)
        cv2.flip(tmat, 1, tmat) # flip y-axis
        
    elif flipcode == 3:
        map = [
            11, 12, 15, 16,
            10, 9, 14, 13,
            7, 8, 3, 4,
            6, 5, 2, 1,
        ]
        for p in tvec:        
            p[1] = rows - p[1] - 1
            p[0] = cols - p[0] - 1
        
        swap_point_map(origin_map, map, tvec)
        cv2.flip(tmat, -1, tmat) # flip xy-axis
        
def geo_rotation_img_pts(mat, center, angle):    
    maxsize = max(mat.shape[0], mat.shape[1])
    r = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    #dst = mat.copy()
    dst = cv2.warpAffine(mat, r, (maxsize,maxsize), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    dst = dst[0:mat.shape[0],0:mat.shape[1]]
    return dst

def geo_rotation(mat, tvec, trans):
    center = (tvec[0] + tvec[10]) * 0.5
    rotmat = geo_rotation_img_pts(mat, center, trans.anglerot)
    pts2 = transform_pts(tvec, 0,0,trans.anglerot, center, 1.0, False)
    return rotmat, pts2

def transform_pts(pts, dx, dy, angle, center, scale, inverse):
    trans = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    if inverse:
        cv2.invertAffineTransform(trans, trans)
    size = len(pts)
    inputMat = np.zeros((3, size))
    i = 0
    for it in pts:
        inputMat[0, i] = it[0] + dx
        inputMat[1, i] = it[1] + dy
        inputMat[2, i] = 1
        i+=1
        #print(i)
    outputMat = np.matmul(trans,inputMat)
    #print("Multiply")
    #print(trans)
    #print(inputMat)
    #print(outputMat)
    pts2 = pts.copy()
    for i in range(0, size):
        pts2[i] = [outputMat[0, i], outputMat[1, i]]
    
    return pts2
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def getroi(mat, box):
    #print(box)
    b2 = box
    return mat[int(b2[1]):int(b2[1] + b2[3]),int(b2[0]):int(b2[0] + b2[2])]

def crop_image(mat, tvec, trans):
    tvec2 = tvec.reshape((-1,1,2)).astype(np.int32)  
    #print(tvec2)  
    bx,by,bw,bh = cv2.boundingRect(tvec2)
    #cv2.rectangle(mat,(bx,by),(bx+bw,by+bh),(0,255,0))
    #cv2.imwrite("crop_bounding.jpg", mat)
    #print(trans)
    dy = bh * trans.delta
    dleft = dy - dy * trans.dleft
    dtop = dy - dy * trans.dtop
    dright = dy - dy * trans.dright
    dbottom = dy - dy * trans.dbottom
    newBound = (bx - dleft, by - dtop,
        bw + dleft + dright, bh + dtop + dbottom)
    #print("intersection")
    #print(newBound)
    newBound = intersection(newBound, (0,0,mat.shape[1]-1,mat.shape[0]-1))
    #cv2.rectangle(mat,(int(newBound[0]),int(newBound[1])),(int(newBound[0]+newBound[2]),int(newBound[1]+newBound[3])),(0,255,0))
    #cv2.imwrite("crop_bounding_ex.jpg", mat)
    #print(newBound)
    mat2 = getroi(mat,newBound)
    #cv2.imwrite("crop.jpg", mat2)
    tvec3 = tvec.copy()
    for p in tvec3:
        p[0] -= newBound[0]
        p[1] -= newBound[1]
    return mat2, tvec3

def output_img_pts(img, pts, name):
    drawing = img.copy()
    pts2 = pts
    pts2 = pts2.astype(int)
    for i in range(0,4):
        for j in range(0,4):
            #print(pts2[i])
            #print(pts2[(i+1)%4])
            cv2.line(drawing, (pts2[i*4 + j][0],pts2[i*4+j][1]), (pts2[i*4+((j+1)%4)][0], pts2[i*4+((j+1)%4)][1]), (0,255,0))
    cv2.imwrite(name, drawing)

def generate_train_data(imgfolder, lblfolder, outputfolder):
    trans_list = generate_transform(imgfolder, lblfolder)
    #print(trans_list)
    shuffle(trans_list)
    #print(trans_list)
    i = 0
    outputsize = 40
    outputimages = []#np.array((len(trans_list), outputsize, outputsize, 3))
    outputlabels = []#np.array((len(trans_list),32))
    for trans in trans_list:
        print(str(i) + "/" + str(len(trans_list)) + " - " + trans.image)
        img = cv2.imread(trans.image)
        pvec = readpts(trans.pts)
        #print(pvec)
        #output_img_pts(img, pvec, outputfolder + "//" + str(i) + ".org.jpg")
        flip_img_pts(img,pvec,trans.flipcode)
        #output_img_pts(img, pvec, outputfolder + "//" + str(i) + ".flip.jpg")
        img, pvec = geo_rotation(img, pvec, trans)
        #output_img_pts(img, pvec, outputfolder + "//" + str(i) + ".rotation.jpg")
        img, pvec = crop_image(img, pvec, trans)
        #output_img_pts(img, pvec, outputfolder + "//" + str(i) + ".crop.jpg")
        #img *= trans.alpha
        #img += trans.beta
        img = cv2.add(cv2.multiply(img, np.array([trans.alpha])), np.array([trans.beta]))
        #output_img_pts(img, pvec, outputfolder + "//" + str(i) + ".alpha.jpg")
        scalex = float(outputsize) / img.shape[1]
        scaley = float(outputsize) / img.shape[0]
        img2 = cv2.resize(img, (outputsize, outputsize)).astype(np.float32)
        #print(pvec)
        for p in pvec:
            p[0] *= scalex
            p[1] *= scaley
        #print(pvec)
        output_img_pts(img2, pvec, outputfolder + "//" + str(i) + ".input.jpg")
        img2 *= 1/255.0
        img2 -= 0.5
        np_image_data = np.asarray(img2)
        
        pvec *= 1.0/outputsize
        #print(pvec)
        pvec = pvec.reshape(32)
        #print(str(pvec.shape))
        #print(str(np_image_data.shape))
        outputimages.append(np_image_data)
        outputlabels.append(pvec)        
        #np.copyto(outputimages[i], np_image_data)
        #np.copyto(outputlabels[i], pvec)
        
        i+=1
        #if i==100 :
        #    break
    #data={"features":outputimages,"labels":outputlabels}
    np.savez("data2.npz",features=outputimages, labels=outputlabels)


imgfolder = "D:/sandbox/utility/scan-image/src/test4"
lblfolder = "D:/sandbox/utility/scan-image/src/test4/pts"
outputfolder = "output"
generate_train_data(imgfolder, lblfolder, outputfolder)


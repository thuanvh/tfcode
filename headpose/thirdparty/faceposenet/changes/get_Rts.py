"""3D pose estimation network: get R and ts                                                                                                                                 
"""
import lmdb
import sys
import time
import csv                                                                                                                                        
import numpy as np
import numpy.matlib
import os

import pose_model as Pose_model
import tf_utils as util

import tensorflow as tf
import scipy
from scipy import ndimage, misc
import os.path
import glob

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'valid', 'train or eval or valid.')
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_string('log_root', '.', 'Directory to keep the checkpoints')
tf.app.flags.DEFINE_string('model_root', '.', 'Directory to keep the checkpoints')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('gpu_id', 0, 'GPU ID to be used.')
tf.app.flags.DEFINE_string('input_csv', 'input.csv', 'input file to process')
tf.app.flags.DEFINE_string('output_lmdb', 'pose_lmdb', 'output lmdb')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')



def run_pose_estimation(root_model_path, inputFile, outputDB, model_used, lr_rate_scalar, if_dropout, keep_rate):

    # Load training images mean: The values are in the range of [0,1], so the image pixel values should also divided by 255 
    file = np.load(root_model_path + "perturb_Oxford_train_imgs_mean.npz")
    train_mean_vec = file["train_mean_vec"]
    del file
    
    # Load training labels mean and std
    file = np.load(root_model_path +"perturb_Oxford_train_labels_mean_std.npz")
    mean_labels = file["mean_labels"]
    std_labels = file["std_labels"]
    del file


    # placeholders for the batches                                                                                                                                      
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    y = tf.placeholder(tf.float32, [FLAGS.batch_size, 6])
    # Thuan
    net_data_v2 = np.load(root_model_path +"PAM_frontal_ALexNet.npy", encoding="bytes").item()
    net_data = dict()
    for k1 in net_data_v2.keys():
        t = dict()
        for k2 in net_data_v2[k1].keys():
            newk=''.join(chr(i) for i in k2)
            t[newk]=net_data_v2[k1][k2]
        net_data[k1] = t
    # Thuan End
    pose_3D_model = Pose_model.ThreeD_Pose_Estimation(x, y, 'valid', if_dropout, keep_rate, keep_rate, lr_rate_scalar, net_data, FLAGS.batch_size, mean_labels, std_labels)
    pose_3D_model._build_graph()
    del net_data

    # #Add ops to save and restore all the variables.   
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.VARIABLES, scope='Spatial_Transformer'))

    #pose_lmdb_env = lmdb.Environment(outputDB, map_size=1e12) #Thuan

    # Thuan
    result_dict = dict()
    graph_location="."
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True )) as sess:
        #pose_lmdb_env.begin(write=True) as pose_txn:

        
        # Restore variables from disk.
        load_path = root_model_path + model_used
        print("Save graph")
        tf.train.write_graph(sess.graph_def, ".", "graph.pbtxt", True) #proto
        saver.restore(sess, load_path)

        print("Model restored.")
        
        save_path = saver.save(sess, "./model_saved_0"+".ckpt")# Thuan
        print("Model saved in file: %s" % save_path)# Thuan 
        # Load cropped and scaled image file list (csv file)
        with open(inputFile, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            lines = csvfile.readlines()
            for lin in lines:
                ### THE file is of the form
                ### key1, image_path_key_1
                mykey = lin.split(',')[0]
                image_file_path = lin.split(',')[-1].rstrip('\n')
                import cv2
                image = cv2.imread(image_file_path)
                image = np.asarray(image)
                # Fix the 2D image
                if len(image.shape) < 3:
                    image_r = np.reshape(image, (image.shape[0], image.shape[1], 1))
                    image = np.append(image_r, image_r, axis=2)
                    image = np.append(image, image_r, axis=2)

                label = np.array([0.,0.,0.,0.,0.,0.])
                id_labels = np.array([0])
                
                # Normalize images and labels 
                nr_image, nr_pose_label, id_label = util.input_processing(image, label, id_labels, train_mean_vec, mean_labels, std_labels, 1, FLAGS.image_size, 739)
                del id_label

                # Reshape the image and label to fit model
                nr_image = nr_image.reshape(1, FLAGS.image_size, FLAGS.image_size, 3)
                nr_pose_label = nr_pose_label.reshape(1,6)

                # Get predicted R-ts
                pred_Rts = sess.run(pose_3D_model.preds_unNormalized, feed_dict={x: nr_image, y: nr_pose_label})
                pose = pred_Rts[0].astype('float32').flatten() # Thuan 
                pitch= pose[0] * 180 / 3.14
                yaw=pose[1] * 180 / 3.14
                roll = pose[2] * 180 / 3.14
                result_dict[mykey] = (yaw, pitch, roll)
                #print('Predicted pose for:  ' , mykey, yaw, pitch, roll)
                sys.stdout.write("%s %f %f %f  " % (mykey, yaw, pitch, roll) )
                sys.stdout.write("\r")
                #pose_txn.put( mykey , pred_Rts[0].astype('float32') )
    return result_dict # Thuan


def esimatePose(root_model_path, inputFile, outputDB, model_used, lr_rate_scalar, if_dropout, keep_rate, use_gpu=False ):
    ## Force TF to use CPU oterwise we set the ID of the string of GPU we wanna use but here we are going use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' #e.g. str(FLAGS.gpu_id)# '7'
    if use_gpu == False:
        dev = '/cpu:0'
        print("Using CPU")# Thuan
    elif usc_gpu == True:
        dev = '/gpu:0'
        print("Using GPU " + os.environ['CUDA_VISIBLE_DEVICES'])# Thuan
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    result_dict = run_pose_estimation( root_model_path, inputFile, outputDB, model_used, lr_rate_scalar, if_dropout, keep_rate )
    return result_dict # Thuan

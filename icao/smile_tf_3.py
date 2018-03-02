#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import tensorflow as tf
import h5py

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""  
  input_layer = tf.reshape(features, [-1, 40, 40, 3], name="input")
#   conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
#   pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#   conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
#   pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#   pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 64])
#   dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#   dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#   logits = tf.layers.dense(inputs=dropout, units=2)
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  #x = tf.placeholder(tf.float32, [None, 40, 40, 3])
  #tf.feed(x,features)
  #conv1 = tf.layers.conv2d(inputs=x,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  conv3 = tf.layers.conv2d(inputs=pool2,filters=60,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  conv4 = tf.layers.conv2d(inputs=pool3,filters=80,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  conv5 = tf.layers.conv2d(inputs=pool4,filters=100,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  pool4_flat = tf.reshape(pool5, [-1, 1 * 1 * 100])
  dense = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu)
  #keep_prob = tf.placeholder_with_default(1.0, None) #tf.float32)
  #dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4)
  logits = tf.layers.dense(inputs=dropout, units=2, name="logits")

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #labelstf = tf.reshape(labels, [-1, 2])
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class H5DataList:
  def __init__(self, listfilename, batch_size, label_slice=slice(1)):
    #threading.Thread.__init__(self)
    with open(listfilename) as f: self.file_list = [line.rstrip('\n') for line in f]
    self.file_idx = 0
    self.batch_size = batch_size
    self.cur_data = []
    self.cur_label = []
    self.cur_idx = 0
    self.y_slice = label_slice
  def get_next_batch(self):
    batch_x = []
    batch_y = []
    while len(batch_x) < self.batch_size :
      if len(self.cur_data) == self.cur_idx :
        self.file_idx += 1
        if self.file_idx >= len(self.file_list) : self.file_idx = 0        
        print("Reading", self.file_list[self.file_idx])
        f = h5py.File(self.file_list[self.file_idx],'r')
        data = f.get('data')
        self.cur_data = np.array(data)
        # Swapaxes for caffe h5
        #self.cur_data = np.swapaxes(np.swapaxes(self.cur_data, 1, 2), 2, 3)
        #img = self.cur_data[0]
        label = f.get('label')
        self.cur_label = np.array(label)
        self.cur_idx = 0
      start = self.cur_idx
      end = min(self.cur_idx + self.batch_size - len(batch_x), len(self.cur_data))
      print("Get data from", start, end)
      if len(batch_x) == 0:
        batch_x = self.cur_data[start:end,:]
        batch_y = self.cur_label[start:end,self.y_slice]
      else:
        #print("Append batch")
        batch_x = np.concatenate((batch_x,self.cur_data[start:end,:]),axis=0)
        batch_y = np.concatenate((batch_y,self.cur_label[start:end,self.y_slice]),axis=0)
      self.cur_idx = end
    return batch_x, batch_y
  def get_all(self):
    x = np.array([])
    y = np.array([])
    for h5file in self.file_list:
      print("read " + h5file)
      f = h5py.File(h5file,'r')
      data = f.get('data')
      x_data = np.array(data)
      label = f.get('label')
      y_data = np.array(label)
      print(x_data.shape, y_data.shape)
      if len(x) == 0:
        x = x_data#[:,:]
        y = y_data[:,self.y_slice]
      else:
        x = np.concatenate((x, x_data[:,:]), axis=0)
        y = np.concatenate((y, y_data[:,self.y_slice]), axis=0)    
      print(x.shape, y.shape)
    return x,y

def h5_feed(h5data):
  with tf.Session() as sess:
    batch_x, batch_y = h5data.get_next_batch()
    print(batch_x.shape, batch_y.shape)
    batch_y = tf.cast(batch_y, tf.int64)
    #batch_y = tf.one_hot(indices=batch_y, depth=2).eval(session=sess)
    #print(batch_y.shape)
    #batch_y = np.reshape(batch_y, (batch_y.shape[0]* 2))
    #batch_y = tf.cast(batch_y, tf.int64)
    return (batch_x, batch_y)

def h5_feed_all(h5data):
  with tf.Session() as sess:
    batch_x, batch_y = h5data.get_all()
    batch_y = tf.cast(batch_y, tf.int64)
    #batch_y = tf.one_hot(indices=batch_y, depth=2).eval(session=sess)
    print(batch_x.shape,batch_y.shape)
    #batch_y = np.reshape(batch_y, (batch_y.shape[0]* 2))
    #batch_y = tf.cast(batch_y, tf.int64)
    return (batch_x, batch_y)

def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="model_40_3")

  if FLAGS.phase == "train":
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir="model_40_3", save_steps=500)

    batch_size = 100
    #data_size = train_data.shape[0]
    file_name = FLAGS.data_file #"D:\\sandbox\\vmakeup\\model\\headpose\\pitch\\data\\filelist.txt"  
    #print("Slice object",label_slice)
    h5data = H5DataList(file_name, batch_size, slice(1))
    # Train the model
    #   train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #       x={"x": train_data},
    #       y=train_labels,
    #       batch_size=100,
    #       num_epochs=None,
    #       shuffle=True)
    train_input_fn = lambda: h5_feed(h5data)
    #mnist_classifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])
    mnist_classifier.train(input_fn=train_input_fn,steps=10000, hooks=[saver_hook])

  if FLAGS.phase == "test":
    file_name = FLAGS.data_file
    batch_size = 100
    h5data = H5DataList(file_name, batch_size, slice(1))
    #eval_data, eval_labels = h5data.get_all()
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data,y=eval_labels,num_epochs=1,shuffle=False)
    eval_input_fn = lambda: h5_feed_all(h5data)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', type=str,
                      default='model_100_3',
                      help='Directory for storing input data')
  parser.add_argument('--restore', type=str,
                      default='',
                      help='Directory for storing input data')
  parser.add_argument('--input_size', type=int,
                      default=100,
                      help='Input size of model 40, 100')
  parser.add_argument('--data_file', type=str,
                      default="train.txt",
                      help='train.txt data file list')
  parser.add_argument('--label_type', type=str,
                      default="cont",
                      help='use continue angle : cont, bin')
  parser.add_argument('--label_slice', type=str,
                      default="1",
                      help='use slice : 1, 1:3')
  parser.add_argument('--max_iter', type=int,
                      default=200000,
                      help='max_iter')
  parser.add_argument('--save_iter', type=int,
                      default=10000,
                      help='save_iter')
  parser.add_argument('--info_iter', type=int,
                      default=1000,
                      help='info_iter')
  parser.add_argument('--phase', type=str,
                      default="train",
                      help='train or test')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()

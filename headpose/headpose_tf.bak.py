# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import h5py

FLAGS = None

# create a slice object from a string
def get_slice_obj(slicearg):
    #slice_ints = tuple([ int(n) for n in slicearg.split(':') ])
    #return apply(slice, slice_ints)
    return slice(*map(lambda x: int(x.strip()) if x.strip() else None, slicearg.split(':')))

def ints_from_slicearg(slicearg):
    slice_obj = get_slice_obj(slicearg)
    return(range(slice_obj.start or 0, slice_obj.stop or -1, slice_obj.step or 1))

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  #with tf.name_scope('reshape'):
  #  x_image = tf.reshape(x, [-1, 100, 100, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 3, 20])
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 20, 40])
    b_conv2 = bias_variable([40])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, 40, 60])
    b_conv3 = bias_variable([60])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Second pooling layer.
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, 60, 80])
    b_conv4 = bias_variable([80])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

  # Second pooling layer.
  with tf.name_scope('pool4'):
    h_pool4 = max_pool_2x2(h_conv4)
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([3, 3, 80, 100])
    b_conv5 = bias_variable([100])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

  # Second pooling layer.
  with tf.name_scope('pool5'):
    h_pool5 = max_pool_2x2(h_conv5)
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv6'):
    W_conv6 = weight_variable([3, 3, 100, 120])
    b_conv6 = bias_variable([120])
    h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)

  # Second pooling layer.
  with tf.name_scope('pool6'):
    h_pool6 = max_pool_2x2(h_conv6)
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv7'):
    W_conv7 = weight_variable([3, 3, 120, 140])
    b_conv7 = bias_variable([140])
    h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)

  # Second pooling layer.
  with tf.name_scope('pool7'):
    h_pool7 = max_pool_2x2(h_conv7)
  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([1 * 1 * 140, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool7, [-1, 1 * 1 * 140])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  #with tf.name_scope('dropout'):
  #  keep_prob = tf.placeholder(tf.float32)
  #  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv#, keep_prob

def cnn_model_fn_224(x):
  """Model function for CNN."""    
  conv1 = tf.layers.conv2d(inputs=x,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  conv3 = tf.layers.conv2d(inputs=pool2,filters=60,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  conv4 = tf.layers.conv2d(inputs=pool3,filters=80,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  conv5 = tf.layers.conv2d(inputs=pool4,filters=100,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  conv6 = tf.layers.conv2d(inputs=pool5,filters=120,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
  conv7 = tf.layers.conv2d(inputs=pool6,filters=140,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool7 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
  pool4_flat = tf.reshape(pool6, [-1, 1 * 1 * 140])
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
  keep_prob = tf.placeholder_with_default(1.0, None)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  logits = tf.layers.dense(inputs=dropout, units=1, name="logits")
  #logits = tf.layers.dense(inputs=dense, units=32)
  return logits, keep_prob
def cnn_model_fn_100(x):
  """Model function for CNN."""    
  conv1 = tf.layers.conv2d(inputs=x,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(inputs=pool1,filters=40,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  conv3 = tf.layers.conv2d(inputs=pool2,filters=60,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  conv4 = tf.layers.conv2d(inputs=pool3,filters=80,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  conv5 = tf.layers.conv2d(inputs=pool4,filters=100,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  conv6 = tf.layers.conv2d(inputs=pool5,filters=120,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[3, 3], strides=1)
  #conv7 = tf.layers.conv2d(inputs=pool6,filters=140,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
  #pool7 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
  pool4_flat = tf.reshape(pool6, [-1, 1 * 1 * 120])
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
  keep_prob = tf.placeholder_with_default(1.0, None)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  logits = tf.layers.dense(inputs=dropout, units=1, name="logits")
  #logits = tf.layers.dense(inputs=dense, units=32)
  return logits, keep_prob
def cnn_model_fn_40(x):
  """Model function for CNN."""    
  conv1 = tf.layers.conv2d(inputs=x,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
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
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
  keep_prob = tf.placeholder_with_default(1.0, None) #tf.float32)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  logits = tf.layers.dense(inputs=dropout, units=1, name="logits")
  #logits = tf.layers.dense(inputs=dense, units=32)
  return logits, keep_prob

def cnn_model_bin_fn_40(x, binsize):
  """Model function for CNN."""    
  conv1 = tf.layers.conv2d(inputs=x,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
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
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
  keep_prob = tf.placeholder_with_default(1.0, None) #tf.float32)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  logits = tf.layers.dense(inputs=dropout, units=binsize, name="logits")
  #logits = tf.layers.dense(inputs=dense, units=32)
  return logits, keep_prob

def cnn_model_fn(x, input_size, is_bin, bins):
  if not is_bin:
    if input_size == 40:
      return cnn_model_fn_40(x)
    elif input_size == 100:
      return cnn_model_fn_100(x)
  else:
    if input_size == 40:
      return cnn_model_bin_fn_40(x, len(bins))

  
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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
        #print("Reading", self.file_list[self.file_idx])
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
      #print("Get data from", start, end)
      if len(batch_x) == 0:
        batch_x = self.cur_data[start:end,:]
        batch_y = self.cur_label[start:end,self.y_slice]
      else:
        #print("Append batch")
        batch_x = np.concatenate((batch_x,self.cur_data[start:end,:]),axis=0)
        batch_y = np.concatenate((batch_y,self.cur_label[start:end,self.y_slice]),axis=0)
      self.cur_idx = end
    return batch_x, batch_y

      

def main(_):
  bins = np.array(range(-99, 102, 3))
  print(bins, len(bins))
  is_binned_label = FLAGS.label_type == 'bin'
  print(FLAGS.label_type)
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #data = np.load("data_4_40.npz")
  #print(data.files)
  #train_data = data["features"]
  #train_labels = data["labels"]
  #print(train_data.shape)
  #print(train_labels.shape)
  input_size = FLAGS.input_size
  # Create the model
  x = tf.placeholder(tf.float32, [None, input_size, input_size, 3])

  # Define loss and optimizer
  if is_binned_label:
    y_ = tf.placeholder(tf.int64, [None, len(bins)])
  else:
    y_ = tf.placeholder(tf.float32, [None, 1])

  # Build the graph for the deep net
  #y_conv = deepnn(x)
  y_conv, keep_prob = cnn_model_fn(x, input_size, is_binned_label, bins)

  with tf.name_scope('loss'):
    if is_binned_label:
      lossfn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv)
    else:
      lossfn = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)      
  tf.summary.scalar('loss', lossfn)
  #lossfn = tf.reduce_mean(lossfn)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(lossfn)

  if is_binned_label:
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

  graph_location = FLAGS.model_dir
  print('Saving graph to: %s' % graph_location)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  batch_size = 100
  #data_size = train_data.shape[0]
  file_name = FLAGS.data_file #"D:\\sandbox\\vmakeup\\model\\headpose\\pitch\\data\\filelist.txt"
  label_slice = get_slice_obj(FLAGS.label_slice)
  print("Slice object",label_slice)
  h5data = H5DataList(file_name, batch_size, label_slice)
  max_iter = FLAGS.max_iter + 1
  save_iter= FLAGS.save_iter
  info_iter= FLAGS.info_iter
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tf.train.write_graph(sess.graph_def, graph_location, "graph.pbtxt", True) #proto
    if FLAGS.restore != "" :
      saver.restore(sess, FLAGS.restore)

    for i in range(max_iter):
      
      # #batch = mnist.train.next_batch(50)
      # #start = (i * batch_size)%data_size
      # #end = ((i+1) * batch_size)%data_size
      # #print("start %d end %d" % (start,end))
      # if start < end :
      #   batch_x = train_data[start:end,:]
      #   batch_y = train_labels[start:end,:]
      #   #print(batch_y.shape)
      # else:
      #   batch_x = np.concatenate((train_data[start:data_size],train_data[0:end]),axis=0)
      #   batch_y = np.concatenate((train_labels[start:data_size],train_labels[0:end]),axis=0)
      batch_x, batch_y = h5data.get_next_batch()
      if is_binned_label :
        batch_y *= 180
        batch_y = np.digitize(batch_y, bins) - 1  
        batch_y = tf.one_hot(indices=batch_y, depth=len(bins)).eval()
        batch_y = np.reshape(batch_y, (batch_y.shape[0],len(bins)))
      #print(batch_x.shape)
      #print(batch_y.shape)
      if i % info_iter == 0:
        #print('step ', i)
        if is_binned_label :
          train_loss = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
          print('step %d, accuracy %g' % (i, train_loss))
        else:
          train_loss = lossfn.eval(feed_dict={x: batch_x, y_: batch_y})
          print('step %d, training loss %g' % (i, train_loss))

      if i % save_iter == 0:
        save_path = saver.save(sess, graph_location + "/"+"model"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)
      #train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
      #train_writer.add_summary(summary, i)
      # if i % 100 == 99:  # Record execution stats
      #   run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      #   run_metadata = tf.RunMetadata()
      #   summary, _ = sess.run([merged, train_step],
      #                          feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5},
      #                          options=run_options,
      #                          run_metadata=run_metadata)
      #   train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      #   train_writer.add_summary(summary, i)
      #   print('Adding run metadata for', i)
      # else: # Record a summary
      #summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
      #train_writer.add_summary(summary, i)
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
      train_writer.add_summary(summary, i)

    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  train_writer.close()

if __name__ == '__main__':
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
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

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

FLAGS = None


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
    W_fc2 = weight_variable([1024, 32])
    b_fc2 = bias_variable([32])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv#, keep_prob

def cnn_model_fn(x):
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
  keep_prob = tf.placeholder(tf.float32)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  logits = tf.layers.dense(inputs=dropout, units=8, name="logits")
  #logits = tf.layers.dense(inputs=dense, units=32)
  return logits, keep_prob

  
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


def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  data = np.load("data_4_40.npz")
  print(data.files)
  train_data = data["features"]
  train_labels = data["labels"]
  print(train_data.shape)
  print(train_labels.shape)
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 40, 40, 3])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 8])

  # Build the graph for the deep net
  #y_conv = deepnn(x)
  y_conv, keep_prob = cnn_model_fn(x)

  with tf.name_scope('loss'):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
    #                                                        logits=y_conv)
    lossfn = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)
  lossfn = tf.reduce_mean(lossfn)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(lossfn)

  #with tf.name_scope('accuracy'):
  #  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  #  correct_prediction = tf.cast(correct_prediction, tf.float32)
  #accuracy = tf.reduce_mean(correct_prediction)

  graph_location = FLAGS.model_dir
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  batch_size = 100
  data_size = train_data.shape[0]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tf.train.write_graph(sess.graph_def, graph_location, "graph.pbtxt", True) #proto
    if FLAGS.restore != "" :
      saver.restore(sess, FLAGS.restore)
    for i in range(20001):
      #batch = mnist.train.next_batch(50)
      start = (i * batch_size)%data_size
      end = ((i+1) * batch_size)%data_size
      #print("start %d end %d" % (start,end))
      if start < end :
        batch_x = train_data[start:end,:]
        batch_y = train_labels[start:end,:]
        #print(batch_y.shape)
      else:
        batch_x = np.concatenate((train_data[start:data_size],train_data[0:end]),axis=0)
        batch_y = np.concatenate((train_labels[start:data_size],train_labels[0:end]),axis=0)
      if i % 100 == 0:
        train_loss = lossfn.eval(feed_dict={
            x: batch_x, y_: batch_y})
        print('step %d, training loss %g' % (i, train_loss))
      if i % 1000 == 0:
        save_path = saver.save(sess, graph_location + "/"+"model"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)
      train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', type=str,
                      default='model_100_3',
                      help='Directory for storing input data')
  parser.add_argument('--restore', type=str,
                      default='',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

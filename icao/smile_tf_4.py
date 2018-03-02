# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import h5py
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

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

def train():
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir,
  #                                   one_hot=True,
  #                                   fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 40, 40, 3], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

  # with tf.name_scope('input_reshape'):
  #   image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  #   tf.summary.image('input', image_shaped_input, 10)

  # # We can't initialize these variables to 0 - the network will get stuck.
  # def weight_variable(shape):
  #   """Create a weight variable with appropriate initialization."""
  #   initial = tf.truncated_normal(shape, stddev=0.1)
  #   return tf.Variable(initial)

  # def bias_variable(shape):
  #   """Create a bias variable with appropriate initialization."""
  #   initial = tf.constant(0.1, shape=shape)
  #   return tf.Variable(initial)

  # def variable_summaries(var):
  #   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  #   with tf.name_scope('summaries'):
  #     mean = tf.reduce_mean(var)
  #     tf.summary.scalar('mean', mean)
  #     with tf.name_scope('stddev'):
  #       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  #     tf.summary.scalar('stddev', stddev)
  #     tf.summary.scalar('max', tf.reduce_max(var))
  #     tf.summary.scalar('min', tf.reduce_min(var))
  #     tf.summary.histogram('histogram', var)

  # def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  #   """Reusable code for making a simple neural net layer.

  #   It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  #   It also sets up name scoping so that the resultant graph is easy to read,
  #   and adds a number of summary ops.
  #   """
  #   # Adding a name scope ensures logical grouping of the layers in the graph.
  #   with tf.name_scope(layer_name):
  #     # This Variable will hold the state of the weights for the layer
  #     with tf.name_scope('weights'):
  #       weights = weight_variable([input_dim, output_dim])
  #       variable_summaries(weights)
  #     with tf.name_scope('biases'):
  #       biases = bias_variable([output_dim])
  #       variable_summaries(biases)
  #     with tf.name_scope('Wx_plus_b'):
  #       preactivate = tf.matmul(input_tensor, weights) + biases
  #       tf.summary.histogram('pre_activations', preactivate)
  #     activations = act(preactivate, name='activation')
  #     tf.summary.histogram('activations', activations)
  #     return activations

  # hidden1 = nn_layer(x, 784, 500, 'layer1')

  # with tf.name_scope('dropout'):
  #   keep_prob = tf.placeholder(tf.float32)
  #   tf.summary.scalar('dropout_keep_probability', keep_prob)
  #   dropped = tf.nn.dropout(hidden1, keep_prob)

  # # Do not apply softmax activation yet, see below.
  # y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

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
  dense = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu)
  keep_prob = tf.placeholder_with_default(1.0, None) #tf.float32)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
  #dropout = tf.layers.dropout(inputs=dense, rate=0.4)
  y = tf.layers.dense(inputs=dropout, units=2, name="logits")

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=y, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(y, name="softmax_tensor")
  }
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  graph_location = FLAGS.model_dir
  train_writer = tf.summary.FileWriter(graph_location + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(graph_location + '/test')
  tf.global_variables_initializer().run()
  saver = tf.train.Saver()
  tf.train.write_graph(sess.graph_def, graph_location, "graph.pbtxt", True) #proto
  if FLAGS.restore != "" :
    saver.restore(sess, FLAGS.restore)
  # # Train the model, and also write summaries.
  # # Every 10th step, measure test-set accuracy, and write test summaries
  # # All other steps, run train_step on training data, & add training summaries

  # def feed_dict(train):
  #   """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  #   if train or FLAGS.fake_data:
  #     xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
  #     k = FLAGS.dropout
  #   else:
  #     xs, ys = mnist.test.images, mnist.test.labels
  #     k = 1.0
  #   return {x: xs, y_: ys, keep_prob: k}

  file_name = FLAGS.data_file
  batch_size = 100
  h5data = H5DataList(file_name, batch_size, slice(1))
  
  max_iter = FLAGS.max_iter + 1
  save_iter= FLAGS.save_iter
  info_iter= FLAGS.info_iter
  for i in range(FLAGS.max_iter):
    batch_x, batch_y = h5data.get_next_batch()
    batch_y = tf.one_hot(indices=batch_y, depth=2).eval()
    #print(batch_y.shape)
    batch_y = np.reshape(batch_y, (batch_y.shape[0], 2))
    if i % info_iter == 0:  # Record summaries and test-set accuracy
      #summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob : 1.0})
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_x, y_: batch_y, keep_prob : 0.5},
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob : 0.5})
        train_writer.add_summary(summary, i)
    if i % save_iter == 0:
      save_path = saver.save(sess, graph_location + "/"+"model"+str(i)+".ckpt")
      print("Model saved in file: %s" % save_path)
  train_writer.close()
  test_writer.close()


def main(_):
  # if tf.gfile.Exists(FLAGS.log_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.log_dir)
  # tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()  
  # parser.add_argument('--max_steps', type=int, default=1000,
  #                     help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
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
  # parser.add_argument(
  #     '--data_dir',
  #     type=str,
  #     default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
  #                          'tensorflow/mnist/input_data'),
  #     help='Directory for storing input data')
  # parser.add_argument(
  #     '--log_dir',
  #     type=str,
  #     default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
  #                          'tensorflow/mnist/logs/mnist_with_summaries'),
  #     help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

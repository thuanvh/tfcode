import argparse
import sys
import tempfile
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import h5py

FLAGS = None

import tensorflow.contrib.slim as slim

def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
  with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=None):
      net = squeeze(inputs, squeeze_depth)
      outputs = expand(net, expand_depth)
      return outputs

def squeeze(inputs, num_outputs):
  return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
  with tf.variable_scope('expand'):
    e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
    e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
  return tf.concat([e1x1, e3x3], 3)
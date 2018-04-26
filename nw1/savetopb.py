"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

# create a session
sess = tf.Session()
model_name = 'model.ckpt-5212'
# import best model
saver = tf.train.import_meta_graph(model_name+'.meta') # graph
saver.restore(sess, model_name) # variables

# get graph definition
gd = sess.graph.as_graph_def()

# fix batch norm nodes
for node in gd.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in range(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

# generate protobuf
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ["logits_set"])
tf.train.write_graph(converted_graph_def, './', 'model.pb', as_text=False)
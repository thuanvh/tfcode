# pylint: disable=g-bad-file-header
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
"""Utilities to remove unneeded nodes from a GraphDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

from google.protobuf import text_format

from tensorflow.core.framework import attr_value_pb2

from tensorflow.core.framework import graph_pb2

from tensorflow.core.framework import node_def_pb2


from tensorflow.python.framework import graph_util

from tensorflow.python.platform import gfile
import argparse
import sys


parser = argparse.ArgumentParser()
parser.register('type', 'bool', lambda v: v.lower() == 'true')
parser.add_argument(
  '--input_graph',
  type=str,
  default='',
  help='TensorFlow \'GraphDef\' file to load.')
parser.add_argument(
  '--input_binary',
  nargs='?',
  const=True,
  type='bool',
  default=True,
  help='Whether the input files are in binary format.')
	  
FLAGS, unparsed = parser.parse_known_args()

input_graph=FLAGS.input_graph
input_binary=FLAGS.input_binary

input_graph_def = graph_pb2.GraphDef()
mode = "rb" if input_binary else "r"
with gfile.FastGFile(input_graph, mode) as f:
  if input_binary:
    input_graph_def.ParseFromString(f.read())
  else:
    text_format.Merge(f.read(), input_graph_def)
nodelist = input_graph_def.node
for n in nodelist:
  print(n.name)
  
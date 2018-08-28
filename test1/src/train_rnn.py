
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import process
import numpy as np
from dataset_text_series import TextSeriesDataFileList
import sys
import os
import numpy as np
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class RNNTraining:
    def __init__(self, 
        learning_rate = 0.0001,
        num_steps = 1000,
        batch_size = 128,
        valid_size = 500,
        display_step = 100,
        save_iter = 100,
        datafile = "",
        train_valid_file = "",
        dictfile = "",        
        model_dir = "",
        restore_file = "",
        process_type = "train",
        predict_output="",
        series_len = 1000,
        num_hidden = 128,
        normalize_data = False,
        neurons = [256, 256]):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.display_step = display_step
        self.save_iter = save_iter
        self.datafile = datafile
        self.dictfile = dictfile
        self.model_dir = model_dir
        self.restore_file = restore_file
        self.process_type = process_type
        self.predict_output = predict_output
        self.neurons = neurons
        self.valid_size = valid_size
        self.series_len = series_len
        self.num_hidden = num_hidden
        self.normalize_data = normalize_data
        
    def run(self):
        wdict = process.open_dict(self.dictfile)        
        num_input = self.series_len 
        timesteps = 1         
        num_classes = 2

        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }


        def RNN(x, weights, biases):

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
            x = tf.unstack(x, timesteps, 1)

            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

            # Get lstm cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        logits = RNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        class_pred = tf.argmax(prediction, 1)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        graph_location = self.model_dir
        if self.process_type == "train":
            print('Saving graph to: %s' % graph_location)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(graph_location)
            train_writer.add_graph(tf.get_default_graph())

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)
            saver = tf.train.Saver()
            if self.restore_file != "" :
                saver.restore(sess, self.restore_file)

            if self.process_type == "train":
                tf.train.write_graph(sess.graph_def, graph_location, "graph.pbtxt", True) #proto
                textdata = TextSeriesDataFileList([self.datafile], wdict, self.batch_size, self.series_len, self.normalize_data)
                valid_x, valid_y = textdata.get_valid_set(self.valid_size)
                valid_y = tf.one_hot(valid_y, num_classes).eval()
                valid_x = valid_x.astype(np.float).reshape((valid_x.shape[0],1,self.series_len))

                for step in range(1, self.num_steps + 1):
                    batch_x, batch_y = textdata.get_next_batch() #mnist.train.next_batch(batch_size)
                    batch_y = tf.one_hot(batch_y, num_classes).eval()
                    batch_x = batch_x.astype(np.float).reshape((batch_x.shape[0],1,self.series_len))
                    #batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Reshape data to get 28 seq of 28 elements
                    #batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    if step % self.display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: valid_x,
                                                                            Y: valid_y})
                        print("Step " + str(step) + ", Minibatch Loss= " + \
                            "{:.4f}".format(loss) + ", Training Accuracy= " + \
                            "{:.3f}".format(acc))

                    if step % self.save_iter == 0:
                        save_path = saver.save(sess, graph_location + "/"+"model"+str(step)+".ckpt")
                        print("Model saved in file: %s" % save_path)

                print("Optimization Finished!")

            if self.process_type == "predict" :
                textdata = TextSeriesDataFileList([self.datafile], wdict, self.batch_size, self.series_len, self.normalize_data, False)
                f = open(self.predict_output, "w")
                while True:
                    batch_x, batch_y = textdata.get_next_batch() #mnist.train.next_batch(batch_size)
                    if batch_x is None or len(batch_x) == 0:
                        break
                    batch_x = batch_x.astype(np.float).reshape((batch_x.shape[0],1,self.series_len))
                    pred_ = sess.run(class_pred, feed_dict={X: batch_x})
                    print("Classification: ", pred_)
                    for i in pred_:
                        f.write(str(i) +"\n")
                f.close()

                
            if self.process_type == "eval" :
                textdata = TextSeriesDataFileList([self.datafile], wdict, self.batch_size, self.series_len, self.normalize_data, False)
                result = np.zeros((num_classes, num_classes))
                while True:
                    batch_x, batch_y = textdata.get_next_batch() #mnist.train.next_batch(batch_size)
                    if batch_x is None  or len(batch_x) == 0:
                        print("Exit")
                        break
                    yplus = batch_y
                    batch_y = tf.one_hot(batch_y, num_classes).eval()
                    batch_x = batch_x.astype(np.float).reshape((batch_x.shape[0],1,self.series_len))
                    pred_, acc_ = sess.run([class_pred, accuracy], feed_dict={X: batch_x, Y: batch_y})

                    print("Classification: ", len(pred_), pred_)
                    print("Testing Accuracy:", acc_)
                    result = process.compare_result(yplus, pred_, result)
                    print(result)
                print(result)
                print("Accuracy:",(result[0][0] + result[1][1]) / (result[0][0] + result[1][1] + result[0][1] + result[1][0]))
                
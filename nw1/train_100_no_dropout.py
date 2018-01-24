import numpy as np
import tensorflow as tf
import argparse
import sys
import cv2

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""  
  input_layer = tf.reshape(features["x"], [-1, 100, 100, 3], name="input")
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=20,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
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
  pool4_flat = tf.reshape(pool6, [-1, 1 * 1 * 120])
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
  #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  #logits = tf.layers.dense(inputs=dropout, units=32, name="logits")
  logits = tf.layers.dense(inputs=dense, units=32)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.reshape(logits, [-1, 32], name="output")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={"output":tf.estimator.export.RegressionOutput(logits)})
  #if mode == tf.estimator.ModeKeys.PREDICT:
  #  return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

  loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=logits)}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  data = np.load("data_100.npz")
  print(data.files)
  train_data = data["features"]
  train_labels = data["labels"]
  print(train_data.shape)
  print(train_labels.shape)
  assert train_data.shape[0] == train_labels.shape[0]
  eval_data = train_data[5:10]
  eval_labels = train_labels[5:10]
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"loss": "loss"}
  #logging_hook = tf.train.LoggingTensorHook(every_n_iter=50)
    
  if FLAGS.phase == "train" :
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        #hooks=[logging_hook]
        )

  if FLAGS.phase == "eval" :
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  
  if FLAGS.phase == "predict":
    imginput = cv2.imread("189_tf_input.jpg")
    outputsize = 100
    img2 = cv2.resize(imginput, (outputsize, outputsize)).astype(np.float32)        
    #output_img_pts(img2, pvec, outputfolder + "//" + str(i) + ".input.jpg")
    img2 *= 1/255.0
    img2 -= 0.5
    datainput = img2.reshape(1,outputsize,outputsize,3)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": datainput},        
        num_epochs=1,
        shuffle=False)
    predict_value = mnist_classifier.predict(input_fn=eval_input_fn)
    #for val in list(predict_value):
    #  print(val["probabilities"])
    pval = list(predict_value)[0]["probabilities"]
    print(pval)
    def output_img_pts(img, pts, name):
      drawing = img.copy()
      pts2 = pts
      pts2 = pts2.astype(int)
      for i in range(0,4):
          for j in range(0,4):
              #print(pts2[i])
              #print(pts2[(i+1)%4])
              cv2.line(drawing, (pts2[(i*4 + j) * 2],pts2[(i*4 + j) * 2 + 1]), 
                  (pts2[(i*4+((j+1)%4))*2], pts2[(i*4+((j+1)%4)) * 2 + 1]), (0,255,0))
      cv2.imwrite(name, drawing)    
    for i in range(0,16):
      pval[2*i] *= imginput.shape[1]
      pval[2*i+1] *= imginput.shape[0]
    pval = pval.astype(np.int)
    output_img_pts(imginput, pval, "tf_predict.jpg")
    

  if FLAGS.phase == "save":
    feature_spec = {'x':tf.FixedLenFeature(dtype=tf.float32,shape=[1,40,40,3])}
    def serving_input_receiver_fn():
      """An input receiver that expects a serialized tf.Example."""
      
      serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[1],name="input_example_tensor")
      receiver_tensors = {"examples": serialized_tf_example}
      features = tf.parse_example(serialized_tf_example, feature_spec)
      return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    export_dir = FLAGS.model_dir + "/export/"
    mnist_classifier.export_savedmodel(export_dir, serving_input_receiver_fn)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--phase",
      type=str,
      default="train",
      help="TensorFlow train phase.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="model6",
      help="TensorFlow train phase.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()




#features_placeholder = tf.placeholder(features.dtype, features.shape)
#labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

#dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
#dataset = ...
#iterator = dataset.make_initializable_iterator()

#sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                          labels_placeholder: labels})
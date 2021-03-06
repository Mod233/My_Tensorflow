from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS =None

def main():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10])) 
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  
  # loss function
  y_ = tf.placeholder(tf.float32, [None, 10])
  #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  # launch the model
  sess = tf.InteractiveSession()
  
  # initial the variables
  tf.global_variables_initializer().run()
  
  # train--run the training steps 1000 times
  # Using small batches of random data is called stochastic training,
  # in this case, stochastic gradient descent.
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  # judge the model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))      


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
   FLAGS, unparsed = parser.parse_known_args()
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
    

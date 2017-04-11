# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:03:15 2017

@author: Noah_Huang
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def weight_variable (shape):
    init = tf.truncated_normal (shape, stddev = 0.1)
    return tf.Variable (init)
    
def bias_variable (shape):
    init = tf.constant (0.1, shape = shape)
    return tf.Variable (init)
    
def conv2d(x, W, strideShape2d):
    return tf.nn.conv2d (x, W, strides=[1,strideShape2d[0],strideShape2d[1],1], padding = 'SAME')
    
def max_pool2d (x, filterShape2d, strideShape2d):
    return tf.nn.max_pool (x, ksize=[1,filterShape2d[0],filterShape2d[1],1], strides=[1,strideShape2d[0],strideShape2d[1],1], padding='SAME')


#Get image database
mnist_images = input_data.read_data_sets ("MNIST_data/",one_hot=True)
    
#Declare variable
learning_rate = 0.0001
batch_size = 50
train_epoch = 1000

num_of_image_pixel = 784
num_of_input = 1

num_of_layer1 = 32
shape_of_stride_conv1 = [1, 1]
shape_of_filter_conv1 = [5, 5]
shape_of_stride_pool1 = [2, 2]
shape_of_filter_pool1 = [2, 2]

num_of_layer2 = 64
shape_of_stride_conv2 = [1, 1]
shape_of_filter_conv2 = [5, 5]
shape_of_stride_pool2 = [2, 2]
shape_of_filter_pool2 = [2, 2]

number_of_den_output = 1024

num_of_CNN_output = 10

#Declare placeholder
x = tf.placeholder (tf.float32, [None, num_of_image_pixel])
labels = tf.placeholder (tf.float32, [None, num_of_CNN_output])
keep_prob = tf.placeholder (tf.float32)

#config input
CNN_input = tf.reshape (x, [-1, 28, 28, 1])

#convolution layer 1
W1 = weight_variable ([shape_of_filter_conv1[0], shape_of_filter_conv1[1], num_of_input, num_of_layer1])
b1 = bias_variable ([num_of_layer1])

conv_output1 = tf.nn.relu (conv2d(CNN_input, W1, shape_of_stride_conv1) + b1)
pool_output1 = max_pool2d (conv_output1, shape_of_filter_pool1, shape_of_stride_pool1)

#convolution layer 2
W2 = weight_variable ([shape_of_filter_conv2[0], shape_of_filter_conv2[1], num_of_layer1, num_of_layer2])
b2 = bias_variable ([num_of_layer2])

conv_output2 = tf.nn.relu (conv2d(pool_output1, W2, shape_of_stride_conv2) + b2)
pool_output2 = max_pool2d (conv_output2, shape_of_filter_pool2, shape_of_stride_pool2)

#Densely connected layer
image_size = [28 / 2 / 2, 28 / 2 / 2]
num_of_densely_input = int(num_of_layer2 * image_size[0] * image_size[1])

Wd = weight_variable ([num_of_densely_input, number_of_den_output])
bd = bias_variable ([number_of_den_output])

pool2_flat = tf.reshape (pool_output2, [-1, num_of_densely_input])
den_output = tf.nn.relu  (tf.matmul (pool2_flat, Wd) + bd)

#Dropout
den_drop_output = tf.nn.dropout (den_output, keep_prob)

#Readout layer
Wr = weight_variable ([number_of_den_output, num_of_CNN_output])
br = bias_variable ([num_of_CNN_output])
readout_output = tf.matmul (den_drop_output, Wr) + br

#Construct cost function & training model
cost_function = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=readout_output))
train = tf.train.AdamOptimizer (learning_rate).minimize (cost_function)

#Correct rate function
correct_prediction = tf.equal(tf.argmax(readout_output,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#create session
sess = tf.InteractiveSession ()

tf.global_variables_initializer ().run ()

#start train loop
for i in range (0, 10000):
    batchPixels, batchLabels = mnist_images.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval ({x:batchPixels, labels:batchLabels, keep_prob:1.0})
        print ("step:%d, training accuracy %g" %(i, train_accuracy))
    
    train.run ({x:batchPixels, labels:batchLabels, keep_prob:0.5})

print ("final accuracy %g" %(accuracy.eval ({x:mnist_images.test.images, labels:mnist_images.test.labels, keep_prob:1.0})))
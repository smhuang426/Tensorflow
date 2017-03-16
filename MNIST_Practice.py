# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:41:40 2017

@author: Noah_Huang
"""

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def ParseDigit (digitArray):
    for i in range (0,10):
        if (digitArray[i] == 1):
            return i
            
            
def MaxValueWithMatix (array):
    maxIndex = 0;
    for i in range (1,10):
        if (array[i] > array[maxIndex]):
            maxIndex = i
            
    return maxIndex
    
            
mnist_images = input_data.read_data_sets ("MNIST_data/",one_hot=True)

num_Of_trainData = 1000
num_Of_testData = 100

train_pixels, train_labels = mnist_images.train.next_batch(num_Of_trainData)
test_pixels, test_labels = mnist_images.train.next_batch(num_Of_testData)

print ("------Start training------")

W = tf.Variable (tf.random_uniform ([784, 10], -1.0, 1.0))
b = tf.Variable (tf.random_uniform ([10], -1.0, 1.0))

x = tf.placeholder (tf.float32, [None, 784])
labels = tf.placeholder (tf.float32, [None, 10])

output = tf.matmul (x, W) + b

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

train = tf.train.GradientDescentOptimizer (0.5).minimize (cost_function)

sess = tf.InteractiveSession ()

tf.global_variables_initializer ().run ()

for _ in range (0,1000):
    sess.run (train, {x: train_pixels, labels: train_labels})


print ("-----end of training------") 

testResult = sess.run (output, {x: test_pixels})

numOfCorrect = 0

for i in range (0, num_Of_testData):
    digit = ParseDigit (test_labels[i,:])
    predictDigit = MaxValueWithMatix (testResult[i,:])
    if (digit == predictDigit):
        numOfCorrect = numOfCorrect + 1
        print ("actual number:", digit," ,predict number:", predictDigit)
    else:
        print ("actual number:", digit," ,predict number:", predictDigit,"   **")
        image = np.reshape (test_pixels[i,:], [28,28])
        plt.imshow (image)
        plt.show ()

print ("accuracy rate:", numOfCorrect/num_Of_testData*100,"%")


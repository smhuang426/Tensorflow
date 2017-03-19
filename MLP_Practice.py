# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:08:37 2017

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

num_Of_input = 784
num_Of_Hidden1 = 256
num_Of_Hidden2 = 256
num_Of_Output = 10

trainPixel, trainLabels = mnist_images.train.next_batch(num_Of_trainData)
testPixel, testLabels = mnist_images.train.next_batch(num_Of_testData)

print ("Start training")

#Declare placeHolder
x = tf.placeholder (tf.float32, [None, num_Of_input])
y = tf.placeholder (tf.float32, [None, num_Of_Output])

#Hiden Layer 1
W1 = tf.Variable (tf.random_uniform ([num_Of_input, num_Of_Hidden1], -1.0, 1.0))
b1 = tf.Variable (tf.random_uniform ([num_Of_Hidden1], -1.0, 1.0))
hidden_output1 = tf.nn.sigmoid (tf.add (tf.matmul(x, W1) , b1))

#Hidden Layer2
W2 = tf.Variable (tf.random_uniform ([num_Of_Hidden1, num_Of_Hidden2], -1.0, 1.0))
b2 = tf.Variable (tf.random_uniform ([num_Of_Hidden2], -1.0, 1.0))
hidden_output2 = tf.nn.sigmoid (tf.add (tf.matmul(hidden_output1, W2), b2))

#output Layer
W3 = tf.Variable (tf.random_uniform ([num_Of_Hidden2, num_Of_Output], -1.0, 1.0))
b3 = tf.Variable (tf.random_uniform ([num_Of_Output], -1.0, 1.0))
output = tf.add (tf.matmul(hidden_output2, W3), b3)

#Cost function & training model
softmax_output = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
cost_function = tf.reduce_mean (softmax_output)

train = tf.train.AdamOptimizer (0.01).minimize (cost_function)
#train = tf.train.GradientDescentOptimizer (0.4).minimize (cost_function)

#Declare session
sess = tf.InteractiveSession ()

#Iniitalize variables
tf.global_variables_initializer ().run ()

for _ in range (0, 500):
    sess.run (train, {x:trainPixel, y:trainLabels})

print ("end of training")

#Varify
testResult = sess.run (output, {x: testPixel})

numOfCorrect = 0

for i in range (0, num_Of_testData):
    digit = ParseDigit (testLabels[i,:])
    predictDigit = MaxValueWithMatix (testResult[i,:])
    if (digit == predictDigit):
        numOfCorrect = numOfCorrect + 1
        print ("actual number:", digit," ,predict number:", predictDigit)
    else:
        print ("actual number:", digit," ,predict number:", predictDigit,"   **")
        image = np.reshape (testPixel[i,:], [28,28])
        plt.imshow (image)
        plt.show ()

print ("accuracy rate:", numOfCorrect/num_Of_testData*100,"%")
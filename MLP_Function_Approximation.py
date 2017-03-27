# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:07:18 2017

@author: Noah_Huang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def TargetFunction (points):
    return np.cos (points) + 0.1 * np.random.randn (*points.shape)

#Variable declare
num_of_input = 1
num_of_hidden1 = 10
num_of_output = 1

batch_size = 100
num_of_epochs = 1500

training_size = 900
validate_size = 100

#prepare training set and validation data
x = np.float32 (np.random.uniform(-2*math.pi, 2*math.pi, (1, training_size + validate_size))).T
np.random.shuffle (x)

x_training = x[0:training_size]
y_training = TargetFunction (x_training)

x_validate = x[training_size: training_size + validate_size]
y_validate = TargetFunction (x_validate)

#plot training & validation set
plt.figure (1)
plt.scatter (x_training, y_training, c='blue', label='train')
plt.scatter (x_validate, y_validate, c='red', label='validate')
plt.legend ()
plt.show ()


print ("-----start training-----")
#declare placeholder
data_input = tf.placeholder (tf.float32, [None, num_of_input])
desire_output = tf.placeholder (tf.float32, [None, num_of_output])

#hidden layer 1
W1 = tf.Variable (tf.random_uniform ([num_of_input, num_of_hidden1]))
b1 = tf.Variable (tf.random_uniform ([num_of_hidden1]))
hidden_output1 = tf.sigmoid (tf.add (tf.matmul (data_input, W1), b1))

#output layer
W2 = tf.Variable (tf.random_uniform ([num_of_hidden1, num_of_output]))
b2 = tf.Variable (tf.random_uniform ([num_of_output]))
ANN_output = tf.add (tf.matmul (hidden_output1, W2), b2)

#get cost function & create training model
#cost_function = tf.reduce_sum (0.5 * (desire_output - ANN_output) ** 2)
cost_function = tf.nn.l2_loss (desire_output - ANN_output)

train = tf.train.AdamOptimizer ().minimize (cost_function)

#assign session & initialize variables
sess = tf.InteractiveSession ()

tf.global_variables_initializer ().run ()

#training & validate
errors = []
for i in range (0, num_of_epochs):
    for start, end in zip (range (0, len(x_training), batch_size), range (batch_size, len(x_training), batch_size)):
        sess.run (train, {data_input:x_training[start:end], desire_output:y_training[start:end]})
    cost = sess.run (tf.nn.l2_loss (y_validate - ANN_output), {data_input:x_validate})
    errors.append (cost)
    if (i%100 == 0):
        print ("epoch: %d, cost: %g" % (i, cost))
    
#show plot
plt.plot (errors, label = 'MLP function approximation')
plt.xlabel ('epochs')
plt.ylabel ('error')
plt.legend ()
plt.show ()

#plot function curve
testX = np.float32 (np.random.uniform(-2*math.pi, 2*math.pi, (1, 500))).T
np.random.shuffle (x)
correctY = np.cos (testX)
ANN_Y = sess.run (ANN_output, {data_input:testX})

plt.scatter (testX, correctY, c='blue', label = 'expected output')
plt.scatter (testX, ANN_Y, c='red', label = 'ANN output')
plt.legend ()
plt.show ()
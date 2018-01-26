# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from data_utilities import reshape_all
from data_utilities import one_hot_encode
from data_utilities import open_raw_data
from data_utilities import rgb2gray


raw_train_data, raw_train_labels, raw_test_data, raw_test_labels = open_raw_data()


bw_train_data = rgb2gray(raw_train_data)
bw_test_data = rgb2gray(raw_test_data)


#THE WHOLE RESHAPED TRAINING AND DATA WILL BE A 4D-TENSOR OF SHAPE [number of images,width,height,number of channels (in this case 3)]
one_hot_train_labels = one_hot_encode(raw_train_labels)
one_hot_test_labels = one_hot_encode(raw_test_labels)


#FIRST LET'S DEFINE INITIALIZATION METHODS

# FOR THE SAKE OF RELU LET'S USE POSITIVE INITIALIZATION


#NOW WE DEFINE THE MODEL STRUCTURE AS SIMPLE CONVOLUTIONAL NETWORK WITH MAX-POOLING AND RELU FOR RGB

#VARIABLES
    
x = tf.placeholder(tf.float32, shape=[None,32*32])

true_y = tf.placeholder(tf.float32, shape=[None,10])

#THE CONVOLUTION USES TENSORFLOW'S OWN CONV_2D -FUNCTION, WHICH DICTATES OUR 
#SHAPE OF THE WEIGHT TENSOR TO [KERNEL HEIGHT, KERNEL WIDTH, NUMBER OF INPUTLAYERS, NUMBER OF OUTPUTLAYERS (features)]
    

W_init = tf.truncated_normal([5,5,1,16], stddev=0.1) 
W_conv = tf.Variable(W_init)

b_init = tf.truncated_normal([16], stddev=0.1)
b_conv = tf.Variable(b_init)

x_image = tf.reshape(x, [-1,32,32,1])

# ACTUAL CONVOLUTION OPERATION
conv_layer = tf.nn.conv2d(x_image,W_conv,strides=[1,1,1,1] ,padding='SAME')+b_conv
#ACTIVATION
conv_layer_output = tf.nn.relu(conv_layer)
# still [-1,1,32,32]

# POOLING
#we will use VALID for padding so each of our channel sizes will be reduced by four 
pool_layer = tf.nn.max_pool(conv_layer_output,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID')

#FLATTEN CHANNELS

# NOW A FULLY CONNECTED LAYER WITH 200 NEURONS FOR SIXTEEN 28*28 (32-4*32-4) SIZED CHANNELS

W_fc_init = tf.truncated_normal([28*28*16,200], stddev=0.1) 
W_fc = tf.Variable(W_fc_init)

b_fc_init = tf.constant(value=0.1, shape = [200])
b_fc = tf.Variable(b_fc_init)


flattened_pool_layer = tf.reshape(pool_layer,[-1,28*28*16])

fc_hidden_output = tf.nn.relu(tf.matmul(flattened_pool_layer,W_fc)+b_fc)

# FINAL FULLY-CONNECTED OUTPUT "VOTES"

W_out_init = tf.truncated_normal([200,10], stddev=0.1) 
W_out = tf.Variable(W_out_init)

b_out_init = tf.truncated_normal([10], stddev=0.1)
b_out = tf.Variable(b_out_init)

output = tf.matmul(fc_hidden_output,W_out)+b_out

# ERRORFUNCTION

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_y, logits=output))
train_step = tf.train.AdamOptimizer().minimize(error)

# INFO
correct_prediction = tf.equal(tf.argmax(true_y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# RUN

with tf.Session() as sess:
    writer = tf.summary.FileWriter('C:/Users/Ilmari Pohjola/.spyder-py3/Projects/Test/logs', sess.graph)
    initializer = tf.global_variables_initializer()
    #sess.run(b_conv,{x:bw_train_data[1:50,:],true_y:one_hot_train_labels[1:50,:]})
    sess.close()





#im = raw_train_data[nr,:].reshape(3,32,32)
#label = raw_train_labels[nr]

#bw = rgb2gray(im)



#plt.imshow(bw,cmap='Greys')
#print(label)


# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:52:31 2018

@author: Ilmari Pohjola
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], initializer=tf.truncated_normal_initializer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(my_int_variable))
#W = tf.Variable(np.zeros(2,2))
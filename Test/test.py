# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from data_utilities import load_dataset
from data_utilities import rgb2gray
from data_utilities import give_random_inserts
from data_utilities import minibatches

tf.reset_default_graph()

def build_net(x, weights, biases):
    conv_layer_1 = tf.nn.conv2d(x,weights['CW_1'],strides=[1,1,1,1] ,padding='SAME',name="conv_layer_1_mult")
    conv_layer_1 = tf.nn.bias_add(conv_layer_1,biases['cb_1'], name='conv_layer_1_bias_add')
    rectified_1 = tf.nn.relu(conv_layer_1,name="rectified_1")
    pooling_1 = tf.nn.max_pool(rectified_1,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID',name="pooling_1")
    
    conv_layer_2 = tf.nn.conv2d(pooling_1,weights['CW_2'],strides=[1,1,1,1] ,padding='SAME',name="conv_layer_2_mult")
    conv_layer_2 = tf.nn.bias_add(conv_layer_2,biases['cb_2'], name='conv_layer_2_bias_add')
    rectified_2 = tf.nn.relu(conv_layer_2,name="rectified_2")
    pooling_2 = tf.nn.max_pool(rectified_2,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID',name="pooling_2")
    
    conv_layer_3 = tf.nn.conv2d(pooling_2,weights['CW_3'],strides=[1,1,1,1] ,padding='SAME',name="conv_layer_2_mult")
    conv_layer_3 = tf.nn.bias_add(conv_layer_3,biases['cb_3'], name='conv_layer_3_bias_add')
    rectified_3 = tf.nn.relu(conv_layer_3,name="rectified_3")
    pooling_3 = tf.nn.max_pool(rectified_3,strides=[1,2,2,1],ksize=[1,2,2,1],padding='VALID',name="pooling_3")
    
    
    dim = pooling_3.get_shape().as_list()
    flattened = tf.reshape(pooling_3,[-1,dim[1]*dim[2]*dim[3]],name="flattening_before_fully_connected_layers")
    fc_hidden_output = tf.nn.relu(tf.matmul(flattened,weights['FW_1']),name="fully_connected_mult")
    fc_hidden_output = tf.nn.bias_add(fc_hidden_output,biases['fb_1'],name='fully_connected_biases')
    
    output = tf.matmul(fc_hidden_output,weights['OUT_W'],name='output_mult')
    output = tf.nn.bias_add(output,biases['out_b'], name='output_biases')
    
    softmax=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name="softmax")
    return output, softmax

X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(dirpath='C:/Users/Ilmari Pohjola/.spyder-py3/Projects/Test/cifar-10-batches-py')


#NOW WE DEFINE THE MODEL STRUCTURE AS SIMPLE CONVOLUTIONAL NETWORK WITH MAX-POOLING AND RELU FOR RGB

#VARIABLES
    
x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y = tf.placeholder(tf.float32, shape=[None,10])



#THE CONVOLUTION USES TENSORFLOW'S OWN CONV_2D -FUNCTION, WHICH DICTATES OUR 
#SHAPE OF THE WEIGHT TENSOR TO [KERNEL HEIGHT, KERNEL WIDTH, NUMBER OF INPUTLAYERS, NUMBER OF OUTPUTLAYERS (features)]
weights = {
        'CW_1': tf.Variable(tf.truncated_normal([5,5,3,90], stddev=0.1,mean=0),name="CW_1"),
        'CW_2': tf.Variable(tf.truncated_normal([5,5,90,45], stddev=0.1,mean=0),name="CW_2"),
        'CW_3': tf.Variable(tf.truncated_normal([4,4,45,30], stddev=0.1,mean=0),name="CW_3"),
        'FW_1': tf.Variable(tf.truncated_normal([30*4*4,30], stddev=0.1,mean=0) ,name="FW_1"),
        'OUT_W': tf.Variable(tf.truncated_normal([30,10], stddev=0.1,mean=0),name="OUT_W")
}

biases = {
        'cb_1': tf.Variable( tf.truncated_normal([90], stddev=0.1,mean=0),name="cb_1"),
        'cb_2': tf.Variable(tf.truncated_normal([45], stddev=0.1,mean=0),name="cb_2"),
        'cb_3': tf.Variable(tf.truncated_normal([30], stddev=0.1,mean=0),name="cb_3"),
        'fb_1': tf.Variable(tf.truncated_normal([30], stddev=0.1,mean=0),name="fb_1"),
        'out_b': tf.Variable(tf.truncated_normal([10], stddev=0.1,mean=0),name="out_b"),
}
 


output, prediction = build_net(x=x,weights=weights,biases=biases)

# BACK PROPAGATION WITH ADAM

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
train_step = tf.train.AdamOptimizer().minimize(error)

# INFO
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# RUN

batch_size = 500
num_epochs = 400

tf.summary.scalar('accuracy',accuracy)
tf.summary.scalar('error',error)

merged = tf.summary.merge_all()
ep=0
with tf.Session() as sess:
    writer = tf.summary.FileWriter('C:/Users/Ilmari Pohjola/.spyder-py3/Projects/Test/logs', sess.graph)
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    
    
    print("Untrained network with 1000 examples")
    print(sess.run(accuracy,{x:X_train[1:1000,:],y:y_train[1:1000,:]}))
    i=0
    for epoch in range(num_epochs):
        train_error = 0
        train_accuracy = 0
        train_batches = 0
        for batch in minibatches(X_train,y_train,batch_size):
            X_batch, y_batch = batch
            sess.run(train_step,{x:X_batch,y:y_batch})
            #err,acc = sess.run([error,accuracy],{x:X_batch,y:y_batch})
            if train_batches % 15 == 0:    
                summary = sess.run(merged,{x:X_batch,y:y_batch})
                writer.add_summary(summary,i)
                #valid_batch_x, valid_batch_y = give_random_inserts(batch_size, X_valid, y_valid)
                #summary = sess.run(merged,{x:valid_batch_x,y:valid_batch_y})
            i+=1
        ep+=1
        print("Epoch %s done." % ep)
            #train_batches+=1
            #train_error+=err
            #train_accuracy+=acc
        
    print('Trained network with 1000 train_examples.')
    print(sess.run(accuracy,{x:X_train[1:1000,:],y:y_train[1:1000,:]}))
    print('Trained network with 1000 test examples.')
    print(sess.run(accuracy,{x:X_test[1:1000,:],y:y_test[1:1000,:]}))
    sess.close()





#im = raw_train_data[nr,:].reshape(3,32,32)
#label = raw_train_labels[nr]

#bw = rgb2gray(im)



#plt.imshow(bw,cmap='Greys')
#print(label)


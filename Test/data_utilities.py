# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:14:38 2018

@author: Ilmari Pohjola
"""

import numpy as np
import glob
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def rgb2gray(rgb):
    r = rgb[:,0:1024]
    g = rgb[:,1024:2048]
    b = rgb[:,2048:3072]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def load_dataset(dirpath='<dataset path of CIFAR10 for python>'):# give path as example "/home/username/folder name"
    X, y = [], []
    # take data from the data batch
    for path in glob.glob('%s/data_batch_*' % dirpath):
        with open(path, 'rb') as f:
            batch = pickle.load(f,encoding='bytes')#,encoding='latin1' (if gives error of encoding)
        # append all data and labels from the 5 data betch
        X.append(batch[b'data'])
        y.append(batch[b'labels'])
    # devide by 255 for making value 0 to 1
    X = np.concatenate(X) /np.float32(255)
    # making labels as int
    y = np.concatenate(y).astype(np.int32)
    #seperate in to RGB colors
    X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))
    # reshape data into 4D tensor with compatible to CNN model
    X = X.reshape((X.shape[0], 32, 32, 3))
    # initialize labels for training ,validation and testing 
    Y_train = np.zeros((40000,10),dtype = np.float32)
    Y_valid = np.zeros((10000,10), dtype = np.float32)
    y_test = np.zeros((10000,10),dtype = np.int32)
    
    # divide 40000 as training data and it's labels
    X_train = X[-40000:]
    y_train = y[-40000:]
    #devide 10000 as validation data and it's labelss
    X_valid = X[:-40000]
    y_valid = y[:-40000]
    
    # make training labels compatables with CNN model
    for i in range(40000):
        a = y_train[i]
        Y_train[i,a] = 1

    # make validation labels compatables with CNN model
    for i in range(10000):
        a = y_valid[i]
        Y_valid[i,a] = 1
    
    # load test set
    path = '%s/test_batch' % dirpath
    with open(path, 'rb') as f:
        batch = pickle.load(f,encoding='bytes')#,encoding='latin1'
    X_test = batch[b'data'] /np.float32(255)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
    y_t = np.array(batch[b'labels'], dtype=np.int32)
    # make test labels compatables with CNN model
    for i in range(10000):
        a = y_t[i]
        y_test[i,a] = 1

    # normalize to zero mean and unity variance
    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    X_test = (X_test - offset) / scale
    return X_train, Y_train, X_valid, Y_valid, X_test, y_test

def give_random_inserts(k,x,y):
    n = x.shape[0]
    indexes=np.random.choice(n, k, replace=False)
    return x[indexes,:], y[indexes,:]
    
def minibatches(ims, labels, batchsize):
    assert len(ims) == len(labels)
    indices = np.arange(len(ims))
    np.random.shuffle(indices)
    for start_idx in range(0, len(ims) - batchsize + 1, batchsize):       
        excerpt = indices[start_idx:start_idx + batchsize]
    yield ims[excerpt], labels[excerpt]

  
    
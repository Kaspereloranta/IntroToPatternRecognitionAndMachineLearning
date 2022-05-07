'''
Course: Data.ml.100 Introduction to pattern recognition and machine learning
Exercise 4

Student name: Kasper Eloranta
Student ID: H274212
E-mail: kasper.eloranta@tuni.fi
'''

import numpy
import pickle
import numpy as np
import statistics
import matplotlib.pyplot as plt
from random import random, randint
from skimage.transform import resize, downscale_local_mean
from scipy.stats import norm
from math import sqrt
from scipy.stats import multivariate_normal
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def load_cifar10():
    datadictBatch1 = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/data_batch_1')
    datadictBatch2 = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/data_batch_2')
    datadictBatch3 = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/data_batch_3')
    datadictBatch4 = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/data_batch_4')
    datadictBatch5 = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/data_batch_5')
    datadictTest = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/test_batch')
    X = datadictTest["data"].astype("int")
    Y = datadictTest["labels"]
    Xtr1 = datadictBatch1["data"].astype("int")
    Ytr1 = datadictBatch1["labels"]
    Xtr2 = datadictBatch2["data"].astype("int")
    Ytr2 = datadictBatch2["labels"]
    Xtr3 = datadictBatch3["data"].astype("int")
    Ytr3 = datadictBatch3["labels"]
    Xtr4 = datadictBatch4["data"].astype("int")
    Ytr4 = datadictBatch4["labels"]
    Xtr5 = datadictBatch5["data"].astype("int")
    Ytr5 = datadictBatch5["labels"]
    trainingdata = np.concatenate((Xtr1, Xtr2, Xtr3, Xtr4, Xtr5), axis=0)
    traininglabels = np.concatenate((Ytr1, Ytr2, Ytr3, Ytr4, Ytr5), axis=0)
    labeldict = unpickle('/Users/K-A-S/anaconda3/envs/dataml100/EX03/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]
    return X, Y, trainingdata, traininglabels

# Task 4.1 Convert class numbers to one-hot vectors
def convertClassNumbers(Y):
    Yc = np.zeros((len(Y),10))
    for index, value in enumerate(Y):
        Yc[index][value] = 1
    return Yc

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def main():

    X, Y, trainingdata, traininglabels = load_cifar10()

    # To convert classes to 1x10 vectors.
    Y = convertClassNumbers(Y)
    traininglabels = convertClassNumbers(traininglabels)

    # And data to image format and normalizing it afterwads
    X = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")
    trainingdata = trainingdata.reshape(len(trainingdata), 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")
    trainingdata, X = trainingdata/255, X/255

    # Let's create first simple neural network with no convolutional layers.

    nnModel = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid'),
    ])

    # To compile and train the neural network with all Cifar-10 training data.
    nnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nnModel.fit(trainingdata,traininglabels,epochs=10,verbose=2)

    # And a neural network with convolutional layers to compare its accuracy to previous one.

    cnnModel = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape = (32,32,3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.5, noise_shape = None, seed=None),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'sigmoid')
    ])

    # To compile and train the neural network with all Cifar-10 training data.
    cnnModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    cnnModel.fit(trainingdata, traininglabels, epochs=10, verbose=2)

    nn_test_loss, nn_test_acc = nnModel.evaluate(X,Y,verbose=1)
    print(nnModel.summary())

    cnn_test_loss, cnn_test_acc = cnnModel.evaluate(X,Y,verbose=1)
    print(cnnModel.summary())

    print("Simple neural network classifier's accuracy by using CIFAR-10 test samples and 10 epochs:", nn_test_acc)
    print("Neural network classifier's accuracy with convolutional layers by using CIFAR-10 test samples and 10 epochs:", cnn_test_acc)

main()

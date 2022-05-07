'''
Course: Data.ml.100 Introduction to pattern recognition and machine learning
Exercise 3

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

# CIFAR-10 – Evaluation
def class_acc(pred,gt):
    correctlyClassified = 0
    j = 0
    while j < len(pred):
         if pred[j] == gt[j]:
            correctlyClassified += 1
         j += 1
    accuracy = correctlyClassified / len(gt)
    return accuracy

# Task 3.1  Cifar-10-Bayesian classifier
def cifar10_color(X):
    Xp = np.zeros((len(X),3))
    for i,n in enumerate(X):
        img_1x1 = resize(X[i],(1,1))
        r_val = img_1x1[:, :, 0].reshape(1 * 1)
        g_val = img_1x1[:, :, 1].reshape(1 * 1)
        b_val = img_1x1[:, :, 2].reshape(1 * 1)
        mu_r = r_val.mean()
        mu_g = g_val.mean()
        mu_b = b_val.mean()
        Xp[i] = (mu_r,mu_g,mu_b)
    return Xp

# Task 3.1  Cifar-10-Bayesian classifier
def cifar_10_naivebayes_learn(Xp,Y):
    p = np.zeros((10, 1))
    mu = np.zeros((10, 3))
    sigma = np.zeros((10, 3))
    for i in range(0,10):
        classValues = np.zeros((5000,3))  # Assuming that images are evenly distributed to classes ( as they are
        classImagesFound = 0              # in Cifar-10 training data, 5000 images of each class.
        for j in range(0,len(Y)):
            if Y[j] == i:
                classValues[classImagesFound] = Xp[j]
                classImagesFound += 1
        mu[i] = (classValues[:,0].mean(),classValues[:,1].mean(),classValues[:,2].mean())
        sigma[i] = (classValues[:,0].std(),classValues[:,1].std(),classValues[:,2].std())
        p[i] = classImagesFound / len(Y)
    return mu, sigma, p

# Task 3.1  Cifar-10-Bayesian classifier
def cifar10_classifier_naivebayes(x,mu,sigma,p):
    Y = []
    for img in x:
        imgClassProbs = []
        for j in range(0,10):
            imgClassProbs.append(norm.pdf(img[0], mu[j, 0], sigma[j, 0]) *
                                 norm.pdf(img[1],mu[j, 1], sigma[j, 1]) *
                                 norm.pdf(img[2], mu[j, 2], sigma[j, 2]) * p[j])
        highestProb = max(imgClassProbs)
        Y.append(imgClassProbs.index(highestProb))
    return Y

# Task 3.2 Cifar-10 - Bayesian classifier (better)
def cifar_10_bayes_learn(Xf, Y):
    Yarray = np.array(Y)
    mu = np.zeros((10, Xf.shape[1]))
    sigma = np.zeros((10, Xf.shape[1], Xf.shape[1]))
    p = np.zeros((10, 1))
    for i in range(10):
        bool = Yarray == i
        mean_values = Xf[bool].mean(axis=0)
        cov_variance = np.cov(Xf[bool], rowvar=False)
        mu[i, :] = mean_values
        sigma[i, :] = cov_variance
        p[i] = bool.sum() / len(Y)
    return mu, sigma, p

# Task 3.2 Cifar-10 - Bayesian classifier (better)
def cifar10_classifier_bayes(x, mu, sigma, p):
    probabilities = np.zeros((10, 10000))
    for i in range(10):
        if mu.shape[1] < 192:
            probability = multivariate_normal.pdf(x, mu[i], sigma[i]) * p[i]
            probabilities[i] = probability
        else:
            probability = multivariate_normal.logpdf(x, mu[i], sigma[i]) * p[i]
            probabilities[i] = probability
    return np.argmax(probabilities, axis=0)

# Task 3.3 Cifar-10 - Bayesian classifier (best)
def cifar10_NxN_color(X,n):
    dimension = int(32 / n)
    Xmean = downscale_local_mean(X, (1, dimension, dimension, 1)).reshape((len(X), 3 * n ** 2))
    return Xmean

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def main():
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

    # To convert the data to image format.
    X = X.reshape(len(X),3,32,32).transpose(0,2,3,1).astype("int32")
    trainingdata = trainingdata.reshape(len(trainingdata),3,32,32).transpose(0,2,3,1).astype("int32")

    # Task 3.1 Cifar-10-Bayesian classifier, naive, using 1X1 images for training and testing.
    Xtr1X1 = cifar10_color(trainingdata)
    Xtest1X1 = cifar10_color(X)
    Y = np.array(Y)
    traininglabels = np.array(traininglabels)

    mu,sigma,p = cifar_10_naivebayes_learn(Xtr1X1,traininglabels)
    YbayesNaive1X1 = cifar10_classifier_naivebayes(Xtest1X1,mu,sigma,p)
    naiveacc = class_acc(YbayesNaive1X1,Y)
    print("Naive Bayes classifier's accuracy for 1X1 images is",naiveacc)

    # Task 3.2 Cifar-10-Bayesian classifier, better, using 1X1 images for training and testing.
    mu, sigma, p = cifar_10_bayes_learn(Xtr1X1, traininglabels)
    Ybayes1X1 = cifar10_classifier_bayes(Xtest1X1, mu, sigma, p)
    acc1x1 = class_acc(Ybayes1X1,Y)
    print("Better Bayes classifier's accuracy for 1X1 images is:",acc1x1)

    # Task 3.3 Cifar-10 Bayesian classifier (best), 2x2, 4x4, .. , 32x32 images.

    # Using 2X2 images for training and testing.
    Xtr2X2 = cifar10_NxN_color(trainingdata,2)
    Xtest2X2 = cifar10_NxN_color(X,2)
    mu, sigma, p = cifar_10_bayes_learn(Xtr2X2, traininglabels)
    Ybayes2X2 = cifar10_classifier_bayes(Xtest2X2, mu, sigma, p)
    acc2x2 = class_acc(Ybayes2X2,Y)
    print("Better Bayes classifier's accuracy for 2X2 images is:",acc2x2)

    # Using 4X4 images for training and testing.
    Xtr4X4 = cifar10_NxN_color(trainingdata, 4)
    Xtest4X4 = cifar10_NxN_color(X, 4)
    mu, sigma, p = cifar_10_bayes_learn(Xtr4X4, traininglabels)
    Ybayes4X4 = cifar10_classifier_bayes(Xtest4X4, mu, sigma, p)
    acc4x4 = class_acc(Ybayes4X4,Y)
    print("Better Bayes classifier's accuracy for 4X4 images is:",acc4x4)

    # Using 8X8 images for training and testing.
    Xtr8X8 = cifar10_NxN_color(trainingdata, 8)
    Xtest8X8 = cifar10_NxN_color(X, 8)
    mu, sigma, p = cifar_10_bayes_learn(Xtr8X8, traininglabels)
    Ybayes8X8 = cifar10_classifier_bayes(Xtest8X8, mu, sigma, p)
    acc8x8 = class_acc(Ybayes8X8,Y)
    print("Better Bayes classifier's accuracy for 8X8 images is:",acc8x8)

    # Using 16X16 images for training and testing.
    Xtr16X16 = cifar10_NxN_color(trainingdata, 16)
    Xtest16X16 = cifar10_NxN_color(X, 16)
    mu, sigma, p = cifar_10_bayes_learn(Xtr16X16, traininglabels)
    Ybayes16X16 = cifar10_classifier_bayes(Xtest16X16, mu, sigma, p)
    acc16X16 = class_acc(Ybayes16X16,Y)
    print("Better Bayes classifier's accuracy for 16X16 images is:",acc16X16)

    # Using 32X32 images for training and testing.
    Xtr32X32 = cifar10_NxN_color(trainingdata, 32)
    Xtest32X32 = cifar10_NxN_color(X, 32)
    mu, sigma, p = cifar_10_bayes_learn(Xtr32X32, traininglabels)
    Ybayes32X32 = cifar10_classifier_bayes(Xtest32X32, mu, sigma, p)
    acc32X32 = class_acc(Ybayes32X32,Y)
    print("Better Bayes classifier's accuracy for 32x32 images is:",acc32X32)

    # Plot the results
    Accuracies = [naiveacc,acc1x1,acc2x2,acc4x4,acc8x8,acc16X16,acc32X32]
    Methods = ["Naive Bayes 1X1","Bayes 1X1", "Bayes 2X2", "Bayes 4X4", "Bayes 8X8", "Bayes 16X16", "Bayes 32X32"]
    plt.figure(1);
    plt.plot(Methods,Accuracies, color='green', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='red', markersize=10)
    plt.title("Classification accuracies by using different Bayesian methods and by using different size of CIFAR-10 images")
    plt.show()
main()

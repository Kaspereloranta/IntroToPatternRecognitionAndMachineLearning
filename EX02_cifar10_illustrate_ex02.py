import numpy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint

# Task 2 CIFAR-10 – Evaluation
def class_acc(pred,gt):
    correctlyClassified = 0
    j = 0
    while j < len(pred):
         if pred[j] == gt[j]:
            correctlyClassified += 1
         j += 1
    accuracy = correctlyClassified / len(gt)
    return accuracy

# Task 3 CIFAR-10 – Random classifier
def cifar10_classifier_random(x):
    return randint(0,9)

#Task4  CIFAR-10 – 1-NN classifier
def cifar10_classifier_1nn(x,trdata,trlabels):
    shortestDistance = 0
    classification = 0
    j = 0
    for i in trdata:
        distance = np.linalg.norm(i-x)
        if j == 0:
            shortestDistance = distance
            classification = trlabels[j]
        elif distance < shortestDistance:
            shortestDistance = distance
            classification = trlabels[j]
        j += 1
    return classification

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def main():
    datadictBatch1 = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/data_batch_1')
    datadictBatch2 = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/data_batch_2')
    datadictBatch3 = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/data_batch_3')
    datadictBatch4 = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/data_batch_4')
    datadictBatch5 = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/data_batch_5')
    datadictTest = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/test_batch')
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
    labeldict = unpickle('/Users/K-A-S/OneDrive/Työpöytä/Koulu/Johdatus hahmontunnistukseen ja koneoppimiseen/Harkat/EX02/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    # Task2; To test that class_acc works properly.
    print("Testing the class_acc, 1.0 should be returned here: ", class_acc(Y,Y))

    # Task3; Script to generate a random class labels for all the CIFAR-10 test samples
    # and to evaluate its accuracy.
    randomY = []
    for i in X:
        randomY.append(cifar10_classifier_random(i))
    print("Random classifier accuracy by using all CIFAR-10 test samples:", class_acc(randomY,Y))

    # Task4; 1-NN Classifier, A script to input all CIFAR-10 test samples to NN-classifier with all the training data
    # and to evaluate its accuracy
    NNclassifiedYTest = []
    #for i in X:
        #NNclassifiedYTest.append(cifar10_classifier_1nn(i,trainingdata,traininglabels))
    print("1-NN accuracy with all the training data:", class_acc(NNclassifiedYTest,Y))

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)

    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
main()
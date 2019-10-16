# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:30:07 2019

@author: anspa
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

--------NORMAILZATION AND PCA----------

def get_transformed_data():
    print("Reading in and transforming data...")

    if not os.path.exists('large_files/train.csv'):
        print('Looking for ../large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('large_files/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    Y = data[:, 0].astype(np.int32)

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:]
    Ytest  = Y[-1000:]

    # center the data
    mu = Xtrain.mean(axis=0)
    Xtrain = Xtrain - mu
    Xtest  = Xtest - mu

    # transform the data
    pca = PCA()
    Ztrain = pca.fit_transform(Xtrain)
    Ztest  = pca.transform(Xtest)

    plot_cumulative_variance(pca)

    # take first 300 cols of Z
    Ztrain = Ztrain[:, :300]
    Ztest = Ztest[:, :300]

    # normalize Z
    mu = Ztrain.mean(axis=0)
    std = Ztrain.std(axis=0)
    Ztrain = (Ztrain - mu) / std
    Ztest = (Ztest - mu) / std
    return Ztrain, Ztest, Ytrain, Ytest


def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P


--------NORMAILZATION----------


def get_normalized_data():
    print("Reading in and transforming data...")

    if not os.path.exists('large_files/train.csv'):
        print('Looking for ../large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('large_files/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:]
    Ytest  = Y[-1000:]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    
    return Xtrain, Xtest, Ytrain, Ytest


-------------------------




def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind







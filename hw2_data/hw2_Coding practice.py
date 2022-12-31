# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 17:40:00 2022

@author: river
"""

import numpy as np
test = np.arange(12).reshape(3,4)
vector = np.arange(4)

print(test[1], test.shape, len(test), vector + test, '\n\n', vector * test)
x_mean = np.mean(test, 0) #x_mean = np.mean(test, 1) => (3,) cannot be operated with test, (3, 4)
print('\n\n\n', x_mean)
print(test-x_mean.reshape(1, -1))
# matrix minus array, highest-d len() must be the same (最裡面的框框個數需與array個數相同)

      

# * / dot / matmul difference
x = np.arange(8).reshape(1, 8)
y = np.arange(16).reshape(8, 2)

Dot = np.dot(x, y)
Matmul = np.matmul(x, y)
print('~~~~~~~~~~~~\n', x, '\n~~~~~~~\n', y, '\n~~~~~~~~\n', Dot, '\n\n', Matmul)
print('\n\n\n', x.reshape(8), x.reshape(8).shape, '\n\n', y.T, y.T.shape, '\n\n', x.reshape(8) * y.T)

#shape維度長度 & len長度同樣寫法
test_shape = np.arange(8)
test_shape2 = np.arange(8).reshape(8, 1)
print(test_shape.shape, test_shape2.shape)



import pandas as pd
X_train_pd = pd.read_csv('./X_train')
X_train_pd = X_train_pd.iloc[:, 1:]
X_train = X_train_pd.to_numpy(dtype = float)
Y_train_pd = pd.read_csv('./Y_train')
Y_train_pd = Y_train_pd.iloc[:, 1:]
Y_train = Y_train_pd.to_numpy(dtype = float)
Y_train = Y_train.reshape(len(Y_train),)

def normalize(X):
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis = 0)
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i, j] = (X[i, j] - X_mean[j]) / ( X_std[j] + 1e-8)
    return X

X_train = normalize(X_train)

w = np.zeros(X_train.shape[1])
b = np.zeros((1, ))

def sigmoid(z):
    '''Sigmoid function'''
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-(1e-8))

def f(X, w, b):
    '''Logistic regression function
    Arguements:
     X: input data, shape = [batch_size, data_dimension]
     w: weight vector, shape = [data_dimension, ]
     b: bias, scalar
    '''
    return sigmoid(np.matmul(X, w) + b)

def gradient(X, Y_label, w, b):
    '''Computes gradient of cross entropy loss'''
    Y_pred = f(X, w, b)
    print(Y_pred.shape)
    pred_error = Y_label - Y_pred
    print(pred_error.shape)
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return Y_pred, w_grad, b_grad
Y_pred, w, b = gradient(X_train, Y_train, w, b)

def accuracy(Y_pred, Y_label):
    '''Returns accuracy %'''
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

test_pred = np.round(f(X_train, w, b))
acc = accuracy(test_pred, Y_train)

pd.DataFrame(test_pred).to_csv('./test_pred.csv')
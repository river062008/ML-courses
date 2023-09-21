# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 09:33:31 2022

@author: river
"""

import numpy as np
import pandas as pd
data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

#reshaping month_data in dictionary
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, 24 * day: 24 * (day + 1)] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

#print(month_data[0].shape)
#print(month_data[0])
#help(month_data[0])
#test = pd.DataFrame(month_data[0])
#test.to_csv('C:\\Users\\river\\Downloads\\hw1_data\\test.csv')

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471 + day*24 + hour, :] = month_data[month][:, day*24 + hour:day*24 + hour + 9].reshape(1, -1)
            y[month*471 + day*24 + hour, 0] = month_data[month][9, day*24 + hour + 9]
print(x)
print(y)

mean_x = np.mean(x, axis = 0)
std_x = np.std(x, axis = 0)
for i in range(len(x)): #471*12
    for j in range(len(x[0])): #9*18
        x[i][j] = (x[i][j]-mean_x[j]) / std_x[j]

print(x)

import math
x_train_set = x[:math.floor(len(x)*0.8), :]
y_train_set = y[:math.floor(len(y)*0.8), 0]
x_validation_set = x[math.floor(len(x)*0.8):, :]
y_validation_set = y[math.floor(len(y)*0.8):, 0]
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation_set))
print(len(y_validation_set))


dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([471 * 12, 1]), x), axis = 1)
learning_rate = 100
iter_time = 2000
adagrad = np.zeros([dim, 1])
eps = 0.000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum((np.dot(x, w) - y) ** 2) / (471 * 12))
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * (np.dot(x.transpose(), (np.dot(x, w)-y)))
    adagrad += gradient ** 2
    w = w - learning_rate*(gradient / np.sqrt(adagrad + eps))

np.save('weight.npy', w)
w

test_data = pd.read_csv('./test.csv', header = None, encoding = 'big5')

test_data = test_data.iloc[:, 2:]
test_data[test_data=='NR'] = 0
np_test_data = test_data.to_numpy()
test_x = np.empty([240, 9*18])
for i in range(240):
    test_x[i, :] = np_test_data[18*i:18*(i+1), :].reshape(1, -1)
print(test_x, test_x.shape, mean_x.shape, std_x.shape, w.shape)


for i in range(240):
    for j in range(9*18):
        test_x[i, j] = (test_x[i, j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
                                            
ans_y = np.dot(test_x, w)
print(ans_y)
        

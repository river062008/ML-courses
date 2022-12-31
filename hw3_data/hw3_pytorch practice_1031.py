# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 09:33:31 2022

@author: river
"""

import numpy as np
import pandas as pd
data = pd.read_csv('./train.csv', encoding = 'big5')
print(data, '\n', 'Hello World!')

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

import torch
import torch.nn as nn
x_torch = torch.from_numpy(x)
x_torch = torch.tensor(x_torch, dtype=torch.float)
y_torch = torch.from_numpy(y)
y_torch = torch.tensor(y_torch, dtype=torch.float)

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([471 * 12, 1]), x), axis = 1)
learning_rate = 100
iter_time = 2000
adagrad = np.zeros([dim, 1])
eps = 0.000000001
loss_record = []
for t in range(iter_time):
    loss = np.sqrt(np.sum((np.dot(x, w) - y) ** 2) / (471 * 12))
    if(t%100==0):
        print(str(t) + ":" + str(loss))
        loss_record.append(loss)
    gradient = 2 * (np.dot(x.transpose(), (np.dot(x, w)-y)))
    adagrad += gradient ** 2
    w = w - learning_rate*(gradient / np.sqrt(adagrad + eps))


input_size = 18*9
output_size = 1

# we can call this model with samples X


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)


# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 2000

loss_torch = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_record_torch = []

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(x_torch)

    # loss
    l = loss_torch(y_torch, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 100 == 0:
        loss_record_torch.append(l)
        print('epoch ', epoch+1, ' loss = ', l)


    
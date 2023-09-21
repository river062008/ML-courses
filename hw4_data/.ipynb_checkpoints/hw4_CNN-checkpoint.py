# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:53:34 2022

@author: river
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

#Read image function
def readfile(path, label):
#    '''return np array x & y'''
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

#Read data
workspace_dir = 'C:/Users/river/Desktop/Data science/Python/ML-courses/hw4_data/food-11'
print('Read data')
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print('Size of traing data: {}'.format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print('Size of validation data: {}'.format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, 'testing'), False)
print('Size of testing data: {}'.format(len(test_x)))

#Augmentation / Data transforming
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])   

#Dataloader / Dataset
class ImgDataset():
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 16
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

#Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# torch.nn.MaxPool2d(kernel_size, stride, padding)
# input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), #[64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[64, 64, 64]
            
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[128, 32, 32]
            
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[256, 16, 16]
            
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[512, 4, 4]
            )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
            )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


#Tranining
model = Classifier().cuda()
loss = nn.CrossEntropyLoss() #classification task
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Optimizer by Adam
num_epoch = 30

for epoch in range(num_epoch):
    start_time = time.time()
    train_acc = 0
    train_loss = 0
    val_acc = 0
    val_loss = 0
    
    model.train() #train mode on, be able to dropout neurons
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() #Dump gradients in model
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda()) #pred & label must be either GPU or CPU
        batch_loss.backward()
        optimizer.step()
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1)==data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
            
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
             (epoch + 1, num_epoch, time.time()-start_time, \
              train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
#        print('[{}/{}], time:{}sec(s), train_acc:{}, loss:{}; val_acc:{}, loss:{}'.format(epoch+1, num_epoch, \
#        time.time()-start_time, train_acc/train_set.__len__(), train_loss/train_set.__len__(), \
#        val_acc/val_set.__len__(), val_loss/val_set.__len__())
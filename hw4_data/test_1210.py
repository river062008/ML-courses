# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:04:14 2022

@author: river
"""
import time
testlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
num = 0
for i in testlist:
    globals()[i] = num
    num = num + 1
    print(i)
print(testlist)

start_time = time.time()
print('a/b={}, a*b={}, [{}/{}]'.format(a/b, a*b, c, d))
print('[{}/{}], time:{}sec(s), train_acc:{}, loss:{}; val_acc:{}, loss:{}'.format(a+1, b, time.time()-start_time, c/d, e/f, 
g/h, i/j))
     

for i in testlist:
    print(type(i))
    
for i in testlist:
    globals()[i] = int(i)
    print(type(i))
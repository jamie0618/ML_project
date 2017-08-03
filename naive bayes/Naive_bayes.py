# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:09:14 2017

@author: melon
"""

import numpy as np
import random
import csv
from sklearn.naive_bayes import GaussianNB

file_name = "pima-indians-diabetes.data.txt"

def LoadData(filename, test_ratio=0.2, seed=0):
    random.seed(seed)
    train_data = []
    test_data = []
    test_label = []
    with open(filename,'r') as f:
        lines = list(csv.reader(f))
        data_size = len(lines)
        test_index = random.sample(range(0,data_size),int(test_ratio*data_size))        
        for index,row in enumerate(lines):
            data = [ float(v) for v in row]
            if index in test_index:
                test_data.append(data[:-1])
                test_label.append(data[-1])
            else:
                train_data.append(data)
    return train_data, test_data, test_label 

def SeperateByClass(dataset):
    data_seperated = {}
    for data in dataset:
        label = data[-1]
        feature = data[:-1]
        if label not in data_seperated:
            data_seperated[label] = []
        data_seperated[label].append(feature)
    return data_seperated

def MeanAndStd(data_seperated):
    mean_std = {}
    for key, value in data_seperated.items():
        mean_array = np.mean(value,axis=0)
        std_array = np.std(value,axis=0)
        mean_std[key] = []
        mean_std[key].append(mean_array)
        mean_std[key].append(std_array)
    return mean_std

def Gaussian(x, mean, std):
    prob = 1
    for i in range(len(x)):
        exponent = np.exp( -( (x[i]-mean[i])**2 / (2 * std[i]**2 )))
        exponent = (1 / ( (2*np.pi)**0.5 * std[i]) ) * exponent
        prob *= exponent        
    return prob

def Predict(x, mean_std):
    best_prob = -1
    label = None
    for key, value in mean_std.items():
        prob = Gaussian(x, value[0], value[1])
        prob = np.prod(prob)
        if prob > best_prob:
            best_prob = prob
            label = key
    return label

def PredictAllData(dataset, mean_std):
    predict_array = []
    for x in dataset:
        predict = Predict(x, mean_std)
        predict_array.append(predict)
    return predict_array

def Accuracy(y, predict):
    num_data = len(y)
    count = 0
    for i in range(num_data):
        if y[i] == predict[i]:
            count +=1 
    return count/num_data    
    
print("read data...")
train_data, test_data, test_label = LoadData(file_name)
print("total {0} training data, {1} testing data".format(len(train_data),len(test_data)))
data_seperated = SeperateByClass(train_data)
mean_std = MeanAndStd(data_seperated)
predict_array = PredictAllData(test_data, mean_std)
acc = Accuracy(test_label, predict_array)
print("Acc = {0}".format(acc))

x_train = []
y_train = []
for data in train_data:
    x_train.append(data[:-1])
    y_train.append(data[-1])
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(test_data)
acc = Accuracy(test_label, y_pred)
print("Acc using scikit-learn = {0}".format(acc))
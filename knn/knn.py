# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:23:23 2017

@author: melon

k-nearest neighbor example
"""

import csv
import random
import operator 

def LoadData(filename, test_ratio=0.1, seed=0):
    random.seed(seed)
    train_data = []
    test_data = []
    with open(filename,'r') as f:
        lines = list(csv.reader(f))[:-1] 
        data_size = len(lines)
        test_index = random.sample(range(0,data_size),int(test_ratio*data_size))        
        for index,row in enumerate(lines):
            if index in test_index:
                test_data.append(row)
            else:
                train_data.append(row)
    return train_data, test_data

def EuclideanDistance(data1, data2):
    feature_len = len(data1)-1
    distance = 0
    for i in range(feature_len):
        distance += (float(data1[i]) - float(data2[i]))**2
    distance = distance**0.5
    return distance

def FindNeighbors(train_set, test, k):
    distance_array = []
    for train in train_set:
        dist = EuclideanDistance(train,test)
        distance_array.append([train,dist])
    distance_array.sort(key=operator.itemgetter(1))
    neighbors = distance_array[:k]
    return neighbors
 
def Predict(neighbors):
    neighbors_class = {}
    for n in neighbors:
        label = n[0][-1]
        if label in neighbors_class:
            neighbors_class[label] += 1
        else:
            neighbors_class[label] = 1
    predict = max(neighbors_class, key=neighbors_class.get)
    return predict

def Accuracy(predict,test_set):
    correct = 0
    for i,v in enumerate(predict):
        if v == test_set[i][-1]:
            correct += 1
    return correct/len(predict)

   
train_set, test_set = LoadData("iris.data.txt", test_ratio=0.33)
k = 5
predict_array = []
for data in test_set:
    neighbors = FindNeighbors(train_set, data, k)
    answer = Predict(neighbors)
    predict_array.append(answer)
acc = Accuracy(predict_array,test_set)
print("k = {0}, Accuracy = {1}".format(k,acc))








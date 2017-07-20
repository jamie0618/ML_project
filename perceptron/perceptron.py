# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:58:17 2017

@author: melon

perceptron learning example
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class Perceptron():

    def __init__(self, max_iter = 1000):
        self.max_iter = max_iter
        self.w = []
        self.b = 0
        self.num_data = 0
        self.num_features = 0
    
    def train(self, X, Y):
        self.num_data, self.num_features = np.shape(X)
        self.w = np.zeros(self.num_features)
        for i in range(self.max_iter):
            update = False
            for j in range(self.num_data):
                y_pred = self.b + np.dot(self.w, X[j])
                if np.sign(y_pred) != np.sign(Y[j]):
                    update = True
                    self.w += Y[j] * X[j]
                    self.b += Y[j]
            if not update:
                print("Converged in {0} iterations".format(i))
                break
 
    def classify_instance(self, x):
        if len(self.w) == 0 :
            self.w = np.zeros(len(x))
        ans = self.b + np.dot(self.w, x)
        return 1 if ans >= 0 else -1
            
    def classify(self, X):
        Y_pred = []
        for i in range(np.shape(X)[0]):
            y_pred = self.classify_instance(X[i])
            Y_pred.append(y_pred)
        return Y_pred
        
def GenerateData(num_points,seed=0):
    random.seed(seed)
    X = np.zeros((num_points, 2))
    Y = np.zeros(num_points)
    for i in range(num_points):
        X[i][0] = random.randint(1,9)+0.1*random.randint(1,9)
        X[i][1] = random.randint(1,9)+0.1*random.randint(1,9)
        Y[i] = 1 if X[i][0]+X[i][1] >= 10 else -1
    return X, Y
            
def PlotData(X,Y,title):        
    for i,v in enumerate(X):
        if Y[i] == 1:
            plt.plot(v[0],v[1],marker='o')
        else:
            plt.plot(v[0],v[1],marker='x')
    plt.xlim(0,10)     
    plt.title(title)
    plt.show()              
    
def ErrorRate(Y,Y_pred):
    error = 0
    for i in range(len(Y)):
        if Y[i] != Y_pred[i]:
            error += 1
    return error/len(Y)
        
X_train,Y_train = GenerateData(100,seed=0)        
PlotData(X_train,Y_train,"training data")        

model = Perceptron()
Y_init = model.classify(X_train)
print("initial error rate = {0}".format(ErrorRate(Y_train,Y_init)))
model.train(X_train,Y_train)
Y_final = model.classify(X_train)
print("final error rate = {0}".format(ErrorRate(Y_train,Y_final)))
print("final b = {0}, final w = {1},{2}".format(model.b,model.w[0],model.w[1]))
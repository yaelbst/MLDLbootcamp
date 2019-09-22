#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:48:11 2019

@author: yael
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class DataSet:
#instanciation D is dataset with all features for all instances and L labels    
    def __init__(self, D, L):
        self.D = D
        self.L = L
        
    def prepare_train_test(self, percent):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=percent, shuffle=True)
        return X_train, Y_train, X_test, Y_test
    
class KNN:
    
    def __init__(self, k, X_train, X_test,  Y_train, Y_test):
        self.k = k
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def classify_test_data(self, k, X_train, X_test,  Y_train):
        predictions = []
        for i in range(len(X_test)):
            knear = self.find_k_nearest(k, X_train, X_test[i, :])
            y_pred = self.predict(knear, Y_train, X_test[i, :])
            predictions.append(y_pred)
        return predictions
        
    def compute_distance(self, instance1, instance2, features_number):
        distance = 0
        for i in range(features_number):
            distance += np.sum(np.square(instance1[i] - instance2[i]))  
        return np.sqrt(distance)

    def find_k_nearest(self, k, X_train, test_instance):
        distances = []
        features_number = len(test_instance)-1
        for i in range(len(X_train)):
            dist = self.compute_distance(test_instance, X_train[i], features_number)
            distances.append((dist, i))
#    print(dist)
        distances.sort(key=lambda tup: tup[0]) 
        print(distances)
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][1])
        return neighbors

    def most_common(self,lst):
        return max(set(lst), key=lst.count)

    def predict(self, knear, Y_train, test_instance):
        vote = []
        print("len: {}".format(len(knear)))
        print(Y_train[knear[0]])
        for i in range(len(knear)):
            vote.append(Y_train[knear[i]])
        print(vote)
        print(self.most_common(vote))
        y_pred = self.most_common(vote)
        return y_pred
    
data = load_iris()

X = data.data
print(len(X))
print('Shape: {}'.format(X.shape))
Y = data.target

dat = DataSet(X,Y)
X_train, Y_train, X_test, Y_test =  dat.prepare_train_test(0.3)

knn = KNN(7,X_train, Y_train, X_test, Y_test)
predictions = knn.classify_test_data(7, X_train, X_test,  Y_train)

def accuracy(predictions, Y_test):
    count = 0
    for i in range(len(X_test)):
        if (predictions[i]==Y_test[i]):
            count += 1
    accuracy = count / (len(X_test))
    print(accuracy)
    return accuracy

accuracy(predictions, Y_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:54:40 2019

@author: yael
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1.
#We are going along the instructions from the following link:
#http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# Handle Data: Load the data from CSV file and split it into training and test datasets.
 
Data = pd.read_csv("pima-indians-diabetes.csv", sep=",", engine="python", header=None)
print(Data.head())

##Divide the data into 80% for train n and 20% test 
def splitDataset(Data, splitRatio):
    print("Data {}".format(len(Data)))
    n = int(splitRatio * len(Data))
    Data_train = Data[Data.index < n]
    print(len(Data_train))
    Data_test = Data[Data.index >= n]
    print(len(Data_test))
    return Data_train, Data_test

Data_train, Data_test = splitDataset(Data, 0.8)

#2. Summarize Data (train): summarize the properties in the training dataset by
#calculate for every feature and class (prediction value) the mean and the std.

print(Data_train.groupby(Data_train.columns[8])[Data_train.columns[0]].mean())

Train0 = Data_train.loc[Data_train[8] == 0]
Train1 = Data_train.loc[Data_train[8] == 1]
sample_test = np.array(Data_test)[1,:]
Test = np.array(Data_test)
Y_Test = Test[:,-1]

T0 = np.array(Train0)
m0 = np.mean(T0[:,0])
print(m0)

T1 = np.array(Train1)

print(T0.shape[1])

def summarize_data(T0,T1):
    m = []
    m1 = []
    st = []
    st1 = []
    for i in range(T0.shape[1]-1):
        m.append(np.mean(T0[:,i]))
        st.append(np.std(T0[:,i]))
    for i in range(T1.shape[1]-1):
        m1.append(np.mean(T1[:,i]))
        st1.append(np.std(T1[:,i]))
    S = np.array((m,st,m1,st1))
    return S

S = summarize_data(T0, T1)
print(S)

#3.Write a function which makes a prediction: Use the summaries of the dataset to
#generate a single prediction, which based on the gaussian distribution with the
#corresponding mean and std of each of the features. You can find the equation for
#the probability of an event given a Gaussian distribution in:
#https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes

#gaussian distribution for one feature and one class
#p(x = v/Ci)
def probability_featx_v_forCk(v,m,st):
    p = (1/np.sqrt(2*np.pi*np.square(st)))*np.exp(-np.square(v-m)/(2*np.square(st)))
    return p

print(sample_test)

print(S[0,0])
print(S[1,0])

p = probability_featx_v_forCk(11,S[0,0],S[1,0])
print("p {}".format(p))

a = probability_featx_v_forCk(71.5,73,6.2)
print("a {}".format(a))

#p(Ck): probability of one class over the entire dataset
def compute_probability_of_characteristic(nberw, nbert):
    return nberw/nbert


nber_total = len(Data_train)
nber_healthy = len(T0)
print(nber_healthy)
print(nber_total)
pH = compute_probability_of_characteristic(nber_healthy,nber_total)
print(pH)



nber_sick = len(T1)
print(nber_healthy)
pS = compute_probability_of_characteristic(nber_sick,nber_total)
print(pS)

print(pH + pS)
#P the probability of the sample_test person to be healthy

#P = p(x1=v1|C=0)*p(x2=v2|C=0)*...*p(x7=v7|C=0)*p(C=0)
#Here P(C=0) is pH probability of being healthy
#The feature probabilities are given by the function(formula) probability_featx_v_forCk

print(len(sample_test))


p_feat = []
for i in range(len(sample_test)-1):
#    print(probability_featx_v_forCk(sample_test[i],S[0,i],S[1,i]))
    p_feat.append(probability_featx_v_forCk(sample_test[i],S[0,i],S[1,i]))
    
p_feat_sick = []
for i in range(len(sample_test)-1):
    p_feat_sick.append(probability_featx_v_forCk(sample_test[i],S[2,i],S[3,i]))
    


#print(p_feat_sick)

print(p_feat[0]*p_feat[1]*p_feat[2]*p_feat[3]*p_feat[4]*p_feat[5]*p_feat[6]*p_feat[7])
pfe = p_feat[0]*p_feat[1]*p_feat[2]*p_feat[3]*p_feat[4]*p_feat[5]*p_feat[6]*p_feat[7]
print(pfe * pH)

print("Probability of the sample person to be sick")
print(p_feat_sick[0]*p_feat_sick[1]*p_feat_sick[2]*p_feat_sick[3]*p_feat_sick[4]*p_feat_sick[5]*p_feat_sick[6]*p_feat_sick[7]*pS)


product = 1
for i in range(len(sample_test)-1):
    print(p_feat[i])
    product *= p_feat[i]
#print(product)

print("Whole dataset --------------")
#4.Make Predictions: Generate predictions on the whole test dataset.
y_pred = []
for pers in range(len(Test)):
    
    p_feat = []
    product = 1
    for i in range(len(Test[pers,:])-1):
        #print(probability_featx_v_forCk(sample_test[i],S[0,i],S[1,i]))
        p_feat.append(probability_featx_v_forCk(Test[pers,i],S[0,i],S[1,i]))
#    for j in range(len(Test[pers,:])-1):
#        print(p_feat[i])
        product *= p_feat[i]
    pHpers = product * pH
    print("proba of pers to be healthy {}".format(pHpers))
    p_feat_sick = []
    prods = 1
    for i in range(len(Test[pers,:])-1):
        p_feat_sick.append(probability_featx_v_forCk(Test[pers,i],S[2,i],S[3,i]))
#    for j in range(len(Test[pers,:])-1):
#        print(p_feat_sick[i])
        prods *= p_feat_sick[i]
    pSpers = prods * pS
    print("proba of pers to be sick {}".format(pSpers))
    if pHpers > pSpers:
        y_pred.append(0)
    else:
        y_pred.append(1)
print("Prediction {}".format(y_pred))

def accuracy(predictions, Y_test):
    count = 0
    for i in range(len(Y_test)):
        if (predictions[i]==Y_test[i]):
            count += 1
    accuracy = count / (len(Y_test))
    print("prediction accuracy {}".format(accuracy))
    return accuracy

accuracy(y_pred, Y_Test)

    
    

    



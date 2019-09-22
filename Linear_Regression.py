#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:31:49 2019

@author: yael
"""
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')

#1.1

X = np.array([[31,22],[22,21],[40,37],[26,25]])
Y = np.array([2,3,8,12])

#I write θ1 ,θ2 in the following lines t1, t2
#such as h(x1,x2) = t1x1 +t2x2
#We know that for the best t1,t2, the derivative of the loss function (loss value function of t1,t2)
#is 0, and it comes down to XtXbestt = XtY, which comes down to bestt= inv(XtX).(XtY)

Xt = np.matrix.transpose(X)
print(Xt)

XtX = np.dot(Xt,X)
print(XtX)

XtXinv = np.linalg.inv(XtX)
print(XtXinv)

XtY = np.dot(Xt,Y)
print(XtY)

bestt = np.dot(XtXinv,XtY) 
print(bestt)

#loss calculation
besttT = np.transpose(bestt)
Y_pred =  X @ besttT
loss = np.sum((Y_pred - Y) ** 2) / Y_pred.shape[0]
print("loss calculation")
print(loss)
#1.2
# Add a vector (colummn) of ones to X such as x3, i=1 ∀ i
# It will introduce a bias and a new parameter t3
# It will influence the output, but it is not related to X

#Now we have
#h(x1,x2,x3)=t1x1 + t2x2 + t3
#which is
#h(x1,x2,x3)=t1x1 + t2x2 + t3x3

#it is only a trick so we learn a single matrix of parameters
#but it has an influence on the fitting, because without it, the line will go through the origin
#and may be far from the data.
#With the loss calculation as a check, i see it does give a lower loss, but the difference is very small

newX = np.insert(X, 2, 1, axis=1)
print(newX)
print(newX.shape)

#Now we have one more parameter: t1, t2, t3
newXt = np.matrix.transpose(newX)
print(newXt)

newXtnewX = np.dot(newXt,newX)
print(newXtnewX)

newXtnewXinv = np.linalg.inv(newXtnewX)
print(newXtnewXinv)

newXtY = np.dot(newXt,Y)
print(newXtY)

newbestt = np.dot(newXtnewXinv,newXtY) 
print(newbestt)

newbesttT = np.transpose(newbestt)
new_Y_pred =  newX @ newbesttT
newloss = np.sum((new_Y_pred - Y) ** 2) / new_Y_pred.shape[0]
print("new loss calculation")
print(newloss)

#2
#Test your code from the previous question on a big data and show the results on a graph with matplotlib. 
#To install matplotlib type on cmd/terminal – python -m pip install -U matplotlib

#2.1 
#Load the data from the file “data_for_linear_regression.csv”. (you can use pandas - read_csv)
import pandas

#This returns a Pandas DataFrame
Data = pandas.read_csv("data_for_linear_regression.csv", sep=",").dropna()
#print(Data.shape)
#print(Data.dtype)

#2.2
#We want to convert it to a numpy array
D = Data.values

print(D[0])
#Why is there such a big space between the 2 columns in the printout?!!
#However the shape is right

#2.3
#Show the data on the graph, use matplotlib.pyplot to show the data with scatter plot. 
#(read about matplotlib in this link - https://matplotlib.org/gallery/shapes_and_collections/scatter.html)
X = D[:,[0]]
#print(X)
Y = D[:,[1]]
#print(Y)
fig = plt.figure()
ax = plt.axes()
plt.scatter(X,Y)
plt.show()

#2.4
#Now, take some data and try to use the solution from the previous question to find the trend line of these points. 
#Remember because you want to try to find equation like this form: y = ax + b, 
#So you need to add bias, add b as one's vector to x with numpy.hstack.
SX = X[0:100]
SY = Y[0:100]
print(SX.shape)
print(SY.shape)

#plt.scatter(SX,SY)
#plt.show()

#Add an all-one vector
V = np.ones(100)
all_one = np.reshape(V,(100,1))
print(V.shape)

M = np.hstack((SX, all_one))
print(M.shape)

#M transpose
MT = np.matrix.transpose(M)
print(MT.shape)

#Apply equation to find t and b, the bias
T = np.linalg.inv(MT.dot(M)).dot(MT).dot(SY)
print(T)

#loss calculation
SY_pred =  M @ T
loss = np.sum((SY_pred - SY) ** 2) / SY_pred.shape[0]
print("Sample data loss calculation")
print(loss)

#2.5
#Draw the line y=xt + b
#Question: how do i know which is the bias?
#Obviously if i try to switch i see the line doesn't fit at all so i know which is which,
#but i would've liked to be sure how to differentiate between them

#It says to use matplotlib.pyplot.hold
#But the documentation says hold is deprecated since matplotlib 2.0

x = np.linspace(0, 100, 1000)
plt.scatter(SX,SY)
#plt.hold(true) if it wasn't deprecated to have the line on the same graph
plt.plot(x, x*T[0] + T[1], 'r',linestyle='--') # dashed
plt.show()

#2.6
x = np.linspace(0, 100, 1000)
plt.scatter(X,Y)
#plt.hold(true) if it wasn't deprecated to have the line on the same graph
plt.plot(x, x*T[0] + T[1], 'r',linestyle='--') # dashed
plt.show()

#I see it fine so i dont need xlim(0,100) and ylim(0,100)
#but just for the record they are matlab functions to limit the graph:
#xlim(limits) sets the x-axis limits for the current axes or chart. 
#Specify limits as a two-element vector of the form [xmin xmax], where xmax is greater than xmin.

#Just to compare
#calculation of the tetas, Loss to see the difference with more data
Vfd = np.ones(X.shape[0])
all_one_fd = np.reshape(Vfd,(X.shape[0],1))
print(Vfd.shape)

Mfd = np.hstack((X, all_one_fd))
print(Mfd.shape)

#M transpose
MfdT = np.matrix.transpose(Mfd)
print(MfdT.shape)

#Apply equation to find t and b, the bias
P = np.linalg.inv(MfdT.dot(Mfd)).dot(MfdT).dot(Y)
print(P)

#loss calculation, L2
Y_pred =  Mfd @ P
full_data_loss = np.sum((Y_pred - Y) ** 2) / Y_pred.shape[0]
print("Full data loss calculation")
print(full_data_loss)

#loss calculation, L1, just to compare
L1_loss = np.sum(np.abs(Y_pred - Y))/ Y_pred.shape[0]
print("L1 loss calculation")
print(L1_loss)
#Performs much better in this case!

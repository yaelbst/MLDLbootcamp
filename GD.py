#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:53:14 2019

@author: yael
"""
import numpy as np
from matplotlib import pyplot as plt

x = np.array([0,1,2]).reshape(3,1)
Y = np.array([1,3,7]).reshape(3,1)

Ones = np.ones(x.shape[0]).reshape(3,1)
Squared = x**2
X = np.hstack((Ones,x, Squared))
print(X)

θ = np.array([2,2,0]).reshape(3,1)

def gradient_descent(X, Y, theta, lr, epochs):
    loss_vect = []
    plt.figure()
    plt.axes()
    Y_pred = np.dot(X, theta).reshape((3,1))
    for i in range(epochs):
      gradient = np.dot(X,Y_pred - Y)
      theta = theta - lr * gradient
#     print("iteration : {}".format(i))
#     print("θ: {}".format(theta))
      Y_pred = np.dot(X, theta)
      mis_square = (Y_pred - Y)**2
      L = np.sum(mis_square)/3
      loss_vect.append(L)
#     print("Loss: {}".format(L))
    plt.plot(loss_vect)
    plt.show()
    return theta

def compute_loss(Y_pred, Y, m):
    L = np.sum((Y_pred - Y)**2)/(m*2)
    return L

pred = np.dot(X, θ).reshape((3,1))
starting_loss = compute_loss(pred, Y, 3)
print("starting loss: {}".format(starting_loss))

final_θ = gradient_descent(X, Y, θ, 0.1, 200)
final_pred = np.dot(X, final_θ).reshape((3,1))
final_loss = compute_loss(final_pred, Y, 3)

f_θ = gradient_descent(X, Y, θ, 0.01, 100)
f_pred = np.dot(X, f_θ).reshape((3,1))
f_loss = compute_loss(f_pred, Y, 3)
print("GD - lr 0.01 f loss: {}".format(f_loss))

def gradient_descent_momentum(X, Y, theta, mu, lr, epochs):
    loss_v = []
    plt.figure()
    plt.axes()
    Y_pred = np.dot(X, theta).reshape((3,1))
    for iter in range(epochs):
#θj :=θj +α􏰁y(i) −hθ(x(i))􏰂x(i)
       if iter == 0:
          v = np.zeros(theta.shape)
       gradient = np.dot(X,Y_pred - Y)
       v = mu*v - lr*gradient
#       print(v)
       theta = theta + v
#      print("iteration : {}".format(iter))
#      print("θ: {}".format(θ))
       Y_pred = np.dot(X,theta)
       l = (Y_pred-Y)**2
       Loss = (np.sum(l))/(3*2)
       loss_v.append(Loss)
#      print("Loss: {}".format(L))
    plt.plot(loss_v)
    plt.show()
    return theta

fin_θ = gradient_descent_momentum(X, Y, θ, 0.9, 0.01, 100)
fin_pred = np.dot(X, fin_θ).reshape((3,1))
fin_loss = compute_loss(fin_pred, Y, 3)
print("GDM - lr 0.01 fin loss: {}".format(fin_loss))
#gradient_descent_momentum(X, Y, θ, 0.9, 0.1, 100)

#Repeat the process using LR=0.01, but this time with Nesterov accelerated gradient γ = 0.9
def gradient_descent_nesterov(X, Y, theta, mu, lr, epochs):
    loss_v = []
    plt.figure()
    plt.axes()
    Y_pred = np.dot(X, theta).reshape((3,1))
    for iter in range(epochs):
#θj :=θj +α􏰁y(i) −hθ(x(i))􏰂x(i)
       if iter == 0:
          v = np.zeros(theta.shape)
       t = theta - mu*v
       Y_pred = np.dot(X, t).reshape((3,1))
       gradient = np.dot(X,Y_pred - Y)
       v = mu*v - lr*gradient
#       print(v)
       theta = theta + v
#      print("iteration : {}".format(iter))
#      print("θ: {}".format(θ))
       Y_pred = np.dot(X,theta)
       l = (Y_pred-Y)**2
       Loss = (np.sum(l))/(3*2)
       loss_v.append(Loss)
#      print("Loss: {}".format(L))
    plt.plot(loss_v)
    plt.show()
    return theta

last_θ = gradient_descent_nesterov(X, Y, θ, 0.9, 0.01,100)
last_pred = np.dot(X, last_θ).reshape((3,1))
last_loss = compute_loss(last_pred, Y, 3)
print("GDN - lr 0.01 fin loss: {}".format(last_loss))
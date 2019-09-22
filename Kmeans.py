#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:22:02 2019

@author: yael
"""

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from collections import OrderedDict


data = load_iris()

X = data.data
print(len(X))
print('Shape: {}'.format(X.shape))


#means: will i calculate random values in the range of each?


#dictionnary example:
#dict_square = {x: x*x for x in range(1,101)}

#clusters = {mu: list of points}
#clusters is the dictionnary
#clusters[mu] is the list of points
#For every point we calculate its distance to each of the k chosen means and assign it to the closest

#Just to start the exercise, i picked random starting values but within the dataset values
means = np.array([(4,3,1.5,0.2),(5.5,3.5,1.8,1),(7,3,5,2)])
print(means)

origins = means[:,0:2]

class Kmeans:
    
    def __init__(self, X, means):
        self.X = X
        self.means = means

    def cluster(self, X, means):
        k = len(means)
        clusters = OrderedDict({0:[],1:[],2:[]} )
        for x in range(len(X)):
            distances=[]
            for mu in range(k):
#            print(mu)
#            print(X[x])
#            print(means[mu])
#            print(np.linalg.norm(X[x]-means[mu]))
                distances.append(np.linalg.norm(X[x]-means[mu]))
#        print("The closest centroid is")
#        print(np.argmin(distances))
#k lists of points
            try:
                clusters[np.argmin(distances)].append(X[x])
            except KeyError:
                clusters[np.argmin(distances)] = [X[x]]
#    clusters.keys.sort()
        return clusters

#clusters = cluster(X,means)
#
#print(clusters)

    def update_means(self, clusters, dim):
        centroids = np.ones((3,dim))
        for key, value in clusters.items():
#        print(value)
#        print(key, len(value))
            centroids[key] = np.mean(value, axis=0)
        return centroids

#def update_centroids(clusters):
#    for i in range(len(clusters.keys())):
#        print("update---------")
#        print(i)
#        means[i]= np.mean(clusters[i], axis = 0)
#    return means

#m = update_means(clusters)

#cent = update_centroids(clusters)

#print(m)

#print(cent)

    def has_converged(self, means_old, means_new):
        if np.all(means_old == means_new):
            return True
        else:
            return False
    
    def compare_means(self, means_old, means_new):
        return np.linalg.norm(means_old, means_new)

#print(means)
#print(has_converged(means, m))
    def show_clusters(self, clusters, means, Xg):
#    plt.plot(means[:,0], means[:,1],'ro')
        i = 0
        color = ['g','b','y']
        for key,value in clusters.items():
            group = np.asarray(clusters[key])
#        print(group)
#        print(color[i])
            plt.scatter(group[:,0],group[:,1], c=color[i])
            plt.plot(means[i,0], means[i,1],color[i]+'^')
            i += 1

    def find_clusters(self, X, means, max_iter, dim):
        means_previous = np.ones((3,dim))
        iter = 0
        if iter < max_iter:
            while self.has_converged(means_previous, means)!= True:
                iter += 1
                means_previous = means
                clusters = self.cluster(X, means)
                means = self.update_means(clusters, dim)
                if (dim == 2):
                    self.show_clusters(clusters, means, X)
#            plt.show()
#            print("Iteration {}: ".format(iter))
#            print("Iteration {}: means diff: {}".format(iter, compare_means(means_previous, means)))
        return clusters, means

kmeans = Kmeans(X, means)
groups, centroids = kmeans.find_clusters(X, means, 100, 4)

print(groups)
print(centroids)

#To see the representation, Xgraph with only 2 columns
Xg = X[:, 0:2]
print(Xg.shape)

print(origins)
clus = kmeans.cluster(Xg, origins)
grapes, mus = kmeans.find_clusters(Xg, origins, 100, 2)
#print(clus[0])
#clus1 = np.asarray(clus[0])
#print(clus1)

#plt.scatter(clus1[:,0], clus1[:,1])
#plt.show()

#Centroids are represented as triangles
#Data points are coloured according to their cluster

#    plt.show()    
#    plt.scatter(Xg[:,0], Xg[:,1])

#show_clusters(clus, origins, Xg)
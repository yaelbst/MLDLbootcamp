#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:22:08 2019

@author: yael
"""

import numpy as np
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import math


data = load_iris()

X = data.data

print(len(X))
print('Shape: {}'.format(X.shape))

#A function to calculate the distance between 2 points
def dist(point_a, point_b):
   return math.sqrt(np.power(point_a-point_b,2).sum())
#    return np.linalg.norm(P-Q)

def in_eps_neighborhood(point_a,point_b,eps):
#    print(dist(P,Q))
    return dist(point_a,point_b)< eps

P= np.array([5.8, 2.7, 5.1, 1.9]) 

#Q = np.array([5.4,3.9,1.7,0.4])
#
#Z = np.array([4.8,3,1.4,0.1])

#
#W = np.array([6.3,2.5,5,1.9])
#
#A = [P,Q,Z, W]
#
#print(A)
#B = []
#for i in range(len(A)):
#  if A[i] in [Z,W]:
#     print(A[i])
#print(B)    

eps = 1

#print(in_eps_neighborhood(P,Q,eps))
#
#print(in_eps_neighborhood(P,W,eps))

#A function to calculate how many neighbours are in an epsilon envirement of a point (​regionQuery)​
def region_query(all_points, point, eps):
    count = 0
    neighbors = []
    for i in range(0,len(all_points)):
        if in_eps_neighborhood(point,all_points[i,:],eps) == True:
            count += 1
            neighbors.append(all_points[i,:])
    return count, neighbors
            
#count, neighbors = region_query(X, P, eps)
#print(count, neighbors)
#cluster = []
#A function that begins in a core point and expands it till it can’t be expanded anymore (​expandCluster)
#Since it's a core point no need to check the number of neighbors, it's possible to start a cluster
def expand_cluster(core_point, n, neighbors, cluster, all_points, classification, eps, cluster_id, min_points, count):
    
#    neighbors_list = []
    
#    noise = []
#    print("core point")
#    print(core_point)
    print(count)
    if count == 0:
        cluster.append(core_point)
        print(cluster)
        idx = np.where(np.all(X==core_point,axis=1))
        print("core_point cluster_id {}".format(cluster_id))
        classification[0][idx[0][0]]= cluster_id
        print(classification[0][idx[0][0]])
#    print(all_points)
#    n, neighbors = region_query(all_points, core_point, eps)
#    print(n, neighbors)
    for i in range(len(neighbors)):
        ind = np.where(np.all(X==neighbors[i],axis=1))
#        print(ind[0][0])
#        print(classification[0][ind[0][0]])
        if classification[0][ind[0][0]]==0 or classification[0][ind[0][0]] == -1:
#            print("expanding cluster neighbor")
#            print(neighbors[i])
            cluster.append(neighbors[i])
            classification[0][ind[0][0]]= cluster_id
#            For each one the cluster has to be continued if possible
#            print(classification[0][ind[0][0]])
            m, group = region_query(all_points, neighbors[i], eps)
#        print(m)
            if m > min_points:
                count += 1
#                print("Hi! {}".format(neighbors[i]))
                expand_cluster(neighbors[i], m, group, cluster, all_points, classification, eps, cluster_id, min_points, count)
#            ..my pb now is that i have to remember the previous neighbors to continue
    print(cluster)       
    return cluster

#W = np.array([6.3,2.5,5,1.9])
#n, neighbors = region_query(X, W, eps)

min_points = 15
#classification = np.zeros((1,len(X)))   
#cluster = expand_cluster(W, n, neighbors, cluster, X, classification, eps, 1, min_points)
#print(cluster)

#print(classification)


##A function that iterates over all points in the db and if they are core points expands them (DBSCAN)
def dbscan(all_points, eps, min_points):
    classification = np.zeros((1,len(all_points)))
    clusters = {}
    noise = []
    cluster_id = 1
    cnt = 0
    for i in range(len(all_points)):
        cnt += 1
#        print(all_points[i])
        n, neighbors = region_query(all_points, all_points[i,:], eps)
        if n > min_points:
#            then the point is a core point and we start a cluster if it's not already part of one
            if classification[0][i]== -1 or classification[0][i]== 0:
                print("core_point: {}".format(all_points[i,:]))
                print("cluster_id {}".format(cluster_id))
                cluster = []
                count = 0
                cluster = expand_cluster(all_points[i,:], n, neighbors, cluster, all_points, classification, eps, cluster_id, min_points, count)
                clusters[cluster_id]=cluster
#                print(cluster)
#                print(clusters[cluster_id])
                cluster_id += 1
        else:
#           For noise, let's put a value of -1
            classification[0][i]= -1
            noise.append(all_points[i,:])
#    print(classification)
    return clusters, noise, classification, cnt


clusters, noise, classification, cnt = dbscan(X, eps, min_points)
print(cnt)
print(classification)
#print(clusters)
print(noise)

i = 0
color = ['g','b','y']
for key, value in clusters.items():
    group = np.asarray(clusters[key])
    plt.scatter(group[:,0],group[:,1], c=color[i])
    i += 1
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
color = ['g','b','y']
for key, value in clusters.items():
    group = np.asarray(clusters[key])
    ax.scatter(group[:,0],group[:,1], group[:,2], c=color[i])
    i += 1
plt.show()


   
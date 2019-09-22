#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:29:23 2019

@author: yael
"""

import numpy as np
import warnings
from scipy import misc
from sklearn.decomposition import PCA
import os

warnings.filterwarnings("ignore")

path = './faces94'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

count = 0            
for f in files:
  count += 1
print("nber of pictures {}".format(count))  

pictures = []
for i in range(len(files)):
#    print(f)
    pictures.append(np.reshape(misc.imread(files[i]), (1,np.product(misc.imread(files[i]).shape))))

Data = np.array(pictures)
Data = np.squeeze(Data)

print(Data.shape)

X = Data[0:30]
print(X.shape)

face = misc.imread(files[0])
face = np.reshape(face, (1,np.product(face.shape)))

pca = PCA(n_components=30)
pca.fit(Data)

print(pca.singular_values_)    

#print(face.shape)








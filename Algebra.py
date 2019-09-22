#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:32:01 2019

@author: yael
"""
import numpy as np
import sympy as sy

A = np.array([[5,0,2,1],[3,3,2,-1],[1,6,2,-3]], dtype = 'float64')


#Just for test
#A = np.array([[0,0,2,1],[0,3,2,-1],[1,6,2,-3]])
print(A)

#Stopped in the middle!
def go_to_reduced_row_echelon_form(A):
#   Find number of columns of A: c
    c = np.size(A, axis=1)
    print(c)
#    Find number of rows of A: r
    r = np.size(A, axis=0)
    print(r)
    
#    for col in range(0,c):
#   Find index of first non zero entry in 1st column
    cl = A[:,0]
    i = np.nonzero(cl)[0][0]
#   Swap between 1st line and line of the first non-zero entry in 1st column
    A[[0,i]]= A[[i,0]]
    print(A)
#    print(A[0][0])
#    col = 0
    for col in range(0,c-1):
      if(np.count_nonzero(A[r-1])!= 0):
        p = A[col,col]
        print(p)
#    Multiply each element in pivot line by inverse of pivot so pivot becomes one
        A[col]= A[col]*(1/p)
      
      
        print(A)
#    Below the 1 in the column we have to bring zeros
        ind = col+1
      
        while (ind in range(col+1,r)):
        
#        First element

           v = A[ind,col]
#        print(v)
#        print(A[col])
#        print(v*A[col])
#        print(A[ind])
#        print(A[ind] - v*A[col])
           A[ind] = A[ind] - v*A[col]
           A = np.round(A, decimals=2, out=None)
           print(A)
           
#        print(A[ind])
           ind += 1
        
#    Above the 1 in the column we have to bring zeros
#    col += 1  
#    Move to 2nd pivot: on the second row
    print("Row echelon form")
    print(A)
    if(np.count_nonzero(A[r-1])!= 0):
      for col in range(1,c-1):
        print("second part")
        p = A[col,col]
        print(p)
        idx = 0
        while idx in range(idx,col):
          val = A[idx,col]
          A[idx]= A[idx] - val*A[col]
          idx += 1
      print(A)
    return A

go_to_reduced_row_echelon_form(A)
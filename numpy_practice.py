#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:41:44 2019

@author: yael
"""

import numpy as np

print (np.version.version)

v = np.zeros(10)
print(v)

print(v.itemsize)

#5in consle

#6
u = np.zeros(10)
u[4]= 1
print(u)

#7
arr = np.arange(10,50)
print(arr)

#8
rev = arr[::-1]
print(rev)

#9
arr = np.arange(0,9).reshape((3,3))
print(arr)

#10
w = np.array([1,2,0,0,4,0])
print(w)
print(w.nonzero())

#11
idm = np.eye(3)
print(idm)

#12
import random
z = np.random.rand(3,3,3)
print(z)

#13
a = np.random.rand(10,10)
print(a)
print(np.amin(a))
print(np.amax(a))

#14
vec = np.arange(0,30)
print(vec.mean())

#15
arr = np.ones(5)
arr[1:-1]=0
print(arr)

#16
arr = np.array([1,2,3,4,5,6])
arr = np.pad(arr, (1), 'constant', constant_values=(0))
print(arr)

arr = np.arange(0,9).reshape((3,3))
print(arr)
arr = np.pad(arr, (1), 'constant', constant_values=(0))
print(arr)

#17
#python
#nan is not a number
#nan
print(0 * np.nan) 
# always False
print(np.nan == np.nan)
 #False
print(np.inf > np.nan) 
#nan
print(np.nan - np.nan)
#False
print(0.3 == 3 * 0.1)

#18
dia = np.diag([1,2,3,4,7])
print(dia)

#19
arr = np.ones((8,8))
print(arr)
# 
arr[::2,1::2]= 0
#
arr[1::2,::2]=0
print(arr)

#20
print(np.unravel_index(100, (6,7,8)))

#21
A = np.array(([1,0,1,0],[0,1,0,1]))
print(np.tile(A, (4,2)))

#22
B = np.random.rand(5,5)
print(B)
print((B - np.amin(B))/(np.amax(B)-np.amin(B)))

#23
C = np.ndarray(shape=(2,4), dtype = "int16")
print(C)
print(C.dtype)

#24
D = np.ones((5,3))
E = np.ones((3,2))
F = np.dot(D,E)
print(F)

#25
G = np.arange(10)
print(G)
G[3:9]*= -1
print(G)

#26
#sum(iterable, start)
# without import * it doesn't work
#print(sum(range(5),-1)) gives 10: why 10?
# with the from numpy import * it 
#from numpy import * 
#print(sum(range(5),-1))

#print(sum(range(5)))

#27
Z = np.array([2,3,4,5])

# legal
print(Z**Z)
# legal
print(Z <- Z) 
# legal
print(1j*Z)

#28
#nan
#print(np.array(0) / np.array(0))
# 0 but warning divide by 0!
#print(np.array(0) // np.array(0))

#29
#np.copysign changes the sign of x1 to that of x2, element-wise.

#If both arguments are arrays or sequences, they have to be of the same length.
# If x2 is a scalar, its sign will be copied to all elements of x1.
print(np.copysign([2,3,4], -5))
print(np.copysign([2,-3,4],[-1,2,-3]))
print(np.copysign([2.03, -3.0, 4.0], -5.0))

#np.ceil
#Returns the ceiling of the input, element-wise.

#The ceil of the scalar x is the smallest integer i, such that i >= x. 
#It is often denoted as \lceil x \rceil.
print(np.ceil(5.34))
print(np.ceil([2.18, -3.4, 7.8, 4.2, 5.0]))

#30
#Find the intersection of two arrays.
#Return the sorted, unique values that are in both of the input arrays.
print(np.intersect1d([3,4,-5,8,12,-3.5],[-3.5,4,8]))

#31
#How to ignore all numpy warnings (not recommended)
#np.seterr(all='ignore', divide='ignore', over= 'ignore', under='ignore', invalid='ignore')
#Set how floating-point errors are handled.
#Note that operations on integer scalar types (such as int16) are handled like floating point, 
#and are affected by these settings

#32
# The expression is not true
#With the line below i get a warning when entering np.sqrt(-1).
# Invalid value encountered in sqrt
# after the answer (= "False")
#np.seterr(invalid='warn')
# np.sqrt Returns the non-negative square-root of an array, element-wise.
# In comment below to take out the warning from the console
#print(np.sqrt(-1) == np.emath.sqrt(-1))

#33
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print(yesterday)
today = np.datetime64('today', 'D')
print(today)
tomorrow = today + np.timedelta64(1, 'D')
print("tomorrow will be:"+ str(tomorrow))

#34
#Below array with all the dates for the month of July
#Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
#print(Z)
#Now creation of an array with all the dates for August 2018
Z = np.arange('2018-08', '2018-09', dtype='datetime64[D]')
print(Z)

#35 ?
#How to compute ((A+B)\*(-A/2)) in place (without copy)?
A = np.array([1,2])
print(A)
B = np.array([3,4])
print(A+B)
print(-A)
print(-A/2)
#Reminder: If A and B are just Python lists as below is a simple concatination
#C = A+B
#But if they are numpy arrays then it is possible to write A+B
C = (A+B)*(-A/2)
print(C)

#36
# Extract the integer part of a random array using 5 different methods
Arr = np.random.uniform(2,100,(3,2))
print(Arr)
#0. But this rounds up sometimes so it is not exactly what is asked
#print(np.rint(Arr))
#equivalent but same pb: np.around(Arr)
#1. This is good
print(np.floor(Arr))
#2.
print(np.trunc(Arr))
#3
Res = np.modf(Arr)
print(Res[1])
#4
print(Arr - (Arr % 1))
#5
print(np.fix(Arr))

#37
#Create a 5x5 matrix with row values ranging from 0 to 4
A= np.mgrid[0:5,0:5]
print(A[1])

#Another solution
x = np.zeros((5,5))
print("Original array:")
print(x)
print("Row values ranging from 0 to 4.")
x += np.arange(5)
print(x)

#38
#Create a vector of size 10 with values ranging from 0 to 1, both excluded
A = np.linspace(0,1,12)
S = A[1:11]
print(S)

#39-40 there is no 38
#Create a random vector of size 10 and sort it
V = np.random.random((1,10))
print(V)
V.sort()
print("Question 40")
print(V)

W= np.random.randint(12,78,10)
W.sort()
print(W)

#41
#Print the minimum and maximum representable value for each numpy scalar type
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)

#42 
#How to print all the values of an array?
A= np.array([1,2,3,4,5,6,7,8])
print(A)

Z = np.zeros((100,100))
#print(Z)
np.set_printoptions(threshold=10000)
print(Z)
#43
#Find the closest value to a given scalar in an array
Ex = np.random.randint(0,100,10)
print(Ex)
#Which is the closest to 42?
print("43-----------------")
def closest_to(arr,val):  
 idx = np.abs(arr - val).argmin()
 return arr[idx]

print(closest_to(Ex, 42))

#44
#Create a structured array representing a position (x,y) and a color (r,g,b)
struc_array = np.zeros(5, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(struc_array)

#45
from scipy import spatial

arr = np.rint((np.random.rand(100,2))*100)
print(arr)
print(arr.shape)

#scipy.spatial.distance.pdist Pairwise distances between observations in n-dimensional space.
distances = spatial.distance.pdist(arr, metric='euclidean')
print(distances)
print(distances.shape)

#46
#How to convert a float (32 bits) array into an integer (32 bits) in place?
#The functions rand and ranf both create np arrays of type float64 
farr = np.random.rand(10)
print(farr)
print(farr.dtype)
farr2 = np.random.ranf(10)
print(farr2)
print(farr2.dtype)

# So it has to be converted in place to float32 to start the exercise
farr = np.float32(farr)
print(farr)
print(farr.dtype)
# The same way i did, i will just convert it in place to int32
# I multiply everything by 100 because the values were all in between 0 and 1,
#but this is only for a "nicer" result without 0 everywhere
farr = np.int32(farr*100)
print(farr)
print(farr.dtype)

#47
#How to read the following file?
#which file?
#לדלג

#48
#What is the equivalent of enumerate for numpy arrays? (★★☆)
#The equivalent is ndenumerate, example below
a = np.array([[1, 2], [3, 4]])
print(a)
for index, x in np.ndenumerate(a):
    print(index, x)

#49
#Generate a generic 2D Gaussian-like array
gauss = np.random.standard_normal((2,3))
print(gauss)

#50
#How to randomly place p elements in a 2D array? 
def rand_p_2Darr(p):
    return np.random.rand(p,1)

print(np.random.rand(12,1))

#Below is a 1D array
#print(np.random.rand(12))

#51
#Subtract the mean of each row of a matrix
#Create sample random integer matrix 3x4
mat = np.random.randint(0,20,size=(3,4))
#Convert to a  float matrix so it allows to show the full results of substracting the mean
mat = np.float32(mat)
print(mat)
#print(len(mat))

for i in range(len(mat)):
    mat[i] = mat[i]- mat[i].mean()
print(mat)

#52
#How to sort an array by the nth column?
#Create sample random integer matrix 3x4
A = np.random.randint(0,20,size=(3,4))
print(A)
print("*******************")

def sort_by_coln(M,n):
#print(A[:,2])
#print(A[:,2].argsort())
  print(M.shape)
  growing_colvalindexes = M[:,n].argsort()
#  print(growing_colvalindexes)
  sorted_M = []
  sorted_M = np.array(sorted_M)
#  print(sorted_A)

#print("----")

  for i in growing_colvalindexes:
#    print(i)
#    print(A[i,])
    sorted_M = np.append(sorted_M, M[i,])
#    print(sorted_A)

#print(sorted_A)    
  sorted_M = sorted_M.reshape(M.shape)
  print("*******************")
  print(sorted_M)
  return sorted_M

sort_by_coln(A,2)

#53
#How to tell if a given 2D array has null columns? 
B = np.random.randint(-1,5,size=(3,4))
print(B)
print(B[:,1])
print(B.shape[1])
for pos in range(B.shape[1]):
    col = B[:,pos]
    if np.all(col==0):
        print("null column!")
print("no null column!")

#54
#Find the nearest value from a given value in an array
#Let's say it's the nearest value, greater than 
C = np.random.randint(0,20,10)
print(C)
val=C[3]
print(val)

def index(array, item):
    the_index = 0
    for idx, val in np.ndenumerate(array):
        if val == item:
            the_index = idx
            return the_index

#Let's say the given value is in the 3rd position in the array
def find_nearest(the_array, given_value):
    the_array.sort()
    print(the_array)
    ind = index(the_array, given_value)
    print(ind)
    if ind == len(the_array)-1:
      n = np.int32(ind)-1
    else:
      n = np.int32(ind)+1
    nearest_val = np.int32(the_array[n])
    return nearest_val

print(find_nearest(C,val))

#55
#Create an array class that has a name attribute
class A_array:
    
    def __init__(self, name):
        self.name = name
        
#56
#Consider a given vector, how to add 1 to each element indexed by a second vector 
#(be careful with repeated indices)?
arr = np.random.randint(0,20,10)
print(arr)
select = np.random.randint(0,5,3)
print(select)
print(arr[select])
dict_flag = {x: 0 for x in select}
for i in select:
    if dict_flag[i]==0:
      arr[i]= arr[i]+1
      dict_flag[i]=1      
    else:
      continue
print(arr)

#57
#How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?

# vector X
X = [1,3,9,3,4,1]

#indexes I weights
I = [1,3,1,3,1,2]
  
F = np.bincount(X, weights = I) 
print("Summation element-wise : \n", F) 


#58
#Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
#every point is RGB
#R is in array of integers [0,255], so is G, so is B
#number of unique colours is 
#For every pixel r*256*256, g*256, b*1, so we get a flatten array of all the pixel values for a picture

#For example h=32, w=32
h = 32
w = 32
I = np.random.randint(0,255,(h,w,3)).astype(np.ubyte)


F= I[...,0]*256*256+I[...,1]*256+I[...,2]
n = len(np.unique(F))
print(n)

#59
#Considering a four dimensions array, how to get sum over the last two axis at once?
A = np.random.randint(0,10,(2,3,2,3))
print(A)
sum = A.sum(axis=(-2,-1))
print(sum)

#60 
#Considering a one-dimensional vector D, how to compute means of subsets of D using a
#vector S of same size describing subset indices? 
D = np.random.randint(0,10,5)
print(D)
S = np.random.randint(0,4,5)
print(S)
#D_sums = np.bincount(S, weights=D)
#print(D_sums)
#D_counts = np.bincount(S)
#print(D_counts)
#D_means = D_sums / D_counts
#print(D_means)

#61
#How to get the diagonal of a dot product? 
#A dot product will result into an np array, there must be a function to get the diagonal
#sample np array
A = np.array([[2,3,4],[6,3,2],[4,7,8]])
print(A)
D = np.diagonal(A)
print(D)

#62
#Consider the vector [1, 2, 3, 4, 5], 
#how to build a new vector with 3 consecutive zeros interleaved between each value?

#3 lines below just study of np.reshape
#v= v.reshape(5,1)
#print(v)
#print(v.reshape(5))

#62 The exercise
v = np.array([1,2,3,4,5])
added = np.zeros(3)
print(added)
positions = np.array([0,1,2,3])

new = np.zeros((4,1,4))

for i in positions:
   print(added)
   print(v[i])
   new[i] = np.insert(v[i],1,added)
   print(new[i])   
#print(new)
new_new = np.reshape(new,16)
#print(new_new)
new_v = np.append(new_new, [5])
print(new_v)

#63
#Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?
#My intuition: Maybe add an all-zero dimension to the smaller array?

#Use of [:,None]
#To explain it in plain english, it allows operations between two arrays of different number of dimensions.
#It does this by adding a new, empty dimension which will automagically fit the size of the other array.
#So basically if:
#
#Array1 = shape[100] and Array2 = shape[10,100]
#
#Array1 * Array2 will normally give an error.
#
#Array1[:,None] * Array2 will work

#Let's try
A = np.random.randint(0,10,(5,5,3))
B = np.random.randint(0,10,(5,5))

print(B[:,:,None]*A)

#64
#How to swap two rows of an array?
A = np.random.randint(0,10,(5,3))

print(A)

A[[1, 3]] = A[[3, 1]]

print(A)

#65
#Consider a set of 10 triplets describing 10 triangles (with shared vertices),
#find the set of unique line segments composing all the triangles

#66
print("66")
#Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
C = np.bincount([1,1,2,3,4,4,6])
print(C)

print(np.arange(len(C)))
A = np.repeat(np.arange(len(C)), C)
print(np.repeat(np.arange(len(C)), C))
print(A)

#67
#How to compute averages using a sliding window over an array? (★★★)

A = np.array([1,2,3,4,5,6,7,8,9])

Av = np.zeros(7)
for n in range(7):
   Av[n] = np.average(A[n:n+2])
   print(Av)
Averages = np.reshape(Av,7)
print(Averages)

#68
#Consider a one-dimensional array Z, 
#build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) 
#and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[- 1])
from numpy.lib import stride_tricks

Z = np.random.randint(0,30,12)
print(Z)

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

#69
#How to negate a boolean, or to change the sign of a float inplace? 
Z = np.random.randint(0,2,10)
print(Z)
np.logical_not(Z, out=Z)
print(Z)

F = np.random.rand(3)
print(F)
np.negative(F,out = F)
print(F)


#70
#Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance
#from p to each line i (P0[i],P1[i])? 

p1=np.array([0,0])
p2=np.array([10,10])
p3=np.array([5,7])
d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
print("distance")
print(d)

#71

#72
#Consider an arbitrary array, write a function that extract a subpart with a fixed shape and
#centered on a given element (pad with a fill value when necessary)
Arr = np.array([2,3,4,6,7,8,9])

#73
#Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R =
#[[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? 

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

#74
#Compute a matrix rank
from numpy.linalg import matrix_rank

G = np.array([[3,0,-4],[7,0,-4],[6,0,8]])
print(G)
print(matrix_rank(G))

#75
#How to find the most frequent value in an array?
A = np.array([1,2,5,7,1,2,2,1,1,1,5,6,7])
B = np.bincount(A)
print(B)
mfv = np.argmax(B)
print(mfv)

#76
#Extract all the contiguous 3x3 blocks from a random 10x10 matrix 

#72
#Consider an arbitrary array, write a function that extract a subpart with a fixed shape and
#centered on a given element (pad with a fill value when necessary)
#Arr = np.array([2,3,4,6,7,8,9])

#86
#Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) 
#M = np.random.randint(0,10, (10,3))
M = np.array(([8,0,1],[8,8,9],[3,2,7],[2,2,2],[3,6,4],[7,9,9],[7,6,7],[2,5,8],[5,2,4],[4,6,0]))
print(M)
print(len(M))
for i in range(len(M)):
    if len(np.unique(M[i]))==1:
        print("There is a line with a unique value! we want all the other rows")
        NM = np.array(M[0:i])
        NMM = np.array(M[i+1:])
        M = np.concatenate((NM,NMM),axis=0)
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:33:33 2019

@author: yael
"""

import numpy as np
import pandas as pd

data = pd.read_csv("wdbc.data", header=None)
print(data.head())

#Build a recursive algorithm for a decision tree with Gini index

#Classification: The Gini cost function is used which provides an indication of how pure the nodes are, 
#where node purity refers to how mixed the training data assigned to each node is.
#Splitting continues until nodes contain a minimum number of training examples or a maximum tree depth is reached.

#1. Use 80% of the dataset as training data
def splitDataset(Data, splitRatio):
    print("Data {}".format(len(Data)))
    n = int(splitRatio * len(Data))
    Data_train = Data[Data.index < n]
    print(len(Data_train))
    Data_test = Data[Data.index >= n]
    print(len(Data_test))
    return Data_train, Data_test

train_data, test_data = splitDataset(data, 0.8)

print(train_data.shape[1])

#בנה פונקציה שמפצלת את ה-dataset לפי חוק.
#￼￼dataset . . אינדקס של פיצ'ר שמפצלים עליו . ערך שמפצלים עליו
#הפונקציה מחזירה שני datasets שהם המקורי מפוצל לשניים

#1.a.
def split_acc_to_feature_value(Dataset, ind, val):
    setA = Dataset[Dataset[ind]<val]
    setB = Dataset[Dataset[ind]>=val]
    setA = setA.reset_index(drop= True)
    setB = setB.reset_index(drop = True)
    return setA, setB

def prop_classk_in_group(nber_classk, nber_total):
    if nber_total != 0:
        return nber_classk/nber_total

def compute_entropy(Dataset):
    DatasetM = Dataset.loc[Dataset[1] == 'M']
#    print(len(groupM))
    DatasetB = Dataset.loc[Dataset[1] == 'B']
#    print(len(groupB))
    if len(DatasetM)!= 0:
        pM = prop_classk_in_group(len(DatasetM),len(Dataset))
    else:
        pM = 0
#    print(pM)
    if len(DatasetB)!= 0:
        pB = prop_classk_in_group(len(DatasetB),len(Dataset))
    else:
        pB = 0
    entropy = -1* pM * np.log2(pM + np.finfo(float).eps) -1 * pB * np.log2(pB + np.finfo(float).eps)
    return entropy
#
print(compute_entropy(train_data))

def get_split(Dataset):
    entering_entropy = compute_entropy(Dataset)
    best_gain = 0  # keep track of the best information gain
    col_bg, row_bg = 0,0 # keep track of col, row giving the value with the best information gain when it's the object of the question
 # Substract 3 because 3 columns are not features: index, id, label
    for col in range(3, Dataset.shape[1]):
        for row in range(len(Dataset)):
#            print("col row {} {}".format(col,row))
            groupA, groupB = split_acc_to_feature_value(Dataset, col, Dataset.iloc[row,col]) 
#in groupA the answer is true, entA is like ent of this feat , this value is true
            entA = compute_entropy(groupA)
#            print(entA)
            entB = compute_entropy(groupB)
#            print(entB)
#           average entropy information:
            I = (len(groupA)/len(Dataset))*entA+ (len(groupB)/len(Dataset))*entB
            gain = entering_entropy - I
            if gain> best_gain:
                best_gain = gain
                col_bg, row_bg = col, row
    parting_value = Dataset.iloc[row_bg,col_bg]
    return col_bg, row_bg, parting_value

print(get_split(train_data))           

class Node:
    
    def __init__(self, tree_level, entering_dataset):
        self.right = None
        self.left = None
        self.parting_feature = None
        self.parting_value = None
        self.tree_level = tree_level
        self.entering_dataset = entering_dataset
        self.label = None
        
        
    def print_node(self):
        print("x{} < {}".format(self.parting_feature, self.parting_value))
        
    def decision(self, rows):
        if len(rows.loc[rows[1] == 'M'])>len(rows.loc[rows[1] == 'B']):
            return 'M'
        else:
            return 'B'
        
    def print_tree(self, spacing=""):
    # Base case: we've reached a leaf
        if self.label!=None:
            print (spacing + "Predict", self.label)
        else:
    # Print the question at this node
           self.print_node()

    # Call this function recursively on the true branch
           print (spacing + '--> Left:')
           self.left.print_tree(spacing + "  ")

    # Call this function recursively on the false branch
           print (spacing + '--> Right:')
           self.right.print_tree(spacing + "  ")
           
    def calc_majority_label(self):
        labeled_M = len(self.entering_dataset.loc[self.entering_dataset[1] == 'M'])
        labeled_B = len(self.entering_dataset.loc[self.entering_dataset[1] == 'B'])
        return 'M' if labeled_M > labeled_B else 'B'
            
    def calc_node_purity(self):
        labeled_M = len(self.entering_dataset.loc[self.entering_dataset[1] == 'M'])
        labeled_B = len(self.entering_dataset.loc[self.entering_dataset[1] == 'B'])
        majority_label = self.calc_majority_label()
        if majority_label == 'M':
            purity = labeled_M/(labeled_M+labeled_B)
        else:
            purity = labeled_B/(labeled_M+labeled_B)
        return purity
    
    def split_node(self, max_depth, depth):
        
        #check for a no split: if all data is the same or if we've reached the max depth
        print("depth: {}".format(depth))
        purity = self.calc_node_purity()
        if (purity == 1) or (depth >= max_depth):
            self.label = self.decision(self.entering_dataset)
            print("Terminal node with decision: {}".format(self.label))
        else:
            left_data, right_data = split_acc_to_feature_value(self.entering_dataset, self.parting_feature, self.parting_value)
            pf, rowbg, pv = get_split(right_data)
            self.right = Node(depth +1, right_data) 
            self.right.parting_feature = pf
            self.right.parting_value = pv
            print("Right node depth {}".format(depth +1))
            self.right.split_node(max_depth, depth +1)
            self.left = Node(depth +1, left_data)
            pf2, rowbg2, pv2 = get_split(left_data)
            self.left.parting_feature = pf2
            self.left.parting_value = pv2
            print("Left node depth {}".format(depth +1))
            self.left.split_node(max_depth, depth +1)
            
    def predict(self,row):
       if self.label != None:
           return self.label
       else:
           if row[self.parting_feature] < self.parting_value:
               return self.left.predict(row)
           else:
               return self.right.predict(row)
           
root = Node(0, train_data)
col_bg, row_bg, parting_value = get_split(root.entering_dataset)
root.parting_feature = col_bg
root.parting_value = parting_value
print("Root depth {}".format(root.tree_level))
root.print_node()
root.split_node(4,0)
print("----------TREE----------")
root.print_tree("")

print(len(test_data))

count = 0
for row in range(len(test_data)):
    if root.predict(test_data.iloc[row,:])==test_data.iloc[row,1]:
        count += 1

print(count)

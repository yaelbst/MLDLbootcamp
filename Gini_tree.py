#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:11:50 2019

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

def calculate_gini(left,right):
    gini = 0
    total_size = len(left) + len(right)
    for branch in [left,right]:
        branch_score = 0
        branch_size = len(branch)
        if branch_size==0:
            continue
        for label in ['M','B']:
            num_instances = len(branch.loc[branch[1] == label])
            probability_in_branch = prop_classk_in_group(num_instances,branch_size)
            branch_score += probability_in_branch**2
        gini += (1-branch_score)*branch_size/total_size
    return gini
#If all the labels are the same and they're all on one side then gini is 0

#Use gini instead of info gain, calculate

#print("impurity {}".format(measure_impurity_in_group(setA)))


#1.b.
#בנה פונקציה )get_split( שמקבלת רשימה של נתונים כולל ה-label ובוחרת לפי כלל ג'יני מה הפיצ'ר
#A node represents a single input variable (X) and a split point on that variable, 
#assuming the variable is numeric. 
#The leaf nodes (also called terminal nodes) of the tree contain an output variable (y) which is used to make a prediction.

#A Gini score gives an idea of how good a split is by how mixed the classes are in the two groups created by the split.
#A high gini score shows a great mixture, that is a high impurity. The purest the data is on the two sides, the best is the split,
#    and the lowest the gini index is: thus we will look for the lowest gini to choose our questions
#    We will compute the gini index on every possible split(question on each of the values of each of the features)
#    and will choose the lowest gini to decide of the "winning" feature/value for the tree node
def get_split(Dataset):
    lowest_gini = 999  # keep track of the lowest gini, so we start on purpose with the high one
    col_bg, row_bg = 0,0 # keep track of col, row giving the value with the best information gain when it's the object of the question
 # Substract 3 because 3 columns are not features: index, id, label
    for col in range(3, Dataset.shape[1]):
        for row in range(len(Dataset)):
#            print("col row {} {}".format(col,row))
            groupA, groupB = split_acc_to_feature_value(Dataset, col, Dataset.iloc[row,col]) 
            gini_ind = calculate_gini(groupA, groupB)
#            print(gain)
            if gini_ind < lowest_gini:
                lowest_gini = gini_ind
                col_bg, row_bg = col, row
#    print(col_bg, row_bg)
#    print(Dataset.shape)
    parting_value = Dataset.iloc[row_bg,col_bg]
    return col_bg, row_bg, parting_value

#col_bg, row_bg, parting_value = get_split(train_data)
#print(train_data.loc[row_bg,col_bg])
#print(col_bg, row_bg)

#sA, sB = split_acc_to_feature_value(train_data, col_bg, parting_value)
#sA = sA.reset_index(drop = True)

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
root.split_node(3,0)
print("----------TREE----------")
root.print_tree("")

print(root.predict(test_data.iloc[0,:]))

print(len(test_data))

count = 0
for row in range(len(test_data)):
    if root.predict(test_data.iloc[row,:])==test_data.iloc[row,1]:
        count += 1

print(count)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:54:53 2019

@author: yael
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#1. Apply
#Special thanks to: https://github.com/justmarkham for sharing the dataset and materials United States - Crime Rates - 1960 - 2014
#Step 1. Import the necessary libraries
#Step 2. Import the dataset from thish
#https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rate s/US_Crime_Rates_1960_2014.csv
#Step 3. Assign it to a variable called crime.
crime = pd.read_csv('uscrime.csv', sep =',',header=0)
print(crime.head(10))
#Step 4. What is the type of the columns?
print(crime.dtypes)
#Have you noticed that the type of Year is int64. But pandas has a different type to work with
#Time Series. Let's see it now.
#Step 5. Convert the type of the column Year to datetime64
crime['Year'] = pd.to_datetime(crime['Year'],yearfirst=True, format='%Y')
print(crime.dtypes)
#Step 6. Set the Year column as the index of the dataframe
crime = crime.set_index(crime['Year'])
#Step 7. Delete the Total column
crime = crime.drop(['Total'], axis=1)
#Step 8. Group the year by decades and sum the values
crime = crime.drop(['Population'], axis=1)
crime['Decade'] = pd.DatetimeIndex(crime['Year']).year
crime['Decade'] = (crime['Decade']//10)*10
print(crime.groupby(crime['Decade']).sum())
#Pay attention to the Population column number, summing this column is a mistake Step 9. What is the mos dangerous decade to live in the US?
print(crime.groupby(crime['Decade']).sum().idxmax())
#Apparently the 90s

#2. Stats
#We are going to use a subset of [US Baby Names](https://www.kaggle.com/kaggle/us-baby-names) from Kaggle
#In the file it will be names from 2004 until 2014
#Step 1. Import the necessary libraries
#Step 2. Download and extract the data from this link
#https://drive.google.com/open?id=1At_YCsguCIEqe3l-gSPPquaOORbGLKmj
baby_names = pd.read_csv('NationalNames.csv')
print(baby_names.head(10))
#Step 3. Assign it to a variable called baby_names. 
#Step 4. See the first 10 entries
#0' and 'Id'
#Step 6. Is there more male or female names in the dataset?
print(baby_names.groupby(baby_names['Gender'])[['Count']].sum())
#More males!
#Step 7. Group the dataset by name and assign to names
names = baby_names.groupby('Name')[['Count']].sum()
#Step 8. How many different names exist in the dataset?
print(len(names))
##Step 9. What is the name with most occurrences?
print(names['Count'].idxmax())
##Step 10. How many different names have the least occurrences? 
minocc = names.min()
#if i put minocc or names.min() instead of 5, i get an error :()
print(names[names.Count == 5].count())
#Step 11. What is the median name occurrence?
print(names[names.Count == names.Count.median()])
#Step 12. What is the standard deviation of names?
print(names.Count.std())
#Step 13. Get a summary with the mean, min, max, std and quartiles.
print(names.Count.mean())
print(names.Count.min())
print(names.Count.max())
print(names[names.Count == 242874].idxmax())
print(names.Count.std())

#3. Visualization
#Step 1. Import the necessary libraries set this so the graphs open internally\n, 
#Step 2. Import the dataset from this
#https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv
#Step 3. Assign it to a variable called chipo.
chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t', header=0)
#Step 4. See the first 10 entries
print(chipo.head(10))
#Step 5. Create a histogram of the top 5 items bought
chipo_most_sold = chipo.groupby(['item_name'])['quantity'].sum().sort_values(ascending=False)
#5_most
print(chipo_most_sold)

chipo_most_sold[0:5].plot.bar()

chipo['item_price'] = chipo['item_price'].str.replace('$', '')
chipo['item_price'] = chipo['item_price'].astype(float)
#Step 6. Create a scatterplot with the number of items orderered per order price\n,
#Price should be in the X-axis and Items ordered in the Y-axis 
c = chipo.groupby('item_price')['quantity'].sum().to_frame()
plt.figure()
plt.axes()
plt.scatter(c.index,c.quantity)
plt.show()
#Create a question and a graph to answer your own question.

#4. Creating Series and DataFrames This time you will create the data
#Step 1. Import the necessary libraries
#Step 2. Create a data dictionary that looks like the DataFrame below
#evolution hp name pokedex type
#          0 Ivysaur 45 Bulbasaur yes grass
#1 Charmeleon 2 Wartortle 3 Metapod
#39 Charmander no fire
#44 Squirtle yes water
#45 Caterpie no bug
data = {'evolution': ['Ivysaur', 'Charmeleon', 'Wartortle', 'Metapod'],
        'hp': ['45','39','44','45'],
        'name': ['Bulbasaur', 'Charmander', 'Squirtle', 'Caterpie'],
        'pokedex': ['yes', 'no', 'yes', 'no'],
        'type': ['grass', 'fire', 'water', 'bug']}

#print(data)
#Step 3. Assign it to a variable called
df = pd.DataFrame(data)
#Step 4. Ops...it seems the DataFrame columns are in alphabetical order. 
#Place the order of the columns as name, type, hp, evolution, pokedex
df = df.reindex(['name', 'type', 'hp', 'evolution', 'pokedex'], axis=1)
#Step 5. Add another column called place, and insert what you have in mind.
df['newcol']= range(4)
print(df)
#Step 6. Present the type of each column Create your own question and answer it.
print(df.dtypes)

#5 Time_Series
#Step 1. Import the necessary libraries Step 2. Import the dataset from this
#https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv
#Step 3. Assign it to a variable called
timeser = pd.read_csv('https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv')
#Step 4. What is the frequency of the dataset?
#Step 5. Set the column Date as the index.
timeser = timeser.set_index('Date')
#Step 6. What is the type of the index?
print('index type: {}'.format(timeser.index.dtype))
#Step 7. Set the index to a DatetimeIndex type
#timeser.index = pd.to_datetime(timeser.index)
timeser = pd.DataFrame(timeser, index=pd.to_datetime(timeser.index))
print(timeser.index.dtype)
#Step 8. Change the frequency to monthly, sum the values and assign it to monthly.
#timeser.index = timeser.index.to_period("M")
timeser = timeser.resample('M').sum()
#Step 9. You will notice that it filled the dataFrame with months that don't have any data with
#NaN. Let's drop these rows.
timeser = timeser[(timeser.T != 0).any()]
#Step 10. Good, now we have the monthly data. Now change the frequency to year.
timeser = timeser.resample('Y').sum()
#Create your own question and answer it.
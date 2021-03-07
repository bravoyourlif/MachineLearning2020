'''
Machine Learning 2020 Spring
Yonsei University
Chaeyeon Han

Dataset : Drug Consumption
Purpose : Ecstasy Consumption Level Prediction

I have manually transformed the original dat file into csv file.
'''

import numpy as np
import pandas as pd
import csv
import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# saved the original .data file in csv format and imported
input_file = "drug_consumption_numeric.csv"

df = pd.read_csv(input_file, header = 0)
original_headers = list(df.columns.values)

# dragging 'ecstasy' column to the end in order to seperate X and y
col = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
       'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
       'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushroom',
       'nicotine', 'semer', 'vsa','ecstasy']

df = df[col]

print("< Prediction with Original Dataset >\n")

# defining X and y by slicing
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

# split train set and test set 7:3
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

# set xgboost model and run
xgb_model = xgb.XGBClassifier(num_class=7, objective="multi:softprob")

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

# evaluation metrics - original (no feature engineering)
print(classification_report(y_test,y_pred))


# FEATURE ENGINEERING
# --------------------

# 1: amphet, cannabis, coke, ketamine, legalh, lsd, mushroom has the most correlation. so I gave weight to them. --> drug_sum

print("< Feature Engineering - 1: Using the Sum of Important Drug consumptions >\n")

col2 = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
       'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
       'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushroom',
       'nicotine', 'semer', 'vsa','ecstasy']

df = df[col2]
df2 = df.copy()

drug_sum = []

for i in range(0,len(df2)):
    count = 0
    if df2.loc[[i],['amphet']].values[0] != [0]:
        count += df2.loc[[i],['amphet']].values[0][0]
    if df2.loc[[i],['cannabis']].values[0] != [0]:
        count += df2.loc[[i],['cannabis']].values[0][0]
    if df2.loc[[i],['coke']].values[0] != [0]:
        count += df2.loc[[i],['coke']].values[0][0]
    if df2.loc[[i],['ketamine']].values[0] != [0]:
        count += df2.loc[[i],['ketamine']].values[0][0]
    if df2.loc[[i],['legalh']].values[0] != [0]:
        count += df2.loc[[i],['legalh']].values[0][0]
    if df2.loc[[i],['lsd']].values[0] != [0]:
        count += df2.loc[[i],['lsd']].values[0][0]
    if df2.loc[[i],['mushroom']].values[0] != [0]:
        count += df2.loc[[i],['mushroom']].values[0][0]
    
    drug_sum.append(count)

df2['drug_sum'] = drug_sum

new_col = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
       'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
       'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushroom',
       'nicotine', 'semer', 'vsa','drug_sum','ecstasy']

df2 = df2[new_col]

X = df2.iloc[:,:-1]
y = df2.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# evaluation metrics
print(classification_report(y_test,y_pred))

# 2: amphet, cannabis, coke, ketamine, legalh, lsd, mushroom has the most correlation. so I gave weight to them.--> drug_count

print("< Feature Engineering - 2: Counting Important Drugs >\n")

df3 = df2.copy()

drug_count = []

for i in range(0,len(df3)):
    count = 0
    if df2.loc[[i],['amphet']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['cannabis']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['coke']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['ketamine']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['legalh']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['lsd']].values[0] != [0]:
        count += 1
    if df2.loc[[i],['mushroom']].values[0] != [0]:
        count += 1
    
    drug_count.append(count)

df3['drug_count'] = drug_count

new_col = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
       'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
       'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushroom',
       'nicotine', 'semer', 'vsa','drug_sum','drug_count','ecstasy']

df3 = df3[new_col]
X = df3.iloc[:,:-1]
y = df3.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# evaluation metrics
print(classification_report(y_test,y_pred))

# 3: oscore, ss --> personality

print("< Feature Engineering - 3: Using Personality Traits >\n")

df4 = df3.copy()

demo_count = []

for i in range(0,len(df4)):
    count = 0
    if df4.loc[[i],['oscore']].values[0][0] < 0:
        count += 1
    if df4.loc[[i],['ss']].values[0][0] < 0.7:
        count += 1
    
    demo_count.append(count)

df4['demo_count'] = demo_count

new_col = ['age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
       'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
       'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
       'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushroom',
       'nicotine', 'semer', 'vsa','drug_sum','drug_count','demo_count', 'ecstasy']

df4 = df4[new_col]

X = df4.iloc[:,:-1]
y = df4.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# evaluation metrics
print(classification_report(y_test,y_pred))

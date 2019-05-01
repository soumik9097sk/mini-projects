# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

import csv

with open('/home/soumik/Downloads/iris.data', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('iris.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(("sepal length", "sepal width" , "petal length", "petal width", "class"))
        writer.writerows(lines)
        
df = pd.read_csv("iris.csv")


df["classValue"] = df['class'].map({"Iris-virginica":2, "Iris-setosa":0,"Iris-versicolor":1})
df.drop(['class'], axis = 1,inplace = True)



X = df.iloc[:,:4]
Y = df['classValue'] 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)
pred = logmodel.predict(x_test)

print(pred)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))


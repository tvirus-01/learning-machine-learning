#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 23:21:33 2022

@author: @tvirus-01
"""
# Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training Simple Linear Regression Model on the Training Set using sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predecting the test results
y_pred = regressor.predict(X_test)

#Visualising Test set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary || Experince (Training set)")
plt.xlabel("Exprince by Years")
plt.ylabel("Salary")
plt.show()

#Visualising Training set result
plt.scatter(X_test, y_test, color='orange')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title("Salary || Experince (Test set)")
plt.xlabel("Exprince by Years")
plt.ylabel("Salary")
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))
#Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.
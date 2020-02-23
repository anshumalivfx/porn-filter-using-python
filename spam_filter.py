# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:12:50 2019

@author: Cipher.Snowden
"""

# Spam Filtering

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Data Import
data = pd.read_csv('DATA/custom_email_spam.csv')
x = data.iloc[:, -2].values
y = data.iloc[:, -1].values

# Label Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x = labelencoder_x.fit_transform(x)
x = x.reshape(-1, 1)

# One Hot Encoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# Data Splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting SVM to the Training Set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)

# Predicting the Test Set Results
y_pred = classifier.predict(x_test)

# Printing Actual and Predicted y
print("----------Actual----------")
print(y_test)
print()

print("--------Predicted---------")
print(y_pred)
print()

plt.scatter(y_test, y_pred)
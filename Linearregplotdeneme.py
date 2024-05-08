# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 05:24:40 2024

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('proccessed_airbnb_data.csv')

# Select a single feature and the output variable
X= data[['amenities']]  # Replace 'feature_name' with your column name
y = data['log_price'].astype(float).values

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.column_stack((np.ones(len(x_train)), x_train))
X_test =  np.column_stack((np.ones(len(x_test)), x_test))


X_train_transpose = np.transpose(X_train)

beta = np.linalg.inv(X_train_transpose.dot(X_train)).dot(X_train_transpose).dot(y_train)


  

y_predict = X_test.dot(beta)
a=X_train[:5000]
a*=np.delete(a,0)
b=y_predict[:5000]
# # Visualizing the Training set results
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train,y_predict, color='blue')
# plt.title('Log Price vs Feature (Training set)')
# plt.xlabel('Feature')
# plt.ylabel('Log Price')
# plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='green')
plt.plot(a,b, color='blue')  # use the same regression line from training
plt.title('Log Price vs Feature (Test set)')
plt.xlabel('Feature')
plt.ylabel('Log Price')
plt.show()


#Select a single feature and the output variable
just_one_feature_train = X_train[:,13] # Replace 'feature_name' with your column name
just_one_feature_test= X_test[:,13]

# plt.scatter(just_one_feature_train, y_train, color='red')
# plt.plot(just_one_feature_train, y_predict, color='blue')
# plt.title('Log Price vs Feature (Training set)')
# plt.xlabel('amenities')
# plt.ylabel('Log Price')
# plt.show()

# Visualizing the Test set results
plt.scatter(just_one_feature_test, y_test, color='green')
plt.plot(just_one_feature_test, y_predict, color='blue')  # use the same regression line from training
plt.title('Log Price vs Number Of Reviews (Test set)')
plt.xlabel('Number of Reviews')
plt.ylabel('Log Price')
plt.show()
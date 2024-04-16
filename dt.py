# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:35:46 2024

@author: User
"""

#Importing Libraries and Functions defined
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from last_functions import *
#Getting the data
dataset = pd.read_csv("proccessed_airbnb_data.csv")
Y = np.array(dataset.log_price) 
X = np.array(dataset.drop('log_price', axis=1)) 
print("Dataset has {0} entries with {1} features".format(X.shape[0], X.shape[1]))

#Separating the dataset
length = Y.shape[0] #gives the number of rows 
datum = np.arange(0, length)
test_length = int(np.round(0.25 * length))
print("Test size: {0}".format(test_length))
validation_length = int(np.round(0.2 * length))
print("Validation size: {0}".format(validation_length))
train_length = length - test_length - validation_length
print("Train size: {0}".format(train_length))
test_datum = datum[0: test_length]
validation_datum = datum[test_length: test_length + validation_length]
train_datum = datum[test_length + validation_length:]
test_Y = Y[test_datum]
test_X = X[test_datum, :]
validation_Y = Y[validation_datum]
validation_X = X[validation_datum, :]
train_Y = Y[train_datum]
train_X = X[train_datum, :]
start_time = time()
dt = training_dt(train_X, train_Y, 0, max_depth = 10)
print("Maximum depth of node is: 10")
print("It has taken {0} seconds to train the network".format(time() - start_time))
test_Y_predicted = predict_set(test_X, dt)
print("The Score: ", (RSS(test_Y_predicted, test_Y)))
plt.figure()
plt.plot(test_Y, label="line1")
plt.plot(test_Y_predicted, label="line2")
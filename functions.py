import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def cost_function(columnname, estimated_log_price, weight, bias):
    total_error = 0.0
    for i in range(len(columnname)):
        total_error += (estimated_log_price[i] - (weight*columnname[i] + bias))**2
    return total_error / len(columnname)

def mean_squared_error (y_test, y_pred_lr):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + (y_test[i]-y_pred_lr[i])**2
    MSE = RSS/len(y_test)
    return MSE

def mean_absolute_error (y_test, y_pred_lr):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + abs(y_test[i]-y_pred_lr[i])
    MSE = RSS/len(y_test)
    return MSE

def train_test_split(x,y):
    
    np.random.seed(42)  
    test_size = int(len(x) * 0.2)  
    
    x = x.to_numpy()
    y = y
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    x_train = x_shuffled[test_size:]
    x_test = x_shuffled[:test_size]
    y_train = y_shuffled[test_size:]
    y_test = y_shuffled[:test_size]
    return x_train, x_test, y_train, y_test
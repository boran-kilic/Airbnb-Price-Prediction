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
    total = 0
    for i in range(len(y_test)):
        total = total + abs(y_test[i]-y_pred_lr[i])
    MSE = total/len(y_test)
    return MSE

def train_test_split(x,y, seed, test_size):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)  
    
    x = x.to_numpy()
    y = y
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shffl = x[indices]
    y_shffl = y[indices]
    
    x_train = x_shffl[test_size:]
    x_test = x_shffl[:test_size]
    y_train = y_shffl[test_size:]
    y_test = y_shffl[:test_size]
    return x_train, x_test, y_train, y_test






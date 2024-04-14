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

data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

np.random.seed(42)  
test_size = int(len(data) * 0.1)  

x = x.to_numpy()
y = y

indices = np.arange(len(data))
np.random.shuffle(indices)

x_shuffled = x[indices]
y_shuffled = y[indices]

x_train = x_shuffled[test_size:]
x_test = x_shuffled[:test_size]
y_train = y_shuffled[test_size:]
y_test = y_shuffled[:test_size]


X_train = np.column_stack((np.ones(len(x_train)), x_train))
X_test =  np.column_stack((np.ones(len(x_test)), x_test))


X_train_transpose = np.transpose(X_train)
beta = np.linalg.inv(X_train_transpose.dot(X_train)).dot(X_train_transpose).dot(y_train)
y_pred_lr = X_test.dot(beta)

mse = mean_squared_error(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)

from sklearn import metrics
mae_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_lr = metrics.mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
r2_lr = metrics.r2_score(y_test, y_pred_lr)


print('\nMean Absolute Error of Linear Regression   : ', mae)
print('\nMean Squarred Error of Linear Regression     : ', mse)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse_lr)
print('\nR2 Score of Linear Regression                : ', r2_lr)


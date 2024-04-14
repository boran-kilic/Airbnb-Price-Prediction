import pandas as pd
import numpy as np
from functions import *

data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x,y)


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


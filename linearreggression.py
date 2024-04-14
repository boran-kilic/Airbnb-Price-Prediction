import pandas as pd
import numpy as np
from functions import *
import time

data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x,y,seed = 42, test_size = 0.2)


X_train = np.column_stack((np.ones(len(x_train)), x_train))
X_test =  np.column_stack((np.ones(len(x_test)), x_test))


X_train_transpose = np.transpose(X_train)
start_time = time.time()
beta = np.linalg.inv(X_train_transpose.dot(X_train)).dot(X_train_transpose).dot(y_train)
end_time = time.time()

training_time = end_time - start_time  
print(f"Training time of linear regression: {training_time} seconds")

y_predict = X_test.dot(beta)

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

from sklearn import metrics
mae_lr = metrics.mean_absolute_error(y_test, y_predict)
mse_lr = metrics.mean_squared_error(y_test, y_predict)
rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
r2_lr = metrics.r2_score(y_test, y_predict)


print('\nMean Absolute Error of Linear Regression   : ', mae)
print('\nMean Squarred Error of Linear Regression     : ', mse)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse_lr)
print('\nR2 Score of Linear Regression                : ', r2_lr)


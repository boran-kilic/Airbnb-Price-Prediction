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

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
r2 = r2_score(y_test, y_predict)


print('\nMean Absolute Error: ', mae)
print('\nMean Squarred Error: ', mse)
print('\nRoot Mean Squarred Error: ', rmse)
print('\nR2 Score: ', r2)

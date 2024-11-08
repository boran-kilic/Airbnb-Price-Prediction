import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import time

data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].values.reshape(-1, 1)

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


plt.figure(figsize=(8, 5))  
plt.scatter(y_test, y_predict, color='lightblue') 
plt.plot(y_test, y_test, color='purple', linewidth=2) 
plt.title("Actual vs predicted Prices")  
plt.xlabel("Actual Price") 
plt.ylabel("Predicted Price")  
plt.grid(True)  
plt.show()


mse_lr = mean_squared_error(y_test, y_predict)
mae_lr = mean_absolute_error(y_test, y_predict)
rmse_lr = root_mean_squared_error(y_test, y_predict)
r2_lr = r2_score(y_test, y_predict)

print('\nMean Absolute Error of Linear Regression   : ', mae_lr)
print('\nMean Squarred Error of Linear Regression     : ', mse_lr)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse_lr)
print('\nR2 Score of Linear Regression                : ', r2_lr)


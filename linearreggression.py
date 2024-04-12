import pandas as pd
import numpy as np

# Read the CSV file
data = pd.read_csv('proccessed_airbnb_data.csv')

# Prepare the data
x = data.drop(["log_price"],axis = 1)
y = data['log_price'].astype(float).values

#######ofc we gonna change them all

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)

from sklearn import metrics
mae_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_lr = metrics.mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
r2_lr = metrics.r2_score(y_test, y_pred_lr)


print('\nMean Absolute Error of Linear Regression     : ', mae_lr)
print('\nMean Squarred Error of Linear Regression     : ', mse_lr)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse_lr)
print('\nR2 Score of Linear Regression                : ', r2_lr)
# # Add an intercept term to X
# X = np.column_stack((np.ones(len(X)), X))

# # Calculate the coefficients using the normal equation
# X_transpose = np.transpose(X)
# beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

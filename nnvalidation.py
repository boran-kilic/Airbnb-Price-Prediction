import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.1)

hidden_dim = 4
output_dim = 1
epochs = 1000
learning_rate = 0.65

hidden_dims = []
mse_values =[]
r2scr = []
epochs = 1000
learning_rate = 0.5
for hidden_dim in range(1,11):
    W1, b1, W2, b2 = neural_network(x_train,y_train, hidden_dim, output_dim, epochs, learning_rate)
    y_predict = predict_neural(x_validation, W1, b1, W2, b2)      
    hidden_dims.append(hidden_dim)
    mse = mean_squared_error(y_validation, y_predict)
    mse_values.append(mse)

plt.figure(figsize=(10, 5))
plt.plot(hidden_dims, mse_values, marker='o')
plt.title('MSE vs. Number of Hidden Units')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

learning_rates = []
mse_values =[]
epochs = 1000
hidden_dim = 4

for learning_rate in np.arange(0.05, 1.05, 0.05):
    W1, b1, W2, b2 = neural_network(x_train,y_train, hidden_dim, output_dim, epochs, learning_rate)
    y_predict = predict_neural(x_validation, W1, b1, W2, b2)      
    learning_rates.append(learning_rate)
    mse = mean_squared_error(y_validation, y_predict)
    mse_values.append(mse)
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, mse_values, marker='o')
plt.title('MSE vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

epochslist = []
mse_values =[]

hidden_dim = 4
learning_rate = 0.65

listx = [1,5, 10,50, 100,500, 1000,2000]    
for epochs in listx:    
    W1, b1, W2, b2 = neural_network(x_train,y_train, hidden_dim, output_dim, epochs, learning_rate)
    y_predict = predict_neural(x_validation, W1, b1, W2, b2)      
    epochslist.append(epochs)
    mse = mean_squared_error(y_validation, y_predict)
    mse_values.append(mse)
plt.figure(figsize=(10, 5))
plt.plot(epochslist, mse_values, marker='o')
plt.title('MSE vs. Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

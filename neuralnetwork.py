import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import time

data = pd.read_csv('proccessed_airbnb_data.csv')
X = data.drop(["log_price"], axis=1)
y = data['log_price'].values.reshape(-1, 1)  
X_train, X_test, y_train, y_test = train_test_split(X, y, seed = 42, test_size=0.2)

hidden_dim = 4
output_dim = 1
epochs = 1000
learning_rate = 0.75

start_time = time.time()
W1, b1, W2, b2 = neural_network(X_train,y_train, hidden_dim, output_dim, epochs, learning_rate)
end_time = time.time()
training_time = end_time - start_time  
print('\n')
print(f"Training time of the decision tree: {training_time} seconds")

y_predict = predict_neural(X_test, W1, b1, W2, b2)

plt.figure(figsize=(8, 5))  
plt.scatter(y_test, y_predict, color='lightblue') 
plt.plot(y_test, y_test, color='purple', linewidth=2) 
plt.title("Actual vs predicted Prices")  
plt.xlabel("Actual Price") 
plt.ylabel("Predicted Price")  
plt.grid(True)  
plt.show()


mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
r2 = r2_score(y_test, y_predict)

print('\nMean Absolute Error of Neural Network: ', mae)
print('\nMean Squarred Error of Neural Network: ', mse)
print('\nRoot Mean Squarred Error of Neural Network: ', rmse)
print('\nR2 Score of Neural Network: ', r2)

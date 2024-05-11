
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
# from functions import *
# data = pd.read_csv('proccessed_airbnb_data.csv')
# x = data.drop(["log_price"], axis=1)
# y = data['log_price']
# X_train, X_test, y_train, y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.2)


data = pd.read_csv('proccessed_airbnb_data.csv')
X = data.drop(["log_price"], axis=1)
y = data['log_price'].values.reshape(-1, 1)  # Reshape y to make it 2D

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# ########################################################
# data = pd.read_csv('proccessed_airbnb_data.csv')
# X = data.drop(["log_price"], axis=1)
# y = data['log_price']
# X = X.to_numpy()
# y = y.to_numpy()
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Network architecture
input_dim = X_train.shape[1]
hidden_dim = 500
output_dim = 1
# Activation function and its gradient
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_grad(x):
#     return sigmoid(x) * (1 - sigmoid(x))
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_grad(x):
    """Gradient of the ReLU function."""
    return (x > 0).astype(float)
# Initialize weights and biases
def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)

def initialize_bias(output_dim):
    return np.zeros((1, output_dim))

# Forward and backward passes
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y_hat = Z2  # Linear output for regression
    return Y_hat, A1

def backward_pass(X, Y, Y_hat, A1, W1, W2):
    # Calculate loss gradients
    dZ2 = Y_hat - Y  # Ensure Y_hat and Y are both (n_samples, 1)
    dW2 = np.dot(A1.T, dZ2) / len(Y)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(Y)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_grad(A1)
    dW1 = np.dot(X.T, dZ1) / len(Y)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(Y)
    
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# Load data
# For demonstration, generate random data (replace with actual dataset)
# X1, y1 = np.random.rand(1000, 10), np.random.rand(1000, 1)




# Initialize network parameters
W1, b1 = initialize_weights(input_dim, hidden_dim), initialize_bias(hidden_dim)
W2, b2 = initialize_weights(hidden_dim, output_dim), initialize_bias(output_dim)

# Training parameters
epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    Y_hat, A1 = forward_pass(X_train, W1, b1, W2, b2)
    loss = np.mean((Y_hat - y_train)**2)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
    dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Y_hat, A1, W1, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

# Evaluation
def predict(X, W1, b1, W2, b2):
    Y_hat, _ = forward_pass(X, W1, b1, W2, b2)
    return Y_hat

predictions = predict(X_test, W1, b1, W2, b2)
print("R2 Score on Test Data:", r2_score(y_test, predictions))
plt.figure(figsize=(8, 5))  
plt.scatter(y_test, predictions, color='lightblue') 
plt.plot(y_test, y_test, color='purple', linewidth=2) 
plt.title("Actual vs predicted Prices")  
plt.xlabel("Actual Price") 
plt.ylabel("Predicted Price")  
plt.grid(True)  
plt.show()
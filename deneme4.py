# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:42:49 2024

@author: User
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
from time import time
from functions import *

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_grad(x):
    """Gradient of the ReLU function."""
    return (x > 0).astype(float)

def initialize_weights(input_dim, output_dim):
    """He initialization for weights of the neural network."""
    return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)

def initialize_bias(output_dim):
    """Initialize biases to zero."""
    return np.zeros((1, output_dim))

def forward_pass(X, W1, b1, W2, b2):
    """Perform the forward pass through the neural network."""
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y_hat = Z2  # Output layer has no activation for regression
    return Y_hat, A1

def compute_loss(Y_hat, Y):
    """Calculate mean squared error loss."""
    return np.mean((Y_hat - Y)**2)

def backward_pass(X, Y, Y_hat, A1, W1, W2):
    """Compute gradients and update parameters."""
    # Output layer gradients
    dZ2 = Y_hat - Y
    dW2 = np.dot(A1.T, dZ2) / len(Y)  
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(Y)

    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_grad(A1)
    dW1 = np.dot(X.T, dZ1) / len(Y)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(Y)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """Update the parameters using the gradients and the learning rate."""
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2
# Generate or load data


data = pd.read_csv('proccessed_airbnb_data.csv')
X = data.drop(["log_price"], axis=1)
y = data['log_price']
X_train, X_test, y_train, y_test, x_validation, y_validation = train_test_split(X,y,seed = 42, test_size = 0.2, validation_size=0.2)

# X, y = np.random.rand(1000, 10), np.random.rand(1000, 1)  # Replace with real data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Network architecture
input_dim = X_train.shape[1]
hidden_dim = 50  # You can tune this
output_dim = 1

# Initialize network parameters
W1, b1 = initialize_weights(input_dim, hidden_dim), initialize_bias(hidden_dim)
W2, b2 = initialize_weights(hidden_dim, output_dim), initialize_bias(output_dim)

epochs = 200
learning_rate = 0.01

for epoch in range(epochs):
    Y_hat, A1 = forward_pass(X_train, W1, b1, W2, b2)
    loss = compute_loss(Y_hat, y_train)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
    dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Y_hat, A1, W1, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    
def predict(X, W1, b1, W2, b2):
    Y_hat, _ = forward_pass(X, W1, b1, W2, b2)
    return Y_hat

predictions = predict(X_test, W1, b1, W2, b2)
print("R2 Score on Test Data:", r2_score(y_test, predictions))
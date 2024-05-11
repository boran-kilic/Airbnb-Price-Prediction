# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:13:01 2024

@author: User
"""

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from functions import *
#Getting the data
# diamond_dataset = pd.read_csv("preprocessed_diamond_data.csv")
# Y = np.array(diamond_dataset.price) #labels
# X = np.array(diamond_dataset.drop("price",axis = 1)) #features

data = pd.read_csv('proccessed_airbnb_data.csv')
x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values
X_train, X_test, Y_train, Y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.2)

def initialize_weights(input_dim, output_dim):
    # He initialization suitable for ReLU
    return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)

def initialize_bias(output_dim):
    return np.zeros((1, output_dim))

# Define the architecture
input_dim = X_train.shape[1]  # Number of features
hidden_dim1 = 50  # Number of neurons in the first hidden layer
hidden_dim2 = 30  # Number of neurons in the second hidden layer
output_dim = 1   # Output dimension

# Initialize weights and biases
W1 = initialize_weights(input_dim, hidden_dim1)
b1 = initialize_bias(hidden_dim1)
W2 = initialize_weights(hidden_dim1, hidden_dim2)
b2 = initialize_bias(hidden_dim2)
W3 = initialize_weights(hidden_dim2, output_dim)
b3 = initialize_bias(output_dim)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    # First hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # Second hidden layer
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # Output layer
    Z3 = np.dot(A2, W3) + b3
    Y_hat = Z3  # Linear activation for regression
    
    return Y_hat, A1, A2

def compute_loss(Y_hat, Y):
    return np.mean((Y_hat - Y)**2)

def backward_pass(X, Y, Y_hat, A1, A2, W1, W2, W3):
    # Calculate gradients
    dZ3 = Y_hat - Y
    dW3 = np.dot(A2.T, dZ3) / len(Y)
    db3 = np.sum(dZ3, axis=0, keepdims=True) / len(Y)
    print("Shape of the array:", dZ3.shape)
    print("Number of elements in the array:", dZ3.size)
    print("Number of dimensions:", dZ3.ndim)
    print("Shape of the array:", W3.T.shape)
    print("Number of elements in the array:", W3.T.size)
    print("Number of dimensions:", W3.T.ndim)
    
    dA2 = np.dot(dZ3, W3.T)
   
    dZ2 = dA2 * relu_grad(A2)
    dW2 = np.dot(A1.T, dZ2) / len(Y)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(Y)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_grad(A1)
    dW1 = np.dot(X.T, dZ1) / len(Y)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(Y)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

epochs = 100
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    Y_hat, A1, A2 = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
    # Compute loss
    loss = compute_loss(Y_hat, Y_train)
    print(f"Epoch {epoch}, Loss: {loss}")

    # Backward pass
    dW1, db1, dW2, db2, dW3, db3 = backward_pass(X_train, Y_train, Y_hat, A1, A2, W1, W2, W3)

    # Update parameters
    W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
    
def predict(X, W1, b1, W2, b2, W3, b3):
    Y_hat, _, _ = forward_pass(X, W1, b1, W2, b2, W3, b3)
    return Y_hat

# Prediction on test data
predictions = predict(X_test, W1, b1, W2, b2, W3, b3)

# Evaluate the model, e.g., using R^2 score
from sklearn.metrics import r2_score
print("R2 Score:", r2_score(Y_test, predictions))
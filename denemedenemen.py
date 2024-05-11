import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)

def initialize_bias(output_dim):
    return np.zeros((1, output_dim))

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    Y_hat = Z3  # Linear output
    return Y_hat, A1, A2

def compute_loss(Y_hat, Y):
    return np.mean((Y_hat - Y)**2)

def backward_pass(X, Y, Y_hat, A1, A2, W1, W2, W3):
    dZ3 = Y_hat - Y
    dW3 = np.dot(A2.T, dZ3) / len(Y)
    db3 = np.sum(dZ3, axis=0, keepdims=True) / len(Y)
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_grad(A2)
    dW2 = np.dot(A1.T, dZ2) / len(Y)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(Y)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_grad(A1)
    dW1 = np.dot(X.T, dZ1) / len(Y)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(Y)
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    return W1, b1, W2, b2, W3, b3

def predict(X, W1, b1, W2, b2, W3, b3):
    Y_hat, _, _ = forward_pass(X, W1, b1, W2, b2, W3, b3)
    return Y_hat


# Example data load and split
X, y = np.random.rand(1000, 10), np.random.rand(1000, 1)  # Simulated data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize parameters
input_dim, hidden_dim1, hidden_dim2, output_dim = 10, 50, 30, 1
W1, b1 = initialize_weights(input_dim, hidden_dim1), initialize_bias(hidden_dim1)
W2, b2 = initialize_weights(hidden_dim1, hidden_dim2), initialize_bias(hidden_dim2)
W3, b3 = initialize_weights(hidden_dim2, output_dim), initialize_bias(output_dim)

# Training loop
epochs, lr = 100, 0.01
for epoch in range(epochs):
    Y_hat, A1, A2 = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
    loss = compute_loss(Y_hat, y_train)
    dW1, db1, dW2, db2, dW3, db3 = backward_pass(X_train, y_train, Y_hat, A1, A2, W1, W2, W3)
    W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Predict and evaluate
predictions = predict(X_test, W1, b1, W2, b2, W3, b3)
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)


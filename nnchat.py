import numpy as np
import pandas as pd



# Initialize random seed
np.random.seed(1)

def train_test_split(x,y, seed, test_size, validation_size=0):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)     
    validation_size = int(len(x) * validation_size) 
    x = x.to_numpy()
    y = y.to_numpy()
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shffl = x[indices]
    y_shffl = y[indices]
    
    x_train = x_shffl[validation_size + test_size:]
    x_test = x_shffl[validation_size:validation_size + test_size]
    x_validation = x_shffl[:validation_size]
    
    y_train = y_shffl[validation_size + test_size:]
    y_test = y_shffl[validation_size:validation_size + test_size]
    y_validation = y_shffl[:validation_size]    
    
    if validation_size == 0:
        return x_train, x_test, y_train, y_test    
    else:
        return x_train, x_test, y_train, y_test, x_validation, y_validation
# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU function
def relu_grad(x):
    return (x > 0).astype(float)

# Forward pass through the network
def forward_pass(x, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_layer_input = np.dot(x, weights_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    output_layer_output = relu(output_layer_input)
    return hidden_layer_output, output_layer_output

# Backward pass through the network (gradient calculation)
def backward_pass(x, y, hidden_output, output, weights_output, learning_rate):
    error = y - output
    output_delta = error * relu_grad(output)
    
    output_weight_gradient = np.dot(hidden_output.T, output_delta)
    output_bias_gradient = np.sum(output_delta, axis=0, keepdims=True)
    
    hidden_delta = np.dot(output_delta, weights_output.T) * relu_grad(hidden_output)
    
    hidden_weight_gradient = np.dot(x.T, hidden_delta)
    hidden_bias_gradient = np.sum(hidden_delta, axis=0, keepdims=True)
    
    return hidden_weight_gradient, hidden_bias_gradient, output_weight_gradient, output_bias_gradient, error

# Update the weights and biases
def update_parameters(weights_hidden, bias_hidden, weights_output, bias_output, 
                      hw_grad, hb_grad, ow_grad, ob_grad, learning_rate):
    weights_hidden -= learning_rate * hw_grad
    bias_hidden -= learning_rate * hb_grad
    weights_output -= learning_rate * ow_grad
    bias_output -= learning_rate * ob_grad
    return weights_hidden, bias_hidden, weights_output, bias_output

# Define the neural network settings
neuron_number_in_hidden_layer = 16
num_of_features = 12
learning_rate = 0.1

# Initialize weights and biases
bias_hidden_layer = np.random.normal(0, 0.3, (1, neuron_number_in_hidden_layer))
weight_hidden_layer = np.random.normal(0, 0.3, (num_of_features, neuron_number_in_hidden_layer))
bias_output = np.random.normal(0, 0.3, (1, 1))
weight_output = np.random.normal(0, 0.3, (neuron_number_in_hidden_layer, 1))



data = pd.read_csv('proccessed_airbnb_data.csv')
data = data [:100]
x = data.drop(["log_price"], axis=1)
y = data['log_price']
x_train, x_test, y_train, y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.2)

train_X = x_train
train_Y = y_train


# Training the network
loss_record = []
epochs = 100
for epoch in range(epochs):
    for x, y in zip(train_X, train_Y):
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        
        # Forward pass
        hidden_output, output = forward_pass(x, weight_hidden_layer, bias_hidden_layer, weight_output, bias_output)
        
        # Backward pass
        hw_grad, hb_grad, ow_grad, ob_grad, error = backward_pass(x, y, hidden_output, output, weight_output, learning_rate)
        
        # Update parameters
        weight_hidden_layer, bias_hidden_layer, weight_output, bias_output = update_parameters(weight_hidden_layer, bias_hidden_layer, weight_output, bias_output,
                                                                                              hw_grad, hb_grad, ow_grad, ob_grad, learning_rate)
        
    loss = np.mean(np.square(error))
    loss_record.append(loss)
    if epoch > 0 and abs(loss_record[-2] - loss_record[-1]) < 0.00005:
        break

print("Training loss over epochs:", loss_record)

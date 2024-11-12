import numpy as np

def r2_score (y_test, y_predict):
    nominator = 0
    denominator=0
    for i in range(len(y_test)):
        nominator = nominator + (y_test[i]-y_predict[i])**2
    for i in range(len(y_test)):
        denominator =  denominator + (y_test[i]- y_test.mean())**2
    return 1-(nominator/denominator)    

def calc_RSS (y_test, y_predict):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + (y_test[i]-y_predict[i])**2    
    return RSS

def mean_squared_error (y_test, y_predict):
    RSS = calc_RSS (y_test, y_predict)
    MSE = RSS/len(y_test)
    return MSE

def root_mean_squared_error(y_test,y_predict):
    MSE = mean_squared_error (y_test, y_predict)
    return np.sqrt(MSE)

def mean_absolute_error (y_test, y_predict):
    total = 0
    for i in range(len(y_test)):
        total = total + abs(y_test[i]-y_predict[i])
    MSE = total/len(y_test)
    return MSE

def train_test_split(x,y, seed, test_size, validation_size=0):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)     
    validation_size = int(len(x) * validation_size) 
    x = x.to_numpy()
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


################################ DECISION TREE ##################################
def fit_tree(x_train, y_train, min_samples, max_depth, depth=0):
    num_samples, num_features = x_train.shape
    if num_samples < min_samples or depth >= max_depth:
        return np.mean(y_train)  

    best_ft, best_thr = best_split(x_train, y_train, num_features)

    left_idxs = x_train[:, best_ft] <= best_thr
    right_idxs = x_train[:, best_ft] > best_thr
    left_child = fit_tree(x_train[left_idxs], y_train[left_idxs], min_samples, max_depth, depth + 1)
    right_child = fit_tree(x_train[right_idxs], y_train[right_idxs], min_samples, max_depth, depth + 1)

    return best_ft, best_thr, left_child, right_child

def DT_RSS(child):
    RSS = 0
    mean = np.mean(child) 
    RSS = np.sum((child - mean) ** 2)
    return RSS

def best_split(x_train, y_train, num_features):
    min_error = float('inf')
    best_ft = None
    best_thr = None
    for fidx in range(num_features):
        possible_thrs = np.unique(x_train[:, fidx])
        for th in possible_thrs:
            left = y_train[x_train[:, fidx] <= th]
            right = y_train[x_train[:, fidx] > th]
            error = DT_RSS(left) + DT_RSS(right)
            if error < min_error:
                min_error = error
                best_ft = fidx
                best_thr = th
    return best_ft, best_thr

def predict_one(tree, row):
    if type(tree) is not tuple: 
        return tree         
    else:        
        feature, threshold, left_child, right_child = tree
        if row[feature] <= threshold:
            return predict_one(left_child, row)
        else:
            return predict_one(right_child, row)
        
def predict_tree(tree, x_test):
    if len(x_test.shape) > 1:
        return [predict_one(tree, row) for row in x_test]
    else:
        return predict_one(tree, x_test)
 
############################## NEURAL NETWORK ###################################

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) 

def initialize_bias(output_dim):
    return np.zeros((1, output_dim))

def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y_hat = Z2  
    return Y_hat, A1

def backward_pass(X, Y, Y_hat, A1, W1, W2):

    dZ2 = Y_hat - Y  
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

def neural_network(X_train, y_train, hidden_dim, output_dim, epochs, learning_rate):
    input_dim = X_train.shape[1]
    W1, b1 = initialize_weights(input_dim, hidden_dim), initialize_bias(hidden_dim)
    W2, b2 = initialize_weights(hidden_dim, output_dim), initialize_bias(output_dim)

    for epoch in range(epochs):
        Y_hat, A1 = forward_pass(X_train, W1, b1, W2, b2)
        # loss = np.mean((Y_hat - y_train)**2)
        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch}, Loss: {loss}')
        dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Y_hat, A1, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    return W1, b1, W2, b2

def predict_neural(X, W1, b1, W2, b2):
    Y_hat, _ = forward_pass(X, W1, b1, W2, b2)
    return Y_hat

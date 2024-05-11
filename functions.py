import numpy as np



##########################common functions#####################################
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



#############################DECISION TREE#####################################

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

def predict(tree, row):
    if type(tree) is not tuple: 
        return tree         
    else:        
        feature, threshold, left_child, right_child = tree
        if row[feature] <= threshold:
            return predict(left_child, row)
        else:
            return predict(right_child, row)
        
def predict_tree(tree, x_test):
    if len(x_test.shape) > 1:
        return [predict(tree, row) for row in x_test]
    else:
        return predict(tree, x_test)


#########################neural network############################

# def relu(x_value): #relu activation func
#  return np.maximum(0, x_value)
 
# def relu_grad(x_value): 
#  return (x_value > 0) * 1

# class NeuralNetwork(): 
#      def __init__(self, neuron_number_in_hidden_layer, num_of_features, learning_rate):
#          self.bias_hidden_layer = np.random.normal(0, 0.3, size = (1, neuron_number_in_hidden_layer)) #Bias for hidden layer
#          self.weight_hidden_layer= np.random.normal(0, 0.3, size = (num_of_features, neuron_number_in_hidden_layer)) #Weight for hidden layer
#          self.bias_output = np.random.normal(0, 0.3, size = (1, 1)) #Bias for output
#          self.weight_output = np.random.normal(0, 0.3, size = (neuron_number_in_hidden_layer, 1)) 
#         #Weight for output
#          self.learning_rate = learning_rate #learning rate
     
#      def train_one_iteration(self, train_X, train_Y): 
#          loss = 0 
#          num_of_samples = train_Y.shape[0] 
#          x_value = np.zeros((1, train_X.shape[1])) 
         
#          for i in range(num_of_samples): 
#              x_value[0, :] = train_X[i, :] 
#              y_value = train_Y[i] 
             
#              hidden_layer_before = x_value.dot(self.weight_hidden_layer) + self.bias_hidden_layer 
#              hidden_layer = relu(hidden_layer_before) #putting into ReLU function
#              output = hidden_layer.dot(self.weight_output) + self.bias_output #output
#              test_Y_predicted = relu(output) #putting into ReLU function
             
#              error_value = y_value - test_Y_predicted #Error calculation
#              loss = loss + error_value * error_value/num_of_samples #Loss calculation
             
#              weight_output_grad = -2 * error_value * hidden_layer.T * relu_grad(output) #derivative
#              bias_output_grad = -2 * error_value * relu_grad(output) #derivative
#              weight_hidden_layer_gradient = -2 * error_value * relu_grad(output) * x_value.T * (self.weight_output.T * relu_grad(hidden_layer_before)) #derivative
#              bias_hidden_layer_gradient = -2 * error_value * relu_grad(output) * (self.weight_output.T * relu_grad(hidden_layer_before)) #derivative
             
#              self.weight_output = self.weight_output - self.learning_rate * weight_output_grad #update
#              self.bias_output = self.bias_output - self.learning_rate * bias_output_grad #update
#              self.weight_hidden_layer = self.weight_hidden_layer - self.learning_rate * weight_hidden_layer_gradient #update
#              self.bias_hidden_layer = self.bias_hidden_layer - self.learning_rate * bias_hidden_layer_gradient #update
     
#          return loss 
     
#      def training(self, train_X, train_Y, maximum_epoch = 25, threshold_to_stop = 0.00005): 
#          loss_record = [] 
#          for i in range(maximum_epoch): #Iteration on each epoch
#              loss = NeuralNetwork.train_one_iteration(self, train_X, train_Y) #Training each epoch
             
#              loss_record.append(loss[0][0]) #Finding and recording loss after  each epoch
#              if (i >= 1) and (loss_record[-2] - loss_record[-1] < threshold_to_stop): #Condition to stop the training when converged
#                  break 
#          return loss_record 
             
#      def predict(self, train_X): 
#         test_Y_predicted_array = np.zeros(train_X.shape[0]) #create the array
#         x_value = np.zeros((1, train_X.shape[1])) #create the array
        
#         for i in range(train_X.shape[0]): 
#             x_value[0, :] = train_X[i, :] 
            
#             hidden_layer_before = x_value.dot(self.weight_hidden_layer) + self.bias_hidden_layer 
#             hidden_layer = relu(hidden_layer_before) #hidden layer update
            
#             output = hidden_layer.dot(self.weight_output) + self.bias_output 
#             test_Y_predicted = relu(output) #Getting the predictions
#             test_Y_predicted_array[i] = test_Y_predicted 
         
#         return test_Y_predicted_array
 














































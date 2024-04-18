import numpy as np

def cost_function(columnname, estimated_log_price, weight, bias):
    total_error = 0.0
    for i in range(len(columnname)):
        total_error += (estimated_log_price[i] - (weight*columnname[i] + bias))**2
    return total_error / len(columnname)

def r2_score (y_test, y_predict):
    nominator = 0
    denominator=0
    for i in range(len(y_test)):
        nominator = nominator + (y_test[i]-y_predict[i])**2
    for i in range(len(y_test)):
        denominator =  denominator + (y_test[i]- y_test.mean())**2
    return 1-(nominator/denominator)    

def mean_squared_error (y_test, y_pred_lr):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + (y_test[i]-y_pred_lr[i])**2
    MSE = RSS/len(y_test)
    return MSE

def root_mean_squared_error(y_test,y_pred_lr):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + (y_test[i]-y_pred_lr[i])**2
    MSE = RSS/len(y_test)
    return np.sqrt(MSE)

def mean_absolute_error (y_test, y_pred_lr):
    total = 0
    for i in range(len(y_test)):
        total = total + abs(y_test[i]-y_pred_lr[i])
    MSE = total/len(y_test)
    return MSE

def train_test_split(x,y, seed, test_size):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)  
    
    x = x.to_numpy()
    y = y
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shffl = x[indices]
    y_shffl = y[indices]
    
    x_train = x_shffl[test_size:]
    x_test = x_shffl[:test_size]
    y_train = y_shffl[test_size:]
    y_test = y_shffl[:test_size]
    return x_train, x_test, y_train, y_test

##############DECISION TREE##################

def fit_tree(X, y, min_samples_split=10, max_depth=5, depth=0):
    num_samples, num_features = X.shape
    if num_samples < min_samples_split or depth >= max_depth:
        return np.mean(y)  


    best_feature, best_threshold = get_best_split(X, y, num_features)
    if best_feature is None:
        return np.mean(y)  


    left_indices = X[:, best_feature] <= best_threshold
    right_indices = X[:, best_feature] > best_threshold
    left_child = fit_tree(X[left_indices], y[left_indices], min_samples_split, max_depth, depth + 1)
    right_child = fit_tree(X[right_indices], y[right_indices], min_samples_split, max_depth, depth + 1)

    return best_feature, best_threshold, left_child, right_child


def get_best_split(X, y, num_features):
    min_error = float('inf')
    best_feature = None
    best_threshold = None
    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left = y[X[:, feature_index] <= threshold]
            right = y[X[:, feature_index] > threshold]
            error = calculate_error(left, right)
            if error < min_error:
                min_error = error
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold


def calculate_error(left, right):
    left_error = np.var(left) * len(left)
    right_error = np.var(right) * len(right)
    return left_error + right_error


def predict_tree(model, samples):

    if samples.ndim > 1:
        return [predict_single_sample(model, sample) for sample in samples]
    else:
        return predict_single_sample(model, samples)


def predict_single_sample(model, sample):
    if isinstance(model, tuple): 
        feature, threshold, left_child, right_child = model
        if sample[feature] <= threshold:
            return predict_single_sample(left_child, sample)
        else:
            return predict_single_sample(right_child, sample)
    else:
        return model 
    



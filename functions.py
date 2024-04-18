import numpy as np

# def cost_function(columnname, estimated_log_price, weight, bias): ##dosyada duracak ama rapora koymacaz
#     total_error = 0.0
#     for i in range(len(columnname)):
#         total_error += (estimated_log_price[i] - (weight*columnname[i] + bias))**2
#     return total_error / len(columnname)

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

def train_test_split(x,y, seed, test_size):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)     
    x = x.to_numpy()
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shffl = x[indices]
    y_shffl = y[indices]
    
    x_train = x_shffl[test_size:]
    x_test = x_shffl[:test_size]
    y_train = y_shffl[test_size:]
    y_test = y_shffl[:test_size]
    return x_train, x_test, y_train, y_test

#############################DECISION TREE#####################################

def fit_tree(x_train, y_train, min_samples, max_depth, depth=0):
    num_samples, num_features = x_train.shape
    if num_samples < min_samples or depth >= max_depth:
        return np.mean(y_train)  

    best_feature, best_threshold = get_best_split(x_train, y_train, num_features)
    if best_feature is None:
        return np.mean(y_train)  

    left_indices = x_train[:, best_feature] <= best_threshold
    right_indices = x_train[:, best_feature] > best_threshold
    left_child = fit_tree(x_train[left_indices], y_train[left_indices], min_samples, max_depth, depth + 1)
    right_child = fit_tree(x_train[right_indices], y_train[right_indices], min_samples, max_depth, depth + 1)

    return best_feature, best_threshold, left_child, right_child

def node_RSS(left, right):
    left_error = np.var(left) * len(left)
    right_error = np.var(right) * len(right)
    return left_error + right_error

def get_best_split(x_train, y_train, num_features):
    min_error = float('inf')
    best_feature = None
    best_threshold = None
    for feature_index in range(num_features):
        thresholds = np.unique(x_train[:, feature_index])
        for threshold in thresholds:
            left = y_train[x_train[:, feature_index] <= threshold]
            right = y_train[x_train[:, feature_index] > threshold]
            error = node_RSS(left, right)
            if error < min_error:
                min_error = error
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold

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



    






#############my desprite trialss
# def RSS_dt(child,mean):
#     RSS = 0
#     for i in range(len(child)):
#         RSS = RSS + (child[i]-mean)**2       
#     return RSS

# def node_RSS(left, right):
#     left_error = RSS_dt(left,np.mean(left)) * len(left)
#     right_error = RSS_dt(right,np.mean(right)) * len(right)
#     return left_error + right_error


















































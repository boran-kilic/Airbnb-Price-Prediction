
#############################DECISION TREE#####################################

def fit_tree(x_train, y_train, min_samples, max_depth, depth=0):
    num_samples, num_features = x_train.shape
    if num_samples < min_samples or depth >= max_depth:
        return np.mean(y_train)  

    best_ft, best_thr = best_split(x_train, y_train, num_features)
    if best_ft is None:
        return np.mean(y_train)  

    left_idxs = x_train[:, best_ft] <= best_thr
    right_idxs = x_train[:, best_ft] > best_thr
    left_child = fit_tree(x_train[left_idxs], y_train[left_idxs], min_samples, max_depth, depth + 1)
    right_child = fit_tree(x_train[right_idxs], y_train[right_idxs], min_samples, max_depth, depth + 1)

    return best_ft, best_thr, left_child, right_child

def DT_RSS(child):
    RSS = 0
    mean = np.mean(child) 
    RSS =sum((x - mean) ** 2 for x in child)
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
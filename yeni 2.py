import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functions import *

def fit_tree(X, y, min_samples_split=10, max_depth=5, depth=0):
    num_samples, num_features = X.shape
    if num_samples < min_samples_split or depth >= max_depth:
        return np.mean(y)  # Return the mean value of y for leaf nodes

    # Otherwise, try to find the best split
    best_feature, best_threshold = get_best_split(X, y, num_features)
    if best_feature is None:
        return np.mean(y)  # If no split was found, return the mean value

    # Split the dataset
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
    # Check if samples contain more than one record
    if samples.ndim > 1:
        return [predict_single_sample(model, sample) for sample in samples]
    else:
        return predict_single_sample(model, samples)


def predict_single_sample(model, sample):
    if isinstance(model, tuple):  # Check if node is not a leaf
        feature, threshold, left_child, right_child = model
        if sample[feature] <= threshold:
            return predict_single_sample(left_child, sample)
        else:
            return predict_single_sample(right_child, sample)
    else:
        return model  # Return the mean value at the leaf
        
data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x,y,seed = 42, test_size = 0.2)



#EXAMPLE
start_time = time.time()
tree_model = fit_tree(x_train, y_train, 200, 10)
y_predict = predict_tree(tree_model, x_test)
print("It has taken {0} seconds to train the network".format(time.time() - start_time))

#
# print(y_test.tolist())
# print(y_predict)


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_predict)
mse = metrics.mean_squared_error(y_test, y_predict)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
r2 = metrics.r2_score(y_test, y_predict)


print('\nMean Absolute Error of Linear Regression   : ', mae)
print('\nMean Squarred Error of Linear Regression     : ', mse)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse)
print('\nR2 Score of Linear Regression                : ', r2)

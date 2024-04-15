import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from time import time 
from last_functions import *


# Read the dataset 
# dataset = pd.read_csv("preprocessed_diamond_data.csv") 
dataset = pd.read_csv("proccessed_airbnb_data.csv") 
#print(dataset.head()) 
Y = np.array(dataset.log_price) 
X = np.array(dataset.drop('log_price', axis=1)) 
print("Dataset has {0} entries with {1} features".format(X.shape[0], X.shape[1]))


def split_dataset(features, labels, split_size=0.75): 
    dataset_length = labels.shape[0] 
    indices = np.arange(0, dataset_length) # [0, 1, 2, 3, 4, ...] 
    np.random.shuffle(indices) # get random indices 
    
    train_size = int(np.round((split_size-0.1) * dataset_length)) 
    validation_size = int(np.round(0.2 * dataset_length)) 
    test_size = dataset_length - train_size - validation_size 
    
    train_indices = indices[0: train_size] # these are random since "indices" is shuffled 
    validation_indices = indices[train_size: train_size + validation_size] 
    test_indices = indices[train_size + validation_size: ] 
    
    train_features = features[train_indices, :] 
    train_labels = labels[train_indices] 
    validation_features = features[validation_indices, :]
    
    validation_labels = labels[validation_indices] 
    test_features = features[test_indices, :] 
    test_labels = labels[test_indices]     
    return train_features, train_labels, validation_features, validation_labels, test_features, test_labels

class Split_Rule(): 
    def __init__(self): 
        self.left = None 
        self.right = None 
        self.feature = None 
        self.threshold = None 
        self.prediction = None 
        
def residual_squared_sum(node_1, node_2): 
    rss_1 = np.sum(np.square(node_1 - np.mean(node_1))) 
    rss_2 = np.sum(np.square(node_2 - np.mean(node_2))) 
    return rss_1 + rss_2 

def rule_finder(train_features, train_labels): 
    node_rule = Split_Rule() 
    rule_rss = np.inf 
    num_of_features = np.shape(train_features)[1] 
    for feature_index in range(num_of_features): 
        feature_vals = np.unique(train_features[:, feature_index]) 
        feature_vals = np.sort(feature_vals) 
        feature_vals = feature_vals[1: -1] # discard the first and last elements 
        for val in feature_vals: 
            right_node_indices = train_features[:, feature_index] > val 
            left_node_indices = ~right_node_indices 
            right_node_labels = train_labels[right_node_indices] 
            left_node_labels = train_labels[left_node_indices] 
            node_rss = residual_squared_sum(right_node_labels, left_node_labels) 
            if rule_rss > node_rss: 
                rule_rss = node_rss 
                node_rule.feature = feature_index 
                node_rule.threshold = val 
    return node_rule

def train_regression_tree(train_features, train_labels, current_depth, max_depth=20, min_entries_per_node=10): 
    num_of_entries = np.shape(train_features)[0] # exit condition of recursion 
    if (num_of_entries < min_entries_per_node) or (current_depth >= max_depth): 
        node_rule = Split_Rule() 
        node_rule.prediction = np.mean(train_labels) 
        return node_rule 
    node_rule = rule_finder(train_features, train_labels)
    if node_rule.threshold is None or node_rule.feature is None: 
        node_rule.prediction = np.mean(train_labels) 
        return node_rule 
    
    right_indices = train_features[:, node_rule.feature] > node_rule.threshold 
    left_indices = ~right_indices 
    left_train_features = train_features[left_indices, :] 
    left_train_labels = train_labels[left_indices] 
    right_train_features = train_features[right_indices, :] 
    right_train_labels = train_labels[right_indices] 
    current_depth = current_depth + 1 
    node_rule.left = train_regression_tree(left_train_features, left_train_labels, current_depth, max_depth) 
    node_rule.right = train_regression_tree(right_train_features, right_train_labels, current_depth, max_depth) 
    return node_rule 

def predict(data, tree_rules): 
    while (tree_rules.prediction is None): 
        feature = tree_rules.feature 
        threshold = tree_rules.threshold 
        if data[feature] > threshold: 
            tree_rules = tree_rules.right 
        else: 
            tree_rules = tree_rules.left 
    return tree_rules.prediction 
    
def predict_all(test_features, tree_rules): 
    yHat = np.zeros(np.shape(test_features)[0]) 
    for i in range(np.shape(test_features)[0]): 
        yHat[i] = predict(test_features[i, :], tree_rules) 
        return yHat 
    
def score_r2(y_predicted, y_label): 
    numerator = np.sum(np.square(y_label - y_predicted)) 
    denominator = np.sum(np.square(y_predicted - np.mean(y_label))) 
    return (1 - numerator / denominator) 

train_features, train_labels, validation_features, validation_labels, test_features, test_labels = split_dataset(X, Y, split_size=0.8) 
# =============================================================================
# print(train_features.shape) 
# print(train_labels.shape) 
# print(validation_features.shape)
# print(validation_labels.shape) 
# print(test_features.shape) 
# print(test_labels.shape) 
# =============================================================================

start_time = time() 
max_depth_arr = np.array([5, 10]) 
min_entries_per_node_arr = np.array([4, 6])
 
validation_mse = np.zeros((max_depth_arr.shape[0], min_entries_per_node_arr.shape[0])) 
for i in range(max_depth_arr.shape[0]): 
    for j in range(min_entries_per_node_arr.shape[0]): 
        tree_rules = train_regression_tree(train_features, train_labels, 0, max_depth=max_depth_arr[i], min_entries_per_node=min_entries_per_node_arr[j]) 
        yHat_validation = predict_all(validation_features, tree_rules) 
        validation_mse[i, j] = np.mean(np.square(yHat_validation - validation_labels)) 
print("Validation MSE are: \n") 
print(validation_mse) 

max_depth_index = validation_mse.argmin(axis=0)[0] 
min_entries_per_node_index = validation_mse.argmin(axis=1)[0] 
max_depth_val = max_depth_arr[max_depth_index] 
min_entries_per_node_val = min_entries_per_node_arr[min_entries_per_node_index] 
print("max_depth is: {0}".format(max_depth_val)) 
print("min_entries_per_node is: {0}".format(min_entries_per_node_val)) 

tree_rules = train_regression_tree(train_features, train_labels, 0, max_depth=max_depth_val, min_entries_per_node=min_entries_per_node_val) 
print("It has taken {0} seconds to train the algorithm".format(time() - start_time)) 

yHat = predict_all(test_features, tree_rules) 
print("The R2 Score over the test set is: ", (score_r2(yHat, test_labels)))
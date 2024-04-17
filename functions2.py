# Decision Tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value
class DecisionTree():
    def __init__(self):
        self.first = None  # first (left) node
        self.sec = None  # second (right) node
        self.feature = None
        self.threshold_to_stop = None  # threshold value
        self.prediction_y = None



def RSS(first_node, sec_node):
    first_rss = np.sum(np.square(first_node - np.mean(first_node)))
    sec_rss = np.sum(np.square(sec_node - np.mean(sec_node)))
    rss = first_rss + sec_rss
    return rss


def basepredict_dt(test, dt):  # predicts the initial condition
    while (dt.prediction_y is None):
        feature = dt.feature
        threshold_to_stop = dt.threshold_to_stop
        if test[feature] < threshold_to_stop:
            dt = dt.sec
        else:
            dt = dt.first
    return dt.prediction_y


def predict_set(test_X, dt):  # next prediction acc. to previous
    test_Y_predicted_array = np.zeros(np.shape(test_X)[0])

    for i in range(np.shape(test_X)[0]):
        test_Y_predicted_array[i] = basepredict_dt(test_X[i, :], dt)
    return test_Y_predicted_array


def training_dt(train_X, train_Y, depth, max_depth=10):
    if depth >= max_depth:
        rule = DecisionTree()
        rule.prediction_y = np.mean(train_Y)
    return rule

    rule = DecisionTree()
    rule_rss = np.inf
    feature_numb = np.shape(train_X)[1]

    for feature_no in range(feature_numb):
        feature_value = np.unique(train_X[:, feature_no])
        feature_value = np.sort(feature_value)
        feature_value = feature_value[1: -1]
        for val in feature_value:
            sec_datum = train_X[:, feature_no] > val
            sec_Y = train_Y[sec_datum]
            first_datum = ~sec_datum
            first_Y = train_Y[first_datum]
            node_rss = RSS(sec_Y, first_Y)
            if rule_rss > node_rss:
                rule_rss = node_rss
                rule.feature = feature_no
                rule.threshold_to_stop = val

        if rule.threshold_to_stop is None or rule.feature is None:
            rule.prediction_y = np.mean(train_Y)
    return rule

    sec_datum = train_X[:, rule.feature] > rule.threshold_to_stop
    sec_train_X = train_X[sec_datum, :]
    sec_train_Y = train_Y[sec_datum]
    first_datum = ~sec_datum
    first_train_X = train_X[first_datum, :]
    first_train_Y = train_Y[first_datum]
    depth = depth + 1
    rule.sec = training_dt(sec_train_X, sec_train_Y, depth, max_depth)
    rule.first = training_dt(first_train_X, first_train_Y, depth, max_depth)
    return rule

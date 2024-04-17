# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:35:46 2024

@author: User
"""

#Importing Libraries and Functions defined
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from last_functions import *
#Getting the data
dataset = pd.read_csv("proccessed_airbnb_data.csv")
Y = np.array(dataset.log_price) 
X = np.array(dataset.drop('log_price', axis=1)) 
print("Dataset has {0} entries with {1} features".format(X.shape[0], X.shape[1]))

#Separating the dataset
length = Y.shape[0] #gives the number of rows 
datum = np.arange(0, length)
test_length = int(np.round(0.25 * length))
print("Test size: {0}".format(test_length))
validation_length = int(np.round(0.2 * length))
print("Validation size: {0}".format(validation_length))
train_length = length - test_length - validation_length
print("Train size: {0}".format(train_length))
test_datum = datum[0: test_length]
validation_datum = datum[test_length: test_length + validation_length]
train_datum = datum[test_length + validation_length:]
test_Y = Y[test_datum]
test_X = X[test_datum, :]
validation_Y = Y[validation_datum]
validation_X = X[validation_datum, :]
train_Y = Y[train_datum]
train_X = X[train_datum, :]
start_time = time()
dt = training_dt(train_X, train_Y, 0, max_depth = 10)
print("Maximum depth of node is: 10")
print("It has taken {0} seconds to train the network".format(time() - start_time))
test_Y_predicted = predict_set(test_X, dt)
print("The Score: ", (RSS(test_Y_predicted, test_Y)))
plt.figure()
plt.plot(test_Y, label="line1")
plt.plot(test_Y_predicted, label="line2")

# from sklearn.tree import plot_tree
# plt.figure(figsize=(20,10))
# plot_tree(dt, filled=True)
# plt.show()
# import graphviz

# class Node:
#     def __init__(self, question=None, true_branch=None, false_branch=None, label=None):
#         self.question = question
#         self.true_branch = true_branch
#         self.false_branch = false_branch
#         self.label = label


# def print_tree(node, depth=0, label="Root"):
#     # Base case: we've reached a leaf
#     if node.label is not None:
#         print(f"{' '*depth*2}{label} - Leaf: {node.label}")
#         return

#     # Print the question at this node
#     print(f"{' '*depth*2}{label} - Q: {node.question}")

#     # Call this function recursively on the true branch
#     print_tree(node.true_branch, depth+1, 'True')

#     # Call this function recursively on the false branch
#     print_tree(node.false_branch, depth+1, 'False')

# def visualize_tree(node, dot=None):
#     if dot is None:
#         dot = graphviz.Digraph()

#     if node.label is not None:  # Leaf node
#         dot.node(name=str(id(node)), label=node.label, shape='box')
#     else:
#         dot.node(name=str(id(node)), label=node.question)
#         # True branch
#         dot.edge(str(id(node)), str(id(node.true_branch)), label='True')
#         visualize_tree(node.true_branch, dot)
#         # False branch
#         dot.edge(str(id(node)), str(id(node.false_branch)), label='False')
#         visualize_tree(node.false_branch, dot)

#     return dot

# # Build and print the tree

# print_tree(dt)

# # Visualize the tree
# dot = visualize_tree(dt)
# dot.render('decision_tree', format='png', cleanup=True)
# print("Decision tree graph saved as 'decision_tree.png'")
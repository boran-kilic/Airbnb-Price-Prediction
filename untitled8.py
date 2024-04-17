import numpy as np
import pandas as pd

data = pd.read_csv("proccessed_airbnb_data.csv")
data = data[:100]
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

class DTR():
    def __init__(self, min_samples_split, max_depth):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    def build_tree(self, dataset):
        ''' Iteratively build the tree using a stack '''
        root = None
        stack = [{
            'dataset': dataset,
            'depth': 0,
            'parent_node': None,
            'is_left': True
        }]

        while stack:
            node_info = stack.pop()
            dataset = node_info['dataset']
            current_depth = node_info['depth']
            parent_node = node_info['parent_node']
            is_left = node_info['is_left']

            X, Y = dataset[:, :-1], dataset[:,-1]
            num_samples, num_features = np.shape(X)

            # Check stopping condition
            if num_samples < self.min_samples_split or current_depth >= self.max_depth:
                leaf_value = np.mean(Y)  # Compute leaf node value
                new_node = Node(value=leaf_value)
                if parent_node:
                    if is_left:
                        parent_node.left = new_node
                    else:
                        parent_node.right = new_node
                continue

            # Find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["var_red"] > 0:
                # Create new decision node
                new_node = Node(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    var_red=best_split["var_red"]
                )
                if parent_node is None:
                    root = new_node  # set root if it's the first node
                else:
                    if is_left:
                        parent_node.left = new_node
                    else:
                        parent_node.right = new_node

                # Add child nodes to stack
                stack.append({'dataset': best_split["dataset_right"], 'depth': current_depth + 1, 'parent_node': new_node, 'is_left': False})
                stack.append({'dataset': best_split["dataset_left"], 'depth': current_depth + 1, 'parent_node': new_node, 'is_left': True})
            else:
                # If no variance reduction, create a leaf node
                leaf_value = np.mean(Y)
                new_node = Node(value=leaf_value)
                if parent_node:
                    if is_left:
                        parent_node.left = new_node
                    else:
                        parent_node.right = new_node

        return root        

        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    def split(self, dataset, feature_index, threshold):
        '''Use numpy boolean indexing for efficient splitting.'''
        left_mask = dataset[:, feature_index] <= threshold
        right_mask = dataset[:, feature_index] > threshold
        return dataset[left_mask], dataset[right_mask]

    def get_best_split(self, dataset, num_samples, num_features):
        '''Optimized function to find the best split.'''
        best_split = {}
        max_var_red = -float("inf")
        y = dataset[:, -1]
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            # Using quantiles to reduce the number of thresholds
            thresholds = np.quantile(feature_values, np.linspace(0, 1, num=10))
            for threshold in thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red > max_var_red:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "var_red": curr_var_red
                        }
                        max_var_red = curr_var_red
        return best_split    

    def rss(self, column):
        return np.sum((column - np.mean(column)) ** 2)
    
    def variance_reduction(self, parent, l_child, r_child):
        total_sum = np.sum(parent)
        total_sq_sum = np.sum(parent**2)
        total_count = len(parent)
        
        def stats(child):
            sum_ = np.sum(child)
            sq_sum = np.sum(child**2)
            count = len(child)
            mean = sum_ / count
            return count * mean**2 - sq_sum
        
        parent_var = total_count * (total_sum / total_count)**2 - total_sq_sum
        left_var = stats(l_child)
        right_var = stats(r_child)
        
        return parent_var - (left_var + right_var)
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root
    
        if tree is None:
            print("None Node")
            return
    
        if tree.value is not None:
            print(tree.value)
        else:
            print(f"X_{tree.feature_index} <= {tree.threshold}")
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + indent)

    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    

X = data.iloc[:, 1:].values  # This will take all columns except the first one for X
Y = data.iloc[:, 0].values.reshape(-1,1)  # This will take the first column for Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)



regressor = DTR(min_samples_split=25, max_depth=5)
regressor.fit(X_train,Y_train)
regressor.print_tree()
# Check if data arrays are not empty and have matching lengths
if X_train.size > 0 and Y_train.size > 0 and len(X_train) == len(Y_train):
    regressor.fit(X_train, Y_train)  # Fit the model
    print("Model fitted successfully.")
else:
    print("Error: Check your training data for inconsistencies.")
    


Y_pred = regressor.predict(X_test) 
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
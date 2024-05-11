import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *

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


        
data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.2)

start_time = time.time()
tree_model = fit_tree(x_train, y_train, 500, 10)
end_time = time.time()
training_time = end_time - start_time  
print(f"Training time of the decision tree: {training_time} seconds")

y_predict = np.array(predict_tree(tree_model, x_test))

# y_predict = np.exp(y_predict)
# y_test = np.exp(y_test)

plt.figure(figsize=(8, 5))  
plt.scatter(y_test, y_predict, color='lightblue') 
plt.plot(y_test, y_test, color='purple', linewidth=2) 
plt.title("Actual vs predicted Prices")  
plt.xlabel("Actual Price") 
plt.ylabel("Predicted Price")  
plt.grid(True)  
plt.show()



mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
r2 = r2_score(y_test, y_predict)

print('\nMean Absolute Error of Decision Tree: ', mae)
print('\nMean Squarred Error of Decision Tree: ', mse)
print('\nRoot Mean Squarred Error of Decision Tree: ', rmse)
print('\nR2 Score of Decision Tree: ', r2)

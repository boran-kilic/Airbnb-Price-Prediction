import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from functions import *


data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x,y,seed = 42, test_size = 0.2)

depths = []
samples_splits = []
mse_values = []


for d in range(5, 25, 2):  
    for s in range(500, 4000, 100):
        tree_model = fit_tree(x_train, y_train, s, d)
        y_predict = predict_tree(tree_model, x_test)        
        mse = mean_squared_error(y_test, y_predict)
        depths.append(d)
        samples_splits.append(s)
        mse_values.append(mse)



min_mse = np.min(mse_values)  
min_index = np.argmin(mse_values) 

print("Minimum MSE:", min_mse)
print("Achieved with max_depth =", depths[min_index], "and min_samples_split =", samples_splits[min_index])
            
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


depths = np.array(depths)
samples_splits = np.array(samples_splits)
mse_values = np.array(mse_values)


scat = ax.scatter(depths, samples_splits, mse_values, c=mse_values, cmap='viridis')

ax.set_xlabel('Max Depth')
ax.set_ylabel('Min Samples Split')
ax.set_zlabel('MSE')


cbar = fig.colorbar(scat, ax=ax, extend='neither', orientation='vertical')
cbar.set_label('MSE')

plt.title('3D Scatter Plot of MSE by Tree Depth and Min Samples Split')
plt.show()


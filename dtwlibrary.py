from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
data = pd.read_csv("proccessed_airbnb_data.csv")
# Example data (replace with your actual data)
X = data.iloc[:, 1:].values  # This will take all columns except the first one for X
Y = data.iloc[:, 0].values.reshape(-1,1)  # This will take the first column for Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Initialize the regressor
regressor = DecisionTreeRegressor(min_samples_split=7000, max_depth=10)

# Check if data arrays are not empty and have matching lengths
if X_train.size > 0 and Y_train.size > 0 and len(X_train) == len(Y_train):
    regressor.fit(X_train, Y_train)  # Fit the model
    print("Model fitted successfully.")
else:
    print("Error: Check your training data for inconsistencies.")

# Now you can safely visualize the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(regressor, filled=True)
plt.show()
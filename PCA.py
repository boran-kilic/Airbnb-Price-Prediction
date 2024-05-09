import pandas as pd
import numpy as np

data = pd.read_csv('proccessed_airbnb_data.csv')

dataarray = data.to_numpy()

cov_mat = np.cov(dataarray, rowvar=False)

eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

# Step 4: Sort the eigenvectors by decreasing eigenvalues
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:,sorted_index]

# Step 5: Select a subset from the rearranged Eigen vectors
num_components = 7  # This is less than or equal to the original dimensions
selected_eigenvectors = sorted_eigenvectors[:, :num_components]

# Step 6: Transform the data
transformed_data = np.dot(dataarray, selected_eigenvectors)
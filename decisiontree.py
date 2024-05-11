import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *
        
data = pd.read_csv('proccessed_airbnb_data.csv')

x = data.drop(["log_price"], axis=1)
y = data['log_price']
x_train, x_test, y_train, y_test, x_validation, y_validation = train_test_split(x,y,seed = 42, test_size = 0.2, validation_size=0.2)

start_time = time.time()
tree_model = fit_tree(x_train, y_train, 500, 10)
end_time = time.time()
training_time = end_time - start_time  
print(f"Training time of the decision tree: {training_time} seconds")

y_predict = np.array(predict_tree(tree_model, x_test))

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

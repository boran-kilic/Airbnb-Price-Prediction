import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(r"C:\Users\User\DUNYANIN EN IYI PROJESI\Airbnb-Price-Prediction\Airbnb_Data.csv")
#data = pd.read_csv("Airbnb_Data.csv")
#print(data.head)
#data.head()
new_data = data.drop(["description","name","thumbnail_url","zipcode","latitude"], axis='columns',inplace=True)
print(new_data)

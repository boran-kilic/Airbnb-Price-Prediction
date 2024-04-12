import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Airbnb_Data.csv")
#data = pd.read_csv(r"C:\Users\User\DUNYANIN EN IYI PROJESI\Airbnb-Price-Prediction\Airbnb_Data.csv")
new_data = data.drop(["description","name","thumbnail_url","zipcode","latitude","longitude","neighbourhood"], axis='columns')
        
new_data.last_review.fillna(method="ffill",inplace=True)
new_data.first_review.fillna(method="ffill",inplace=True)
new_data.host_since.fillna(method="ffill",inplace=True)
new_data["bathrooms"] = new_data['bathrooms'].fillna(round(new_data["bathrooms"].median()))
new_data["review_scores_rating"] = new_data["review_scores_rating"].fillna(0)
new_data["bedrooms"] = new_data['bedrooms'].fillna((new_data["bathrooms"].median()))
new_data["beds"] = new_data["beds"].fillna((new_data["bathrooms"].median()))

amenities_count = []
for i in new_data["amenities"]:
    amenities_count.append(len(i))
    
new_data["amenities"] = amenities_count

categorical_col = []
numerical_col = []
for column in new_data.columns:
    
    if new_data[column].dtypes != "float64" and new_data[column].dtypes != "int64":
        categorical_col.append(column)
    else:
        numerical_col.append(column)

from sklearn.preprocessing import LabelEncoder  # this part will be changed because using this library is illegal
le = LabelEncoder()

for col in categorical_col:
    new_data[col] = le.fit_transform(new_data[col])     #upto there other parts can(and will) be modified but legal
    

plt.figure(figsize = (40,40))
sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.show()

print(new_data.columns)


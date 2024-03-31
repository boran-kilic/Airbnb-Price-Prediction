import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("Airbnb_data.csv")
        
data.last_review.fillna(method="ffill",inplace=True)
data.first_review.fillna(method="ffill",inplace=True)
data.host_since.fillna(method="ffill",inplace=True)
data["bathrooms"] = data['bathrooms'].fillna(round(data["bathrooms"].median()))
data["review_scores_rating"] = data["review_scores_rating"].fillna(0)
data["bedrooms"] = data['bedrooms'].fillna((data["bathrooms"].median()))
data["beds"] = data["beds"].fillna((data["bathrooms"].median()))

amenities_count = []
for i in data["amenities"]:
    amenities_count.append(len(i))
    
data["amenities"] = amenities_count

categorical_col = []
numerical_col = []
for column in data.columns:
    
    if data[column].dtypes != "float64" and data[column].dtypes != "int64":
        categorical_col.append(column)
    else:
        numerical_col.append(column)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in categorical_col:
    data[col] = le.fit_transform(data[col])
    
pd.set_option("display.max_columns",None)
data

plt.figure(figsize = (40,40))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.show()

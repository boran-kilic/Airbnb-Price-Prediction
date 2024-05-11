import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('proccessed_airbnb_data.csv')
old_data = data

plt.figure(figsize=(6, 4)) 
sns.histplot(data['log_price'],bins = len(data['log_price'].unique()), kde=False)
plt.title("Distribution of log_price")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
    
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(data.columns[1:]):
    sns.histplot(data[col],bins = len(data[col].unique()), kde=False, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


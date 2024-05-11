import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def target_encoding(new_data, columnname):
    mean_prices = new_data.groupby(columnname)['log_price'].mean()
    new_data[columnname] = new_data[columnname].replace(mean_prices)
    return new_data

data = pd.read_csv("Airbnb_Data.csv")

categorical_col = []
numerical_col = []
for column in data.columns:
    
    if data[column].dtypes == "float64" or data[column].dtypes == "int64":
        numerical_col.append(column)
    else:
        categorical_col.append(column)
print('\n')
print(len(categorical_col))
print(len(numerical_col))

############################ non-usable columns dropped ############################
new_data = data.drop(["id","description","name","thumbnail_url","zipcode"], axis='columns')
number_of_nans_per_column = new_data.isna().sum()
print('\n')
print(number_of_nans_per_column)

############################ columns with too many nan values dropped ################
new_data = new_data.drop(['first_review','host_response_rate','last_review','review_scores_rating', ],axis = 1)

number_of_nans_per_column = new_data.isna().sum()
print('\n')
print(number_of_nans_per_column)

################## categorical columns converted to numerical ########################## 
today = pd.to_datetime('today')
new_data['host_since'] = pd.to_datetime(new_data['host_since'])
new_data['host_since'] = (today - new_data['host_since']).dt.days

#categories with binary values
new_data['cleaning_fee'] = new_data['cleaning_fee'].replace({True: 1, False: 0})        
new_data['instant_bookable'] = new_data['instant_bookable'].replace({'t': 1, 'f': 0})
new_data['host_has_profile_pic'] = new_data['host_has_profile_pic'].replace({'t': 1, 'f': 0})
new_data['host_identity_verified'] = new_data['host_identity_verified'].replace({'t': 1, 'f': 0})
#categories with natural order
new_data['cancellation_policy'] = new_data['cancellation_policy'].replace({'strict': 0,'super_strict_30': 0,
                                                                           'super_strict_60': 0, 'moderate': 1,'flexible': 2})
new_data['room_type'] = new_data['room_type'].replace({'Entire home/apt': 2, 'Private room': 1,'Shared room': 0})

#target encoding for some categories
new_data = target_encoding(new_data,'city')
new_data = target_encoding(new_data,'property_type')
new_data = target_encoding(new_data,'bed_type')
new_data = target_encoding(new_data,"neighbourhood")


################################## NaN values are handled #######################################      
#some rows with nan values for some features are dropped
new_data = new_data[new_data['log_price'] != 0]
new_data = new_data.dropna(subset=['host_has_profile_pic'])
new_data = new_data.dropna(subset=['host_identity_verified'])
new_data = new_data.dropna(subset=['host_since'])
new_data = new_data.dropna(subset=['neighbourhood'])

#nan values filled with mean values
new_data['bathrooms'].fillna(round(new_data["bathrooms"].mean()),inplace=True)
new_data['bedrooms'].fillna(round(new_data["bedrooms"].mean()),inplace=True)
new_data["beds"].fillna(round(new_data["beds"].mean()),inplace=True)

###new feature defined
amenities_count = []
for i in new_data["amenities"]:
    amenities_count.append(len(i.split(',')))
    
new_data["amenities"] = amenities_count

number_of_nans_per_column = new_data.isna().sum()
print("\nAfter\n" )
print(number_of_nans_per_column)


plt.figure(figsize = (40,30))
sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.subplots_adjust(left=0.2, bottom=0.3)
plt.show()

new_data = new_data.drop(['host_since','latitude', 'longitude',
                          'host_has_profile_pic', 'host_identity_verified', 
                          'instant_bookable','number_of_reviews'], axis='columns')

number_of_nans_per_column = new_data.isna().sum()
print("\nAfter\n" )
print(number_of_nans_per_column)

plt.figure(figsize = (20,10))
sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.subplots_adjust(left=0.2, bottom=0.3)
plt.show()


#################deleting outliers####################
columns_to_delete_outliers = ["log_price","amenities","neighbourhood"]

for column in columns_to_delete_outliers:
    Q1 = new_data[column].quantile(0.25)
    Q3 = new_data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    condition = ~((new_data[column] < (Q1 - 1.5 * IQR)) | (new_data[column] > (Q3 + 1.5 * IQR)))
    
    new_data = new_data[condition]

########################## standardization #####################################

# columns_to_standardize = ["log_price","property_type","room_type","amenities",
#                         "accommodates","bathrooms","bed_type","city","neighbourhood","bedrooms","beds"]
columns_to_standardize = ["property_type","room_type","amenities",
                        "accommodates","bathrooms","bed_type","city","neighbourhood","bedrooms","beds"]


for column in columns_to_standardize:
    # col_min = np.min(new_data[column])
    # col_max = np.max(new_data[column]) 
    # new_data[column] = (new_data[column] - col_min) / (col_max - col_min)
    col_mean = np.mean(new_data[column])
    col_std = np.std(new_data[column])
    new_data[column] = (new_data[column] - col_mean) / (col_std)
    
new_data.to_csv('proccessed_airbnb_data.csv', index=False)   

plt.figure(figsize = (20,10))
sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.subplots_adjust(left=0.2, bottom=0.3)
plt.show()




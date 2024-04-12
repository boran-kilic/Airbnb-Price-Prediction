import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Airbnb_Data.csv")
# data = pd.read_csv(r"C:\Users\User\DUNYANIN EN IYI PROJESI\Airbnb-Price-Prediction\Airbnb_Data.csv")

########################################################non-usable columns dropped
new_data = data.drop(["id","description","name","thumbnail_url","zipcode",
                      "latitude","longitude","neighbourhood","first_review",
                      "host_since","last_review"], axis='columns')

# number_of_nans_per_column = new_data.isna().sum()
# print("\nBefore\n")
# print(number_of_nans_per_column)

############################################ some non numerical values converted to numerical

new_data['cleaning_fee'] = new_data['cleaning_fee'].replace({True: 1, False: 0})        
new_data['instant_bookable'] = new_data['instant_bookable'].replace({'t': 1, 'f': 0})
new_data['host_has_profile_pic'] = new_data['host_has_profile_pic'].replace({'t': 1, 'f': 0})
new_data['host_identity_verified'] = new_data['host_identity_verified'].replace({'t': 1, 'f': 0})
new_data['cancellation_policy'] = new_data['cancellation_policy'].replace({'strict': 0,'super_strict_30': 0,
                                                                           'super_strict_60': 0, 'moderate': 1,'flexible': 2})
new_data['host_response_rate'] = new_data['host_response_rate'].str.replace('%', '').astype(float) / 100

new_data['room_type'] = new_data['room_type'].replace({'Entire home/apt': 2, 'Private room': 1,'Shared room': 0})

#########################################################################NaN values filled        
# Assuming 'data' is your DataFrame
new_data = new_data[new_data['log_price'] != 0]

# This will update the DataFrame 'data' to only include rows where log_price is not equal to 0


#new_data.last_review.fillna(method="ffill",inplace=True)
#new_data.first_review.fillna(method="ffill",inplace=True)
#new_data.host_since.fillna(method="ffill",inplace=True)

# a = 0
# for i in range(74111):
#     if data.host_has_profile_pic[i] == "t":
#        a = a + 1
# print(a)
# # a = 73697 # number of hosts having a profile pic
# # 74111 - a = 414
new_data["host_has_profile_pic"] = new_data['host_has_profile_pic'].fillna(1)


# b = 0
# for i in range(74111):
#     if data.host_identity_verified[i] == "t":
#        b = b+ 1
# print(b)

# b = 49748 #number of hosts that identified themselves
# 74111-b = 24363
new_data["host_identity_verified"] = new_data['host_identity_verified'].fillna(1)
new_data["bathrooms"] = new_data['bathrooms'].fillna(round(new_data["bathrooms"].mean()))
new_data["review_scores_rating"] = new_data["review_scores_rating"].fillna(0)
new_data["host_response_rate"] = new_data["host_response_rate"].fillna((new_data["host_response_rate"].mean()))
new_data["bedrooms"] = new_data['bedrooms'].fillna(round(new_data["bedrooms"].mean()))
new_data["beds"] = new_data["beds"].fillna(round(new_data["beds"].mean()))

###new feature defined
amenities_count = []
for i in new_data["amenities"]:
    amenities_count.append(len(i.split(',')))
    
new_data["amenities"] = amenities_count




# number_of_nans_per_column = new_data.isna().sum()
# print("\nAfter\n" )
# print(number_of_nans_per_column)


###categorical and numarical colums are identified
categorical_col = []
numerical_col = []
for column in new_data.columns:
    
    if new_data[column].dtypes != "float64" and new_data[column].dtypes != "int64":
        categorical_col.append(column)
    else:
        numerical_col.append(column)
print('\n')
print(len(categorical_col))
print(categorical_col)
print('\nproperty_type')
print(new_data['property_type'].value_counts())
print('\nbed_type')
print(new_data['bed_type'].value_counts())
a= new_data['city'].value_counts()
city_counts = a.to_dict()
a1 = pd.Series(a , index = new_data['city'])
print('\ncity')
print(new_data['city'].value_counts())
print('\nwe need to deal with these')
        
def get_numerical_col(categorical_col):
    for i in range(len(categorical_col)):
        my_series=new_data[categorical_col[i]].value_counts()
        my_dictionary=my_series.to_dict()
        sortedDict = sorted(my_dictionary)
        for j in len(sortedDict):
            string= sortedDict[j]
            new_data[categorical_col[i]] = new_data[categorical_col[i]].replace({string : j})
return   

       
  


from sklearn.preprocessing import LabelEncoder  # this part will be changed because using this library is illegal
le = LabelEncoder()

for col in categorical_col:
    new_data[col] = le.fit_transform(new_data[col])     #upto there other parts can(and will) be modified but legal
    
# Convert the DataFrame to a CSV file
new_data.to_csv('proccessed_airbnb_data.csv', index=False)

# plt.figure(figsize = (30,30))
# sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
# plt.show()

# print(new_data.columns)

# plt.figure()
# plt.title("Color")
# plt.hist(new_data.)
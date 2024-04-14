import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_numerical_col(new_data, columnname):
    mean_prices = new_data.groupby(columnname)['log_price'].mean()
    sorted_categories = mean_prices.sort_values().index.tolist()
    category_to_integer = {category: i for i, category in enumerate(sorted_categories)}
    new_data[columnname] = new_data[columnname].map(category_to_integer)    
    return new_data

data = pd.read_csv("Airbnb_Data.csv")
categorical_col = []
numerical_col = []
for column in data.columns:
    
    if data[column].dtypes != "float64" and data[column].dtypes != "int64":
        categorical_col.append(column)
    else:
        numerical_col.append(column)
print('\n')
print(len(categorical_col))
print(len(numerical_col))


number_of_nans_per_column = data.isna().sum()

print(number_of_nans_per_column)


# data = pd.read_csv(r"C:\Users\User\DUNYANIN EN IYI PROJESI\Airbnb-Price-Prediction\Airbnb_Data.csv")

########################################################non-usable columns dropped
new_data = data.drop(["id","description","name","thumbnail_url",
                      "neighbourhood","zipcode",'first_review','host_response_rate','last_review'
                      ,'review_scores_rating','host_since','latitude', 'longitude'], axis='columns')

# new_data = new_data.drop(['host_has_profile_pic', 'host_identity_verified',  'instant_bookable', 
#                             'number_of_reviews', ],axis = 1)

# number_of_nans_per_column = new_data.isna().sum()
# print("\nBefore\n")
# print(number_of_nans_per_column)

############################################ some non numerical values converted to numerical
# today = pd.to_datetime('today')
# new_data['host_since'] = pd.to_datetime(new_data['host_since'])
# new_data['host_since'] = (today - new_data['host_since']).dt.days

# new_data['first_review'] = pd.to_datetime(new_data['first_review'])
# new_data['first_review'] = (today - new_data['first_review']).dt.days

# new_data['last_review'] = pd.to_datetime(new_data['last_review'])
# new_data['last_review'] = (today - new_data['last_review']).dt.days



new_data['cleaning_fee'] = new_data['cleaning_fee'].replace({True: 1, False: 0})        
new_data['instant_bookable'] = new_data['instant_bookable'].replace({'t': 1, 'f': 0})
new_data['host_has_profile_pic'] = new_data['host_has_profile_pic'].replace({'t': 1, 'f': 0})
new_data['host_identity_verified'] = new_data['host_identity_verified'].replace({'t': 1, 'f': 0})
new_data['cancellation_policy'] = new_data['cancellation_policy'].replace({'strict': 0,'super_strict_30': 0,
                                                                           'super_strict_60': 0, 'moderate': 1,'flexible': 2})
#new_data['host_response_rate'] = new_data['host_response_rate'].str.replace('%', '').astype(float) / 100

new_data['room_type'] = new_data['room_type'].replace({'Entire home/apt': 2, 'Private room': 1,'Shared room': 0})

new_data = make_numerical_col(new_data,'city')
new_data = make_numerical_col(new_data,'property_type')
new_data = make_numerical_col(new_data,'bed_type')
#########################################################################NaN values filled        

new_data = new_data[new_data['log_price'] != 0]


# a = 0
# for i in range(74111):
#     if data.host_has_profile_pic[i] == "t":
#        a = a + 1
# print(a)
# # a = 73697 # number of hosts having a profile pic
new_data["host_has_profile_pic"] = new_data['host_has_profile_pic'].fillna(1)


# b = 0
# for i in range(74111):
#     if data.host_identity_verified[i] == "t":
#        b = b+ 1
# print(b)

# b = 49748 #number of hosts that identified themselves


new_data['host_identity_verified'].fillna(1,inplace=True)
new_data['bathrooms'].fillna(round(new_data["bathrooms"].mean()),inplace=True)
new_data['bedrooms'].fillna(round(new_data["bedrooms"].mean()),inplace=True)
new_data["beds"].fillna(round(new_data["beds"].mean()),inplace=True)

###new feature defined
amenities_count = []
for i in new_data["amenities"]:
    amenities_count.append(len(i.split(',')))
    
new_data["amenities"] = amenities_count


# number_of_nans_per_column = new_data.isna().sum()
# print("\nAfter\n" )
# print(number_of_nans_per_column)



    
# plt.figure(figsize = (40,40))
# sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
# plt.show()



plt.figure(figsize = (20,10))
sns.heatmap(new_data.corr(), annot=True, fmt=".2f", cmap="seismic")
plt.show()

new_data.to_csv('proccessed_airbnb_data.csv', index=False)

# print(new_data.columns)

# plt.figure()
# plt.title("Color")
# plt.hist(new_data.)


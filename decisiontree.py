import pandas as pd
import numpy as np
from functions import *

data = pd.read_csv('proccessed_airbnb_data.csv')


x = data.drop(["log_price"], axis=1)
y = data['log_price'].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x,y)



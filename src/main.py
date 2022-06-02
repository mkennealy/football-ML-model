import pandas as pd
import numpy as np


#https://iq-inc.com/importerror-attempted-relative-import/
import data.data_retrieve as dr
import features.data_preprocessing as pp

#could make it object oriented so that different models can be different objects
#could make Preprocess Class

data = dr.get_data()
data = pp.feature_engineering(data)
data, X, y = pp.get_X_and_y_over_under(data,number_goals = 2.5)
X_train, y_train, X_test, y_test = pp.train_test_split(data,X,y)

print(data.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#show performance of model on different teams (perhaps certain teams are predicted better)
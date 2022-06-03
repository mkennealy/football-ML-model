import pandas as pd
import numpy as np


#https://iq-inc.com/importerror-attempted-relative-import/
import data.data_retrieve as dr
import features.data_preprocessing as pp
import models.train_model as tm

#could make it object oriented so that different models can be different objects
#could make Preprocess Class

data = dr.get_data()
data = pp.feature_engineering(data)
data, X, y = pp.get_X_and_y_over_under(data,number_goals = 2.5)
X_train, y_train, X_test, y_test = pp.train_test_split(data,X,y)

print(data.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#optimal_params = tm.get_optimal_parameters(X_train,y_train,n_trials=35)
#print(optimal_params)
optimal_params = {
                'n_estimators': 286,
                'max_depth': 6,
                'num_leaves': 18,
                'min_data_in_leaf': 70,
                'feature_fraction': 0.4,
                'lambda_l1': 10,
                'lambda_l2': 55,
                'bagging_fraction': 0.5,
                'learning_rate': 0.03438248498061834,
                'min_gain_to_split': 0.1030288398937825,
                'bagging_freq': 1,
                }

preds, preds_class = tm.train_and_predict(X_train, y_train,X_test, y_test,optimal_params)
evaluation_metrics = tm.evaluate_predictions(y_test,preds, preds_class)
#show performance of model on different teams (perhaps certain teams are predicted better)
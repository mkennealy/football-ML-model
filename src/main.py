import pandas as pd
import numpy as np


#https://iq-inc.com/importerror-attempted-relative-import/
import data.data_retrieve as dr
import features.data_preprocessing as pp

#could make it object oriented so that different models can be different objects
#could make Preprocess Class

data = dr.get_data()
data = pp.feature_engineering(data)

print(data.shape)


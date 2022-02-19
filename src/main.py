import pandas as pd
import numpy as np


#https://iq-inc.com/importerror-attempted-relative-import/
import data.data_retrieve as dr


#could make it object oriented so that different models can be different objects

data = dr.get_data()
print(data.shape)

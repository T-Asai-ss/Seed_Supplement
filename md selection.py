# Investigation on the correspondence between randomness of output due to seed value in in silico models and sampling frequency in biological tests
# Python code of molecular descriptors selection

# ## import library
# In[1]:
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

# ## input data
# In[2]:
tx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_sx_all.csv", encoding="ms932", sep=","))
ty = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_sy.csv", encoding="ms932", sep=","))

# ## boruta
# In[3]:
pip install boruta
from boruta import BorutaPy
model = RandomForestRegressor(n_jobs=-1, max_depth=5)
feat_selector = BorutaPy(model, n_estimators='auto', two_step=False, perc=100, verbose=3, random_state=42)
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
feat_selector.fit(tx.values, ty.values)
selected = feat_selector.support_

# ## output data
# In[4]:
selected = pd.DataFrame(selected)
selected.to_csv("Boruta 100.csv", encoding="shift_jis")

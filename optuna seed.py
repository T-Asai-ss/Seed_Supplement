# Investigation on the correspondence between randomness of output due to seed value in in silico models and sampling frequency in biological tests
# Python code of lightgbm models with varying Optuna_seed

# ## import library
# In[1]:
import numpy as np
import pandas as pd
import csv
import pprint
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
import random as rn
import os
warnings.filterwarnings("ignore")
def seed_everything(seed: int):
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      rn.seed(seed)        
seed_everything(42)

# ## input data
# In[2]:
tx = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_sx.csv", encoding="ms932", sep=","))
ty = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_sy.csv", encoding="ms932", sep=","))
ex = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_tx.csv", encoding="ms932", sep=","))
ey = pd.DataFrame(pd.read_csv(filepath_or_buffer="./seed_ty.csv", encoding="ms932", sep=","))

# ## optuna
# In[3]:
pip install optuna
pip install lightgbm
import optuna
import optuna.integration.lightgbm as lgb
import lightgbm as lgbm
from sklearn.model_selection import KFold
trainval = lgb.Dataset(tx, ty)
params = {'objective': 'regression',
          'metric': 'rmse',
         'deterministic':True,
        'force_col_wise':True,
         'bagging_seed': 42,
          'n_jobs':1,
         'seed': 42} 
tuner = lgb.LightGBMTunerCV(params, trainval, folds=KFold(n_splits=5, shuffle=True, random_state=42), optuna_seed=0)
tuner.run()
best_params = tuner.best_params
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

# ## prediction
# In[4]:
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
lgb_train = lgb.Dataset(tx, ty)
lgb_eval = lgb.Dataset(ex, ey)
model = lgb.cv(best_params, lgb_train, 
                  num_boost_round=100,
                  folds=KFold(n_splits=5, shuffle=True, random_state=42),
                  shuffle=True,
                  seed=42,
                  return_cvbooster=True,               
                 )
boosters = model['cvbooster'].boosters
pred_cv_tp = [tp.predict(tx)
    for tp in boosters]
pred_cv_tp = np.array(pred_cv_tp)
pred_average_tp = pred_cv_tp.mean(axis=0)
pred_ave_tp = np.array(pred_average_tp)
pred_cv_ep = [ep.predict(ex)
    for ep in boosters]
pred_cv_ep = np.array(pred_cv_ep)
pred_average_ep = pred_cv_ep.mean(axis=0)
pred_ave_ep = np.array(pred_average_ep)
cvbooster = model['cvbooster']
raw_importances = cvbooster.feature_importance(importance_type='gain')
feature_name = cvbooster.boosters[0].feature_name()
importance_df = pd.DataFrame(data=raw_importances,
                                 columns=feature_name)
print(pred_ave_tp)
print(pred_ave_ep)
print(importance_df)

# ## output data
# In[4]:
ytp = pd.DataFrame(pred_ave_tp)
ytp.to_csv("sp_seed0.csv", encoding="shift_jis")
yevp = pd.DataFrame(pred_ave_ep)
yevp.to_csv("tp_seed0.csv", encoding="shift_jis")
importance_df.to_csv("imp_seed0.csv", encoding="shift_jis")

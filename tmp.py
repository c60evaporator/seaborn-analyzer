import pytest
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
import lightgbm as lgb

from seaborn_analyzer._cv_eval_set import cross_val_score_eval_set

# Test data
TARGET_VARIABLE = 'price'  # Target variable
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
    columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
df_california = california_housing.sample(n=1000, random_state=42)  # Sampling to N=1000
y = df_california[TARGET_VARIABLE].values  # Numpy array of target variable
X = df_california[USE_EXPLANATORY].values  # Numpy array of explanatory variables
# Cross validation KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# LightGBM model
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42,
    'boosting_type': 'gbdt',  # boosting_type
    'n_estimators': 10000,
    'verbose': -1,
    'early_stopping_round': 10
    }
lgb_fit_params = {
    'eval_set':[(X, y)]
    }
lgbr = lgb.LGBMRegressor(**lgb_params)

import random

#smp = random.sample(arr, 7)

scores = cross_val_score_eval_set(
            validation_fraction=0.3,
            estimator=lgbr,
            X=X, y=y,  # Data before cross validation division
            scoring='neg_root_mean_squared_error',  # Negative RMSE
            cv=cv, fit_params=lgb_fit_params
        )

print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')
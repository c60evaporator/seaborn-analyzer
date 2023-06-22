import pytest
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier, LGBMRegressor

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from seaborn_analyzer._cv_eval_set import cross_val_score_eval_set

class TestCvClassifiers:
    # Test data
    TARGET_VARIABLE = 'species'  # Target variable
    USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
    df_iris = sns.load_dataset("iris")
    y = df_iris[TARGET_VARIABLE].values  # Numpy array of target variable
    X = df_iris[USE_EXPLANATORY].values  # Numpy array of explanatory variables
    # Cross validation KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # LightGBM model
    lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'random_state': 42,
        'boosting_type': 'gbdt',  # boosting_type
        'n_estimators': 10000,
        'verbose': -1,
        'early_stopping_round': 10
        }
    lgb_fit_params = {
        'eval_set':[(X, y)]
        }
    lgbr = LGBMClassifier(**lgb_params)

    def test_cross_val_score_cls_lgbm(self):
        scores = cross_val_score_eval_set(
            validation_fraction='test',
            estimator=self.lgbr,
            X=self.X, y=self.y,  # Data before cross validation division
            scoring='neg_log_loss',  # Negative LogLoss
            cv=self.cv, fit_params=self.lgb_fit_params
        )
        print(f'LogLoss={scores} \nLogLoss mean={np.mean(scores)}')
        


class TestCvRegressors:
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
    lgbr = LGBMRegressor(**lgb_params)

    def test_cross_val_score_reg_lgbm_float(self):
        scores = cross_val_score_eval_set(
            validation_fraction=0.3,
            estimator=self.lgbr,
            X=self.X, y=self.y,  # Data before cross validation division
            scoring='neg_root_mean_squared_error',  # Negative RMSE
            cv=self.cv, fit_params=self.lgb_fit_params
        )
        print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')
        
if __name__ == "__main__":
    test_regressors = TestCvRegressors()
    test_regressors.test_cross_val_score_reg_lgbm_float()
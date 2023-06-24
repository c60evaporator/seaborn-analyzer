# %%
import pytest
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from seaborn_analyzer import regplot

class TestRegPlot:
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
    lgb = LGBMRegressor(**lgb_params)
    # LightGBM pipeline model
    lgb_pipe_params = {
        'lgbm__objective': 'regression',
        'lgbm__metric': 'rmse',
        'lgbm__random_state': 42,
        'lgbm__boosting_type': 'gbdt',  # boosting_type
        'lgbm__n_estimators': 10000,
        'lgbm__verbose': -1,
        'lgbm__early_stopping_round': 10
        }
    lgb_pipe_fit_params = {
        'lgbm__eval_set':[(X, y)]
        }
    lgb_pipe = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())])
    lgb_pipe.set_params(**lgb_pipe_params)
    # SVM model
    svm_pipe_params = {
        'svm__kernel': 'rbf'
        }
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVR())])
    svm_pipe.set_params(**svm_pipe_params)

    def test_1_1_1_reg_pred_true_cv_lgbm(self):
        regplot.regression_pred_true(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            scores='mse',
                            cv=self.cv,
                            fit_params=self.lgb_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_1_1_2_reg_pred_true_cv_lgbm_pipe(self):
        regplot.regression_pred_true(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            scores='mse',
                            cv=self.cv,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_1_1_3_reg_pred_true_cv_svm_pipe(self):
        regplot.regression_pred_true(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            scores='mse',
                            cv=self.cv,
                            fit_params=self.lgb_pipe_fit_params
                            )
    
    def test_1_2_1_reg_pred_true_float_lgbm(self):
        regplot.regression_pred_true(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            scores='mse',
                            cv=self.cv,
                            fit_params=self.lgb_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_1_2_2_reg_pred_true_float_lgbm_pipe(self):
        regplot.regression_pred_true(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            scores='mse',
                            cv=self.cv,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_2_1_1_average_plot_cv_lgbm(self):
        regplot.average_plot(self.lgb, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_fit_params,
                    validation_fraction='cv'
                    )
    
    def test_2_1_2_average_plot_cv_lgbm_pipe(self):
        regplot.average_plot(self.lgb_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_pipe_fit_params,
                    validation_fraction='cv'
                    )
    
    def test_2_1_3_average_plot_float_svm_pipe(self):
        regplot.average_plot(self.svm_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    )
        
    def test_2_2_1_average_plot_float_lgbm(self):
        regplot.average_plot(self.lgb, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_fit_params,
                    validation_fraction=0.3
                    )
    
    def test_2_2_2_average_plot_float_lgbm_pipe(self):
        regplot.average_plot(self.lgb_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_pipe_fit_params,
                    validation_fraction=0.3
                    )
    
    def test_3_1_1_regression_heat_plot_cv_lgbm(self):
        regplot.regression_heat_plot(self.lgb, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_fit_params,
                    validation_fraction='cv',
                    rounddigit_x1=3,
                    rounddigit_x2=3,
                    rounddigit_x3=3
                    )
    
    def test_3_1_2_regression_heat_plot_cv_lgbm_pipe(self):
        regplot.regression_heat_plot(self.lgb_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_pipe_fit_params,
                    validation_fraction='cv',
                    rounddigit_x1=3,
                    rounddigit_x2=3,
                    rounddigit_x3=3
                    )
    
    def test_3_1_3_regression_heat_plot_cv_svm_pipe(self):
        regplot.regression_heat_plot(self.svm_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    rounddigit_x1=3,
                    rounddigit_x2=3,
                    rounddigit_x3=3
                    )
        
    def test_3_2_1_regression_heat_plot_float_lgbm(self):
        regplot.regression_heat_plot(self.lgb, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_fit_params,
                    validation_fraction=0.3,
                    rounddigit_x1=3,
                    rounddigit_x2=3,
                    rounddigit_x3=3
                    )
    
    def test_3_2_2_regression_heat_plot_float_lgbm_pipe(self):
        regplot.regression_heat_plot(self.lgb_pipe, x=self.X,
                    y=self.y,
                    x_colnames=self.USE_EXPLANATORY,
                    cv=self.cv,
                    fit_params=self.lgb_pipe_fit_params,
                    validation_fraction=0.3,
                    rounddigit_x1=3,
                    rounddigit_x2=3,
                    rounddigit_x3=3
                    )
        
if __name__ == "__main__":
    test_reg_plot = TestRegPlot()
    test_reg_plot.test_3_2_2_regression_heat_plot_float_lgbm_pipe()
# %%

# %%
import pytest
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
from seaborn_analyzer import classplot

class TestClassPlot:
    # Test data
    TARGET_VARIABLE = 'species'  # Target variable
    USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width']  # Explanatory variables
    df_iris = sns.load_dataset("iris")
    y = df_iris[TARGET_VARIABLE].values  # Numpy array of target variable
    X = df_iris[USE_EXPLANATORY].values  # Numpy array of explanatory variables
    # Cross validation KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # LightGBM model
    lgb_params = {
        'random_state': 42,
        'boosting_type': 'gbdt',  # boosting_type
        'n_estimators': 10000,
        'verbose': -1,
        'early_stopping_round': 10
        }
    lgb_fit_params = {
        'eval_metric': 'multi_logloss',
        'eval_set':[(X, y)]
        }
    lgb = LGBMClassifier(**lgb_params)
    # LightGBM pipeline model
    lgb_pipe_params = {
        'lgbm__random_state': 42,
        'lgbm__boosting_type': 'gbdt',  # boosting_type
        'lgbm__n_estimators': 10000,
        'lgbm__verbose': -1,
        'lgbm__early_stopping_round': 10
        }
    lgb_pipe_fit_params = {
        'lgbm__eval_metric': 'multi_logloss',
        'lgbm__eval_set':[(X, y)]
        }
    lgb_pipe = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())])
    lgb_pipe.set_params(**lgb_pipe_params)
    # SVM model
    svm_pipe_params = {
        'svm__kernel': 'rbf',
        'svm__probability': True
        }
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    svm_pipe.set_params(**svm_pipe_params)

    def test_1_1_1_class_separator_plot_cv_lgbm(self):
        classplot.class_separator_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            fit_params=self.lgb_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_1_1_2_class_separator_plot_cv_lgbm_pipe(self):
        classplot.class_separator_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_1_1_3_class_separator_plot_cv_svm_pipe(self):
        classplot.class_separator_plot(self.svm_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5
                            )
    
    def test_1_2_1_class_separator_plot_float_lgbm(self):
        classplot.class_separator_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            fit_params=self.lgb_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_1_2_2_class_separator_plot_float_lgbm_pipe(self):
        classplot.class_separator_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_2_1_1_class_proba_plot_cv_lgbm(self):
        classplot.class_proba_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            proba_type='imshow',
                            fit_params=self.lgb_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_2_1_2_class_proba_plot_cv_lgbm_pipe(self):
        classplot.class_proba_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            proba_type='imshow',
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_2_1_3_class_proba_plot_float_svm_pipe(self):
        classplot.class_proba_plot(self.svm_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            proba_type='imshow',
                            )
    
    def test_2_2_1_class_proba_plot_float_lgbm(self):
        classplot.class_proba_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            proba_type='imshow',
                            fit_params=self.lgb_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_2_2_2_class_proba_plot_float_lgbm_pipe(self):
        classplot.class_proba_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            pair_sigmarange=0.5,
                            proba_type='imshow',
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_3_1_1_roc_plot_cv_lgbm(self):
        classplot.roc_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            fit_params=self.lgb_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_3_1_2_roc_plot_cv_lgbm_pipe(self):
        classplot.roc_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction='cv'
                            )
    
    def test_3_1_3_roc_plot_cv_svm_pipe(self):
        classplot.roc_plot(self.svm_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv
                            )
        
    def test_3_2_1_roc_plot_float_lgbm(self):
        classplot.roc_plot(self.lgb, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            fit_params=self.lgb_fit_params,
                            validation_fraction=0.3
                            )
    
    def test_3_2_2_roc_plot_float_lgbm_pipe(self):
        classplot.roc_plot(self.lgb_pipe, x=self.X,
                            y=self.y,
                            x_colnames=self.USE_EXPLANATORY,
                            cv=self.cv,
                            fit_params=self.lgb_pipe_fit_params,
                            validation_fraction=0.3
                            )
        
if __name__ == "__main__":
    test_class_plot = TestClassPlot()
    test_class_plot.test_3_1_1_roc_plot_cv_lgbm()
# %%

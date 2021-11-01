# %% eval_set_selection引数の動作確認（回帰）1
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'eval_set': None}
not_opt_params={'objective': 'regression', # 最小化させるべき損失関数 
                'random_state': 42, # 乱数シード
                'boosting_type': 'gbdt', # boosting_type 
                'n_estimators': 100 # 最大学習サイクル数(評価指標がearly_stopping_rounds連続で改善しなければ打ち切り) 
                } 
tuning.optuna_tuning(not_opt_params=not_opt_params, fit_params=fit_params)

estimator = LGBMRegressor()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）2
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'eval_set': None}
not_opt_params={'objective': 'regression', # 最小化させるべき損失関数 
                'random_state': 42, # 乱数シード
                'boosting_type': 'gbdt', # boosting_type 
                'n_estimators': 100 # 最大学習サイクル数(評価指標がearly_stopping_rounds連続で改善しなければ打ち切り) 
                } 
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())])
tuning.optuna_tuning(estimator=estimator, not_opt_params=not_opt_params, fit_params=fit_params)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）3
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'verbose': 0, # 学習中のコマンドライン出力
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標
            'eval_set': [(X, y)] 
            }
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())]) 
tuning.optuna_tuning(estimator=estimator, fit_params=fit_params) 

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection='original_transformed'
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection='original_transformed',
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）4
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='original') 
fit_params={'verbose': 0, # 学習中のコマンドライン出力 
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなく なった時点でストップ 
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標 
            'eval_set': [(X, y)] 
            }
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())]) 
tuning.optuna_tuning(estimator=estimator, fit_params=fit_params) 

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）5
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())])
tuning.optuna_tuning(estimator=estimator)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）6
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='all')
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())])
tuning.optuna_tuning(estimator=estimator)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）7
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'verbose': 0, # 学習中のコマンドライン出力
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標
            'eval_set': [(X, y)] 
            }
tuning.optuna_tuning(fit_params=fit_params)

estimator = LGBMRegressor()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）8
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.optuna_tuning()

estimator = LGBMRegressor()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )

# %% eval_set_selection引数の動作確認（回帰）9
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='train')
tuning.optuna_tuning()

estimator = LGBMRegressor()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
regplot.average_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=df_reg,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
regplot.regression_heat_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection,
                            rounddigit_x1=3,
                            rounddigit_x2=3,
                            rounddigit_x3=3
                            )





# %% eval_set_selection引数の動作確認（分類）1
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'eval_set': None}
not_opt_params={'objective': 'regression', # 最小化させるべき損失関数 
                'random_state': 42, # 乱数シード
                'boosting_type': 'gbdt', # boosting_type 
                'n_estimators': 100 # 最大学習サイクル数(評価指標がearly_stopping_rounds連続で改善しなければ打ち切り) 
                } 
tuning.optuna_tuning(not_opt_params=not_opt_params, fit_params=fit_params)

estimator = LGBMClassifier()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）2
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'eval_set': None}
not_opt_params={'objective': 'regression', # 最小化させるべき損失関数 
                'random_state': 42, # 乱数シード
                'boosting_type': 'gbdt', # boosting_type 
                'n_estimators': 100 # 最大学習サイクル数(評価指標がearly_stopping_rounds連続で改善しなければ打ち切り) 
                } 
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())])
tuning.optuna_tuning(estimator=estimator, not_opt_params=not_opt_params, fit_params=fit_params)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）3
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'verbose': 0, # 学習中のコマンドライン出力
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標
            'eval_set': [(X, y)] 
            }
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())]) 
tuning.optuna_tuning(estimator=estimator, fit_params=fit_params) 

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection='original_transformed'
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection='original_transformed'
                    )

# %% eval_set_selection引数の動作確認（分類）4
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='original') 
fit_params={'verbose': 0, # 学習中のコマンドライン出力 
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなく なった時点でストップ 
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標 
            'eval_set': [(X, y)] 
            }
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())]) 
tuning.optuna_tuning(estimator=estimator, fit_params=fit_params) 

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）5
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())])
tuning.optuna_tuning(estimator=estimator)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）6
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='all')
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier())])
tuning.optuna_tuning(estimator=estimator)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）7
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'verbose': 0, # 学習中のコマンドライン出力
            'early_stopping_rounds': 10, # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'rmse', # early_stopping_roundsの評価指標
            'eval_set': [(X, y)] 
            }
tuning.optuna_tuning(fit_params=fit_params)

estimator = LGBMClassifier()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）8
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.optuna_tuning()

estimator = LGBMClassifier()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )

# %% eval_set_selection引数の動作確認（分類）9
import seaborn as sns
from lightgbm import LGBMClassifier
from muscle_tuning import LGBMClassifierTuning
from seaborn_analyzer import classplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIABLE].values
X = iris[USE_EXPLANATORY].values

tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE, eval_set_selection='train')
tuning.optuna_tuning()

estimator = LGBMClassifier()
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
classplot.class_separator_plot(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=iris,
                            cv=tuning.cv,
                            pair_sigmarange=0.5,
                            fit_params=tuning.fit_params,
                            eval_set_selection=tuning.eval_set_selection
                            )
plt.show()
classplot.class_proba_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    pair_sigmarange=0.5,
                    proba_type='imshow',
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
plt.show()
classplot.roc_plot(best_estimator, x=USE_EXPLANATORY,
                    y=OBJECTIVE_VARIABLE, data=iris,
                    cv=tuning.cv,
                    fit_params=tuning.fit_params,
                    eval_set_selection=tuning.eval_set_selection
                    )
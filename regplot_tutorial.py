# %% 概要の「機能1」（相関分析）
from custom_scatter_plot import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris, hue='species', rounddigit=5)

#%% 概要の「機能2」（予測値と実測値のプロット）
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_pred_true(SVR(), x='petal_length', y='sepal_length', data=iris, plot_stats='median', rounddigit=3, rank_number=3, cv=2)

# %% 概要の「機能3」（1次元説明変数回帰モデルの可視化）
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris, plot_stats='median', rounddigit=3, cv=2)

# %% 概要の「機能4」（2～4次元説明変数回帰モデルの可視化）
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_heat_plot(SVR(), x=['sepal_width', 'petal_width', 'petal_length'], y='sepal_length', data=iris, x_heat=['petal_length', 'petal_width'], pair_sigmarange=1.0, rank_number=3, cv=2, display_cv_indices=1)

# %% 散布図
import seaborn as sns
iris = sns.load_dataset("iris")
sns.scatterplot(x='petal_length', y='sepal_length', data=iris)
# %% custom_scatter_plot.regplot.linear_plot
from custom_scatter_plot import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris, rounddigit=3)
# %% 気象庁の温度・標高・緯度データを可視化
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df_temp = pd.read_csv(f'./temp_pressure.csv')
X = df_temp[['altitude', 'latitude']].values  # 説明変数(標高+緯度)
y = df_temp['temperature'].values  # 目的変数(気温)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X[:, 0], X[:, 1], y)
ax.set_xlabel('altitude [m]')
ax.set_ylabel('latitude [°]')
ax.set_zlabel('temperature [°C]')
# %% 線形回帰してr2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
lr = LinearRegression()  # 線形回帰用クラス
lr.fit(X, y)  # 線形回帰学習
y_pred = lr.predict(X)  # 学習モデルから回帰して予測値を算出
print(1000, 0, f'R2={r2_score(y, y_pred)}')  # R2_Scoreを表示
print(1000, 0, f'MAE={mean_absolute_error(y, y_pred)}')  # MAEを表示
# %% 予測値 vs 実測値プロット
import seaborn as sns
sns.scatterplot(x=y, y=y_pred)
ax = plt.gca()
ax.set_xlabel('true')
ax.set_ylabel('pred')

#%% custom_scatter_plot.regplot.regression_pred_true
import pandas as pd
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.linear_model import LinearRegression
df_temp = pd.read_csv(f'./temp_pressure.csv')
regplot.regression_pred_true(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'], rounddigit=3)
# %% ランダムフォレスト回帰
from sklearn.ensemble import RandomForestRegressor
regplot.regression_pred_true(RandomForestRegressor(), x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'], rounddigit=3)
# %% クロスバリデーション
regplot.regression_pred_true(LinearRegression(), cv=2, x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'], rounddigit=3)
# %% 誤差上位の表示
regplot.regression_pred_true(LinearRegression(), rank_number=3, rank_col='city', x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'], rounddigit=3)

# %% 1次元説明変数の場合の回帰線プロット
import numpy as np
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset("iris")
# 散布図プロット
sns.scatterplot(x='petal_length', y='sepal_length', data=iris)
# サポートベクター回帰学習
model = SVR()
X = iris[['petal_length']].values
y = iris['sepal_length'].values
model.fit(X, y)  
# 回帰モデルの線を作成
xmin = np.amin(X)
xmax = np.amax(X)
Xline = np.linspace(xmin, xmax, 100)
Xline = Xline.reshape(len(Xline), 1)
# 回帰線を描画
plt.plot(Xline, model.predict(Xline), color='red')

# %% custom_scatter_plot.regplot.regression_plot_1d
import seaborn as sns
from custom_scatter_plot import regplot
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris, rounddigit=3)
# %% custom_scatter_plot.regplot.regression_heat_plot
import pandas as pd
from sklearn.linear_model import LinearRegression
from custom_scatter_plot import regplot
df_temp = pd.read_csv(f'./temp_pressure.csv')
regplot.regression_heat_plot(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% ランダムフォレスト回帰
from sklearn.ensemble import RandomForestRegressor
regplot.regression_heat_plot(RandomForestRegressor(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% クロスバリデーション
regplot.regression_heat_plot(LinearRegression(), cv=2, display_cv_indices=[0, 1], x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% 誤差上位の表示
regplot.regression_heat_plot(LinearRegression(), rank_number=3, rank_col='city', x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% 3次元ヒートマップ
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_heat_plot(SVR(), x=['sepal_width', 'petal_width', 'petal_length'], y='sepal_length', data=iris, x_heat=['petal_length', 'petal_width'], pair_sigmarange=1.0)
# %%

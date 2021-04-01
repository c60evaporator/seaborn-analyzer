# %% 概要の「機能1」（相関分析）
from custom_scatter_plot import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris, hue='species', rounddigit=5)

#%% 概要の「機能2」（予測値と実測値のプロット）
from custom_scatter_plot import regplot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
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
# %%

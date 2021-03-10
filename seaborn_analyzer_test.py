#%% custom_pair_plot
from custom_pair_plot import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')

#%% custom_dist_plot
from custom_dist_plot import dist
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
dist.hist_dist(iris['sepal_length'], ax=axes[0, 0], bin_width=0.2, norm_hist=False, rounddigit=5)
#%% custom_scatter_plot.regression_plot_pred
from custom_scatter_plot import regplot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut

iris = sns.load_dataset("iris")
regplot.regression_plot_pred(SVR(), 'petal_length', 'sepal_length', iris, plot_stats='median', rounddigit=3, rank_number=3, cv=5, scores=None)
# %% custom_scatter_plot.linear_plot
from custom_scatter_plot import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot('petal_length', 'sepal_length', iris, hue='species', rounddigit=5)
# %% custom_scatter_plot.regression_plot_1d
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), 'petal_length', 'sepal_length', iris, plot_stats='median', rounddigit=3, cv=5)
# %% custom_scatter_plot.regression_plot_1d
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_heat_plot(SVR(), ['sepal_width', 'petal_width', 'petal_length'], 'sepal_length', iris, x_heat=['petal_length', 'petal_width'], rank_number=3, cv=5, display_cv_indices=3)

# %%

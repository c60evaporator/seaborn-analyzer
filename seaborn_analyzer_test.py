#%% custom_pair_plot
from custom_pair_plot import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')


#%% custom_dist_plot.hist.fit_dist
from custom_hist_plot import hist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

iris = sns.load_dataset("iris")
all_params, all_scores = hist.fit_dist(iris, x='sepal_width', dist=['norm', 'gamma', 'lognorm', 'uniform'], bins=20)
df_scores = pd.DataFrame(all_scores).T
df_scores

#%% custom_dist_plot.hist.plot_normality
from custom_hist_plot import hist
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
hist.plot_normality(iris, x='sepal_width', binwidth=0.2, norm_hist=False, rounddigit=3)


# %% custom_scatter_plot.regplot.linear_plot
from custom_scatter_plot import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot('petal_length', 'sepal_length', iris, hue='species', rounddigit=5)

#%% custom_scatter_plot.regplot.regression_pred_true
from custom_scatter_plot import regplot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut

iris = sns.load_dataset("iris")
regplot.regression_pred_true(SVR(), 'petal_length', 'sepal_length', iris, plot_stats='median', rounddigit=3, rank_number=3, cv=5)

# %% custom_scatter_plot.regplot.regression_plot_1d
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), 'petal_length', 'sepal_length', iris, plot_stats='median', rounddigit=3, cv=5)

# %% custom_scatter_plot.regplot.regression_heat_plot
from custom_scatter_plot import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_heat_plot(SVR(), ['sepal_width', 'petal_width', 'petal_length'], 'sepal_length', iris, x_heat=['petal_length', 'petal_width'], rank_number=3, cv=5, display_cv_indices=3, plot_scatter='error')

# %%

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
dist.hist_dist(iris['sepal_length'], ax=axes[0, 0], rounddigit=5)
#%% custom_scatter_plot
from custom_scatter_plot import dist
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

iris = sns.load_dataset("iris")
dist.regression_plot_pred(LinearRegression(), 'petal_length', 'sepal_length', iris, plot_stats='median', rounddigit=5, cv=5)
# %%

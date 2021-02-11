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
# %%

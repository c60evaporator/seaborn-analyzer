#%%
import seaborn as sns
iris = sns.load_dataset("iris")
sns.distplot(iris['sepal_width'], kde=False)

#%%
import seaborn as sns
from custom_hist_plot import hist
iris = sns.load_dataset("iris")
hist.plot_normality(iris, x='sepal_width', norm_hist=False, rounddigit=5)
#%%
from custom_hist_plot import hist
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
boston = load_boston()
X = boston.data
df = pd.DataFrame(X, columns= boston.feature_names)
hist.plot_normality(df, x='LSTAT', norm_hist=False, rounddigit=5)
#%%
all_params, all_scores = hist.hist_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'], norm_hist=True)
df_scores = pd.DataFrame(all_scores).T
df_scores
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
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
hist.plot_normality(df, x='LSTAT', norm_hist=False, rounddigit=5)

#%% ガンマ分布
from custom_hist_plot import hist
from sklearn.datasets import load_boston
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
X = df['LSTAT'].values
sns.distplot(X, kde=False, norm_hist=True, hist_kws={'alpha':0.7, 'edgecolor':'white'})
Xline = np.linspace(min(np.mean(X) - np.std(X) * 4, np.amin(X)), max(np.mean(X) + np.std(X) * 4, np.amax(X)), 100)
line_legends = []
line_labels = []
line_colors = ['red', 'darkmagenta', 'mediumblue', 'darkorange']
params = [(1.0, 3.2),
          (4.0, 3.2),
          (4.0, 6.4),
          (8.0, 3.2)]
for i, (k, theta) in enumerate(params):
    # 対数尤度
    logL = np.sum(stats.gamma.logpdf(X, loc=0, scale=k, a=theta))
    # 線の描画
    Yline = stats.gamma.pdf(Xline, loc=0, scale=k, a=theta)
    leg, = plt.plot(Xline, Yline, color=line_colors[i])
    line_legends.append(leg)
    line_labels.append(f'k={k}, θ={theta}, logL={round(logL, 1)}')
plt.legend(line_legends, line_labels, loc='upper right')

#%% 最尤推定
from custom_hist_plot import hist
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
all_params, all_scores = hist.fit_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'])
df_scores = pd.DataFrame(all_scores).T
df_scores

# %%

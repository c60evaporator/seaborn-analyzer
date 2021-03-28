#### こちらの記事のグラフ作成用スクリプトhttps://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42
#%% ヒストグラム
import seaborn as sns
iris = sns.load_dataset("iris")
sns.distplot(iris['sepal_width'], kde=False)

#%% 
import seaborn as sns
from custom_hist_plot import hist
iris = sns.load_dataset("iris")
hist.plot_normality(iris['sepal_width'])

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
from scipy import stats
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
all_params, all_scores = hist.fit_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'])
df_scores = pd.DataFrame(all_scores).T
df_scores

# %% 正規分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
mean_list = [0, 1]
std_list = [1, 2]
loc = 0
x = np.linspace(-8, 8, 200)
for mean in mean_list:
    for std in std_list:
        plt.plot(x, stats.norm.pdf(x, scale=std, loc=mean), label=f'mean={mean}, std={std}')
        plt.legend()
# %% 対数正規分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
mu_list = [0, 1]
sigma_list = [1, 2]
loc = 0
x = np.linspace(0, 4, 200)
for mu in mu_list:
    for sigma in sigma_list:
        plt.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu), loc=loc), label=f'mu={mu}, sigma={sigma}')
        plt.legend()
# %% ガンマ分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
theta_list = [0.5, 1]
k_list = [1, 3]
loc = 0
x = np.linspace(0, 8, 200)
for theta in theta_list:
    for k in k_list:
        plt.plot(x, stats.gamma.pdf(x, a=k, scale=theta, loc=loc), label=f'theta={theta}, k={k}')
        plt.legend()
# %% コーシー分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
scale = 1
loc = 0
x = np.linspace(-4, 4, 200)
plt.plot(x, stats.cauchy.pdf(x, scale=scale, loc=loc))
# %% t分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
df_list = [1, 2, 3, 4]
scale = 1
loc = 0
x = np.linspace(-4, 4, 200)
for df in df_list:
    plt.plot(x, stats.t.pdf(x, df=df, scale=scale, loc=loc), label=f'df={df}')
    plt.legend()
# %% パレート分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
b_list = [1, 2, 3, 4]
scale = 1
loc = 0
x = np.linspace(0, 4, 200)
for b in b_list:
    plt.plot(x, stats.pareto.pdf(x, b=b, scale=scale, loc=loc), label=f'b={b}')
    plt.legend()
# %% 一様分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
scale = 1
loc = 0
x = np.linspace(-0.2, 1.2, 200)
plt.plot(x, stats.uniform.pdf(x, scale=scale, loc=loc))
# %% 指数分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
lambda_list = [0.5, 1, 2]
loc = 0
x = np.linspace(0, 4, 200)
for lm in lambda_list:
    plt.plot(x, stats.expon.pdf(x, scale=1/lm, loc=loc), label=f'lambda={lm}')
    plt.legend()
# %% ワイブル分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
c_list = [0.5, 1, 2, 4]
scale = 1
loc = 0
x = np.linspace(0, 4, 200)
for c in c_list:
    plt.plot(x, stats.weibull_min.pdf(x, c=c, scale=scale, loc=loc), label=f'c={c}')
    plt.legend()
# %% カイ二乗分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
df_list = [1, 2, 3, 4]
scale = 1
loc = 0
x = np.linspace(0.2, 8, 200)
for df in df_list:
    plt.plot(x, stats.chi2.pdf(x, df=df, scale=scale, loc=loc), label=f'df={df}')
    plt.legend()
# %% 二項分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
n = 100
p_list = [0.01, 0.05, 0.1, 0.2]
loc = 0
x = np.linspace(0, 20, 21)
for p in p_list:
    plt.plot(x, stats.binom.pmf(x, n=n, p=p, loc=loc), label=f'p={p}')
    plt.legend()
# %% ポアソン分布
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
mu_list = [1, 5, 10, 20]
loc = 0
x = np.linspace(0, 20, 21)
for mu in mu_list:
    plt.plot(x, stats.poisson.pmf(x, mu=mu, loc=loc), label=f'mu={mu}')
    plt.legend()
# %%

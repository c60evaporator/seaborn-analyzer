#######散布図行列と相関係数行列 custom_pair_plot.py (https://qiita.com/c60evaporator/items/20f11b6ee965cec48570)######
#%% custom_pair_plot
from seaborn_analyzer import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')


#######ヒストグラムと各種分布フィッティング custom_hist_plot.py (https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42)######
#%% ヒストグラム表示
import seaborn as sns
iris = sns.load_dataset("iris")
sns.distplot(iris['sepal_width'], kde=False)
# %% 概要の「機能1」（正規性検定とQQプロット）
import seaborn as sns
from seaborn_analyzer import hist
iris = sns.load_dataset("iris")
hist.plot_normality(iris['sepal_width'])
#%% 概要の「機能2」（分布フィッティングと評価指標算出）
from seaborn_analyzer import hist
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
all_params, all_scores = hist.fit_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'])
df_scores = pd.DataFrame(all_scores).T
df_scores
# %% Bostonデータセットでの正規性検定とQQプロット
from seaborn_analyzer import hist
from sklearn.datasets import load_boston
import pandas as pd
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
hist.plot_normality(df, x='LSTAT', norm_hist=False, rounddigit=5)
#%% ガンマ分布をパラメータ変えてフィッティング
from seaborn_analyzer import hist
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



#######分類モデルの可視化 custom_scatter_plot.classplot ()######
# %% 概要の「機能1」（決定境界）
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
clf = SVC()
classplot.class_separator_plot(clf, ['petal_width', 'petal_length'], 'species', iris)
# %% 概要の「機能2」（クラス確率）
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
clf = SVC(probability=True)
classplot.class_proba_plot(clf, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='contourf')
# %% アヤメ散布図
import seaborn as sns
iris = sns.load_dataset("iris")
sns.scatterplot(x='petal_width', y='petal_length', data=iris,
                hue='species', palette=['green', 'red', 'blue'])
# %% サポートベクターマシン分類して性能評価指標算出
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
X = iris[['petal_width', 'petal_length']].values  # 説明変数
y = iris['species'].values  # 目的変数(3種類のアヤメ種類)
clf = SVC()  # SVM分類用インスタンス
clf.fit(X, y)  # 学習
y_pred = clf.predict(X)  # 推論
print(f'Accuracy={accuracy_score(y, y_pred)}')  # 正解率を表示
print(f'F1={f1_score(y, y_pred, average="macro")}')  # F1-Macroを表示
# %% mlxtendで決定境界可視化
from mlxtend.plotting import plot_decision_regions
import numpy as np
# 目的変数の配列を2次元に変換
y = y.reshape(len(y), 1)
# 目的変数をstr型→int型に変換
label_names = list(dict.fromkeys(y[:, 0]))
label_dict = dict(zip(label_names, range(len(label_names))))
y_int=np.vectorize(lambda x: label_dict[x])(y)
# 学習
clf.fit(X, y_int)
# mlxtendで決定境界可視化
plot_decision_regions(X, y_int[:, 0], clf=clf,
                      colors='green,red,blue')
# %% 本ツールによる決定境界可視化
from seaborn_analyzer import classplot
clf = SVC()
classplot.class_separator_plot(clf, ['petal_width', 'petal_length'], 'species', iris)
# %% ランダムフォレスト回帰での描画例
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
classplot.class_separator_plot(clf, ['petal_width', 'petal_length'], 'species', iris)
# %% 2分割クロスバリデーション
clf = SVC()
classplot.class_separator_plot(clf, ['petal_width', 'petal_length'], 'species', iris,
                               cv=2, display_cv_indices = [0, 1])
# %% 3次元特徴量
classplot.class_separator_plot(clf, ['petal_width', 'petal_length', 'sepal_width'], 'species', iris,
                               x_chart=['petal_width', 'petal_length'],
                               pair_sigmarange = 1.0, pair_sigmainterval = 0.5)
# %% 4次元特徴量
classplot.class_separator_plot(clf, ['petal_width', 'petal_length', 'sepal_width', 'sepal_length'], 'species', iris,
                               x_chart=['petal_width', 'petal_length'],
                               pair_sigmarange = 0.5, pair_sigmainterval = 0.5,
                               chart_scale=2)
# %% 本ツールによるクラス確率可視化（等高線表示）
clf = SVC(probability=True)  # SVMでpredict_probaを有効にするため、引数"probability"をTrueに
classplot.class_proba_plot(clf, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='contourf')
# %% 本ツールによるクラス確率可視化（RGB画像表示）
classplot.class_proba_plot(clf, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='imshow')
# %% class_separator_plotの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from seaborn_analyzer import classplot
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
clf = SVC()
classplot.class_separator_plot(clf, x=['petal_width', 'petal_length'],
                               y='species', data=iris,
                               scatter_kws={'edgecolors': 'red'})
# %% class_proba_plotの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from xgboost import XGBClassifier
from seaborn_analyzer import classplot
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
clf = SVC(probability=True)
classplot.class_proba_plot(clf, x=['petal_width', 'petal_length'],
                           y='species', data=iris,
                           proba_type='imshow',
                           imshow_kws={'alpha':0.8})
# %% パイプライン
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from seaborn_analyzer import classplot
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])  # 標準化＋SVMパイプライン
classplot.class_separator_plot(pipe, x=['petal_width', 'petal_length'],
                               y='species', data=iris)
# %% 特徴量重要度
import seaborn as sns
iris = sns.load_dataset("iris")
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# モデルの学習
clf = XGBClassifier()
features = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']
X = iris[features].values
y = iris['species'].values
clf.fit(X, y)
# 特徴量重要度の取得と可視化
importances = list(clf.feature_importances_)
plt.barh(features, importances)
# %% 主成分分析
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from seaborn_analyzer import classplot
features = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']
X = iris[features].values
y = iris['species'].values
# 前処理として標準化
ss = StandardScaler()
ss.fit(X)  # 標準化パラメータの学習
X_ss = ss.transform(X)  # 学習結果に基づき標準化
# 主成分分析で4→2次元に圧縮
pca = PCA(n_components=2)
pca.fit(X_ss)  # 主成分分析の学習
X_pca = pca.transform(X_ss)  # 学習結果に基づき次元圧縮
iris['pc1'] = pd.Series(X_pca[:, 0])  # 第1主成分をirisに格納
iris['pc2'] = pd.Series(X_pca[:, 1])  # 第2主成分をirisに格納
clf = SVC()
classplot.class_separator_plot(clf, x=['pc1', 'pc2'],
                               y='species', data=iris)
# %% t-SNE
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
features = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']
X = iris[features].values
y = iris['species'].values
# 前処理として標準化
ss = StandardScaler()
ss.fit(X)  # 標準化パラメータの学習
X_ss = ss.transform(X)  # 学習結果に基づき標準化
# t-SNEで4→2次元に圧縮
tsne = TSNE(n_components=2,  # 圧縮後の次元数
            n_iter=1000,  # 学習繰り返し回数
            random_state=42)  # 乱数シード
points = tsne.fit_transform(iris[features])  # t-SNEの学習と次元圧縮
iris['1st'] = pd.Series(points[:, 0])  # 第1軸をirisに格納
iris['2nd'] = pd.Series(points[:, 1])  # 第2軸をirisに格納
sns.scatterplot(x='1st', y='2nd', hue='species', data=iris)



#######回帰モデルの可視化 seaborn_analyzer.regplot (https://qiita.com/c60evaporator/items/c930c822b527f62796ee)######
# %% 概要の「機能1」（相関分析）
from seaborn_analyzer import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris)
#%% 概要の「機能2」（予測値と実測値のプロット）
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_pred_true(SVR(), x='petal_length', y='sepal_length', data=iris, cv_stats='median', rank_number=3, cv=2)
# %% 概要の「機能3」（1次元説明変数回帰モデルの可視化）
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris, cv_stats='median', cv=2)
# %% 概要の「機能4」（2～4次元説明変数回帰モデルの可視化）
import pandas as pd
from sklearn.linear_model import LinearRegression
from seaborn_analyzer import regplot
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
regplot.regression_heat_plot(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% 散布図
import seaborn as sns
iris = sns.load_dataset("iris")
sns.scatterplot(x='petal_length', y='sepal_length', data=iris)
# %% custom_scatter_plot.regplot.linear_plot
from seaborn_analyzer import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris)
# %% 気象庁の温度・標高・緯度データを可視化
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
X = df_temp[['altitude', 'latitude']].values  # 説明変数(標高+緯度)
y = df_temp['temperature'].values  # 目的変数(気温)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X[:, 0], X[:, 1], y)
ax.set_xlabel('altitude [m]')
ax.set_ylabel('latitude [°]')
ax.set_zlabel('temperature [°C]')
# %% 線形回帰して性能評価指標算出
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
lr = LinearRegression()  # 線形回帰用インスタンス
lr.fit(X, y)  # 線形回帰学習
y_pred = lr.predict(X)  # 学習モデルから回帰して予測値を算出
print(f'R2={r2_score(y, y_pred)}')  # R2_Scoreを表示
print(f'MAE={mean_absolute_error(y, y_pred)}')  # MAEを表示
# %% 予測値 vs 実測値プロット
import seaborn as sns
sns.scatterplot(x=y, y=y_pred)
ax = plt.gca()
ax.set_xlabel('true')
ax.set_ylabel('pred')
#%% seaborn_analyzer.regplot.regression_pred_true
import pandas as pd
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.linear_model import LinearRegression
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
regplot.regression_pred_true(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'])
# %% ランダムフォレスト回帰
from sklearn.ensemble import RandomForestRegressor
regplot.regression_pred_true(RandomForestRegressor(), x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'])
# %% クロスバリデーション
regplot.regression_pred_true(LinearRegression(), cv=2, x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'])
# %% 誤差上位の表示
regplot.regression_pred_true(LinearRegression(), rank_number=3, rank_col='city', x=['altitude', 'latitude'], y='temperature', data=df_temp, scores=['mae', 'r2'])
# %% 1次元説明変数の場合の回帰線プロット
import numpy as np
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset("iris")
# 散布図プロット
sns.scatterplot(x='petal_length', y='sepal_length', data=iris)
# サポートベクター回帰学習
svr = SVR()
X = iris[['petal_length']].values
y = iris['sepal_length'].values
svr.fit(X, y)  
# 回帰モデルの線を作成
xmin = np.amin(X)
xmax = np.amax(X)
Xline = np.linspace(xmin, xmax, 100)
Xline = Xline.reshape(len(Xline), 1)
# 回帰線を描画
plt.plot(Xline, svr.predict(Xline), color='red')
# %% seaborn_analyzer.regplot.regression_plot_1d
import seaborn as sns
from seaborn_analyzer import regplot
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris)
# %% seaborn_analyzer.regplot.regression_heat_plot
import pandas as pd
from sklearn.linear_model import LinearRegression
from seaborn_analyzer import regplot
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
regplot.regression_heat_plot(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% ランダムフォレスト回帰
from sklearn.ensemble import RandomForestRegressor
regplot.regression_heat_plot(RandomForestRegressor(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% クロスバリデーション
regplot.regression_heat_plot(LinearRegression(), cv=2, display_cv_indices=[0, 1], x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% 誤差上位の表示
regplot.regression_heat_plot(LinearRegression(), rank_number=3, rank_col='city', x=['altitude', 'latitude'], y='temperature', data=df_temp)
# %% 3次元ヒートマップ
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_heat_plot(SVR(), x=['sepal_width', 'petal_width', 'petal_length'], y='sepal_length', data=iris, x_heat=['petal_length', 'petal_width'], pair_sigmarange=1.0)
# %% linear_plotの参考図プロット用
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset("iris")
from seaborn_analyzer import regplot
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris, plot_scores=False)
# %% regression_pred_trueの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GroupKFold, LeaveOneGroupOut
from xgboost import XGBRegressor
regplot.regression_pred_true(RandomForestRegressor(), x=['petal_width', 'petal_length'],
                             y='sepal_length', data=iris,
                             cv=2, subplot_kws={'figsize': (3, 9)})
# %% regression_plot_1dの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ShuffleSplit
from xgboost import XGBRegressor
regplot.regression_plot_1d(RandomForestRegressor(), x='petal_length', 
                           y='sepal_length', data=iris,
                           cv=2, subplot_kws={'figsize': (3, 9)})
# %% regression_heat_plotの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, GroupKFold, LeaveOneGroupOut
from xgboost import XGBRegressor
regplot.regression_heat_plot(RandomForestRegressor(), x=['petal_width', 'petal_length'],
                             y='sepal_length', data=iris,
                             scatter_kws={'marker': 'v'})
# %% パイプライン
import seaborn as sns
iris = sns.load_dataset("iris")
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
regplot.regression_heat_plot(pipe, x=['petal_width', 'petal_length'],
                             y='sepal_length', data=iris)
# %% 特徴量重要度
import seaborn as sns
iris = sns.load_dataset("iris")
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
# モデルの学習
xgbr = XGBRegressor()
features = ['sepal_width', 'petal_width', 'petal_length']
X = iris[features].values
y = iris['sepal_length'].values
xgbr.fit(X, y)
# 特徴量重要度の取得と可視化
importances = list(xgbr.feature_importances_)
plt.barh(features, importances)
# %% 残差プロット
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
# モデルの学習
lr = LinearRegression()
features = ['petal_width', 'petal_length']
X = iris[features].values
y = iris['sepal_length'].values
lr.fit(X, y)
# 残差プロット(横軸は目的変数予測値)
y_pred = lr.predict(X)
error = y_pred - y
plt.scatter(y_pred, error)
plt.xlabel('y_pred')
plt.ylabel('error')
# 残差=0の補助線を引く
plt.plot([np.amin(y_pred), np.amax(y_pred)], [0, 0], "red")
# %% 大阪都構想データで4次元プロット
import pandas as pd
from seaborn_analyzer import regplot
from xgboost import XGBRegressor
df = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
regplot.regression_heat_plot(XGBRegressor(), x=['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude'],
                             y='approval_rate', data=df, x_heat=['2_between_30to60', '5_household_member'],
                             pair_sigmarange = 0.5, pair_sigmainterval = 0.5,
                             rank_number=3, rank_col='ward_before',
                             rounddigit_x1=3,
                             estimator_params={'learning_rate': 0.297,
                                           'min_child_weight': 4,
                                           'max_depth': 6,
                                           'colsample_bytree': 0.545,
                                           'subsample': 0.54},
                             fit_params={'early_stopping_rounds': 20,
                                         'eval_set': [(df[['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']].values, df['approval_rate'].values)],
                                         'verbose': 1})

# %% muscle-brain-tuningのテスト用
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
df_boston['price'] = y

estimator1 = SVR()
estimator2 = RandomForestRegressor()
fig, axes = plt.subplots(4, 2, figsize=(8, 16))
fig.suptitle(f'prediction vs ')
axes1 = [row[0] for row in axes]
axes2 = [row[1] for row in axes]
regplot.regression_pred_true(estimator1, USE_EXPLANATORY, 'price', df_boston, cv=3, ax=axes1, scores='mape')
regplot.regression_pred_true(estimator2, USE_EXPLANATORY, 'price', df_boston, cv=3, ax=axes2, scores='mape')
title_before = axes1[0].title._text
axes1[0].set_title(f'SVM\n\n{title_before}')
axes2[0].set_title('RandomForest')
plt.tight_layout()
plt.show()

# %% legend_kws引数テスト用
from seaborn_analyzer import regplot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['NOX', 'RM', 'DIS', 'LSTAT', 'CHAS']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
df_boston['price'] = y
regplot.linear_plot('NOX', 'price', df_boston, hue='CHAS', legend_kws={'loc': 5})

# %% ROC曲線
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List
import numbers
from sklearn.metrics import auc, plot_roc_curve, roc_curve, RocCurveDisplay
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import colors

def _reshape_input_data(x, y, data, x_colnames, cv_group):
    """
    入力データの形式統一(pd.DataFrame or np.ndarray)
    """
    # dataがpd.DataFrameのとき
    if isinstance(data, pd.DataFrame):
        if not isinstance(x, list):
            raise Exception('`x` argument should be list[str] if `data` is pd.DataFrame')
        if not isinstance(y, str):
            raise Exception('`y` argument should be str if `data` is pd.DataFrame')
        if x_colnames is not None:
            raise Exception('`x_colnames` argument should be None if `data` is pd.DataFrame')
        X = data[x].values
        y_true = data[y].values
        x_colnames = x
        y_colname = y
        cv_group_colname = cv_group
        
    # dataがNoneのとき(x, y, cv_groupがnp.ndarray)
    elif data is None:
        if not isinstance(x, np.ndarray):
            raise Exception('`x` argument should be np.ndarray if `data` is None')
        if not isinstance(y, np.ndarray):
            raise Exception('`y` argument should be np.ndarray if `data` is None')
        X = x
        y_true = y.ravel()
        # x_colnameとXの整合性確認
        if x_colnames is None:
            x_colnames = list(range(x.shape[1]))
        elif x.shape[1] != len(x_colnames):
            raise Exception('width of X must be equal to length of x_colnames')
        else:
            x_colnames = x_colnames
        y_colname = 'objective_variable'
        if cv_group is not None:  # cv_group指定時
            cv_group_colname = 'group'
            data = pd.DataFrame(np.column_stack((X, y_true, cv_group)),
                                columns=x_colnames + [y_colname] + [cv_group_colname])
        else:
            cv_group_colname = None
            data = pd.DataFrame(np.column_stack((X, y)),
                                columns=x_colnames + [y_colname])
    else:
        raise Exception('`data` argument should be pd.DataFrame or None')

    return X, y_true, data, x_colnames, y_colname, cv_group_colname

def plot_roc_curve_multiclass(estimator, X_train, y_train, *,
                              X_test=None, y_test=None,
                              sample_weight=None, drop_intermediate=True,
                              response_method="auto", name=None, ax=None, pos_label=None,
                              average='macro', fit_params=None,
                              plot_roc_kws=None, class_average_kws=None,
                              ):
    # X_testがNoneのとき、X_trainを使用
    if X_test is None:
        X_test = X_train
    # y_testがNoneのとき、y_trainを使用
    if y_test is None:
        y_test = y_train
    # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
    if ax is None:
        ax = plt.gca()
    # 学習時パラメータがNoneなら空のdictを入力
    if fit_params is None:
        fit_params = {}
    # plot_roc_kwsがNoneなら空のdictを入力
    if plot_roc_kws is None:
        plot_roc_kws = {}
    # class_average_kwsがNoneなら空のdictを入力
    if class_average_kws is None:
        class_average_kws = {}
    # 目的変数のクラス一覧
    y_labels = sorted(np.unique(np.concatenate([y_train, y_test], 0)).tolist())
    n_classes = len(y_labels)
    
    # 2クラス分類のとき
    if n_classes == 2:
        estimator.fit(X_train, y_train, **fit_params)
        viz = plot_roc_curve(estimator, X_test, y_test,
                             sample_weight=sample_weight, drop_intermediate=drop_intermediate,
                             response_method=response_method, name=name, ax=ax, pos_label=pos_label,
                             **class_average_kws
                             )
    # 多クラス分類のとき
    elif n_classes >= 3:
        # label_binarize()で目的変数を二値化
        y_train_binarize = label_binarize(y_train, classes=y_labels)
        y_test_binarize = label_binarize(y_test, classes=y_labels)
        # fit_paramsにeval_setがあるとき、二値化
        if 'eval_set' in fit_params:
            fit_params['eval_set'] = [(fit_params['eval_set'][0][0], label_binarize(fit_params['eval_set'][0][1], classes=y_labels))]
        # One vs Restの分類器を作成
        clf_ovr = OneVsRestClassifier(estimator)
        clf_ovr.fit(X_train, y_train_binarize) #TODO:あとでfit_paramsを追加 https://github.com/scikit-learn/scikit-learn/issues/10882
        y_score = clf_ovr.predict_proba(X_test)
        # クラスごとのROC曲線を算出
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarize[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # micro-averageしたROC曲線を算出
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # macro-averageしたROC曲線を算出
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # FPRのユニーク値を抽出
        mean_tpr = np.zeros_like(all_fpr)  # Then interpolate all ROC curves at this points
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Micro、Macroを選択
        fpr_avg = fpr[average]
        tpr_avg = tpr[average]
        roc_auc_avg = roc_auc[average]
        fpr_avg_graph = np.concatenate([np.array([0]), fpr_avg])  # グラフ表示用に端点を追加
        tpr_avg_graph = np.concatenate([np.array([0]), tpr_avg])  # グラフ表示用に端点を追加
        
        # 平均ROC曲線をプロット
        ax.plot(fpr_avg_graph, tpr_avg_graph,
                label=f'{average}' + '-average ROC (area = {0:0.2f})'
                    ''.format(roc_auc_avg),
                **class_average_kws)
        # クラスごとのROC曲線をプロット
        color_list = list(colors.TABLEAU_COLORS.values())
        for i, color in zip(range(n_classes), color_list):
            ax.plot(fpr[i], tpr[i], color=color,
                    label='ROC class {0} (area = {1:0.2f})'
                    ''.format(y_labels[i], roc_auc[i]),
                    **plot_roc_kws)

        # FPR、TPR、ROC曲線を保持
        name = estimator.__class__.__name__ if name is None else name
        viz = RocCurveDisplay(
            fpr=fpr_avg,
            tpr=tpr_avg,
            roc_auc=roc_auc_avg,
            estimator_name=name,
            pos_label=pos_label
        )
    
    return viz

def roc_plot(clf, x: List[str], y: str, data: pd.DataFrame = None,
            x_colnames: List[str] = None, 
            cv=5, cv_seed=42, cv_group=None,
            ax=None,
            clf_params=None, fit_params=None,
            subplot_kws=None,
            plot_roc_kws=None, class_average_kws=None, cv_mean_kws=None, chance_plot_kws=None):
    # 入力データの形式統一
    X, y_true, data, x_colnames, y_colname, cv_group_colname = _reshape_input_data(x, y, data,
                                                                                    x_colnames,
                                                                                    cv_group)

    # 学習器パラメータがあれば適用
    if clf_params is not None:
        clf.set_params(**clf_params)
    # 学習時パラメータがNoneなら空のdictを入力
    if fit_params is None:
        fit_params = {}
    # subplot_kwsがNoneなら空のdictを入力
    if subplot_kws is None:
        subplot_kws = {}
    # plot_roc_kwsがNoneなら空のdictを入力
    if plot_roc_kws is None:
        plot_roc_kws = {}
    # class_average_kwsがNoneなら空のdictを入力
    if class_average_kws is None:
        class_average_kws = {}
    # cv_mean_kwsがNoneなら空のdictを入力
    if cv_mean_kws is None:
        cv_mean_kws = {}
    # chance_plot_kwsがNoneなら空のdictを入力
    if chance_plot_kws is None:
        chance_plot_kws = {}

    # クロスバリデーション有無で場合分け
    # クロスバリデーション未実施時(学習データから学習してプロット)
    if cv is None:
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()
        # plot_roc_curveに渡す引数
        name = 'ROC'
        if 'alpha' not in plot_roc_kws.keys():
            plot_roc_kws['alpha'] = 0.5
        if 'lw' not in plot_roc_kws.keys():
            plot_roc_kws['lw'] = 1
        # ROC曲線をプロット
        viz = plot_roc_curve_multiclass(clf, X, y_true,
                            name=name, ax=ax, fit_params = fit_params,
                            plot_roc_kws=plot_roc_kws,
                            class_average_kws=class_average_kws
                            )
    
    # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
    if cv is not None:
        # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
        # LeaveOneOutのときエラーを出す
        if isinstance(cv, LeaveOneOut):
            raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupをグルーピング対象に指定
        split_kws={}
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if cv_group_colname is not None:
                split_kws['groups'] = data[cv_group_colname].values
            else:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need ``cv_group`` argument')
        # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
        if isinstance(cv, LeaveOneGroupOut):
            cv_num = len(set(data[cv_group_colname].values))
        else:
            cv_num = cv.n_splits

        # 表示用のax作成
        if ax is None:
            if 'figsize' not in subplot_kws.keys():
                subplot_kws['figsize'] = (6, (cv_num + 1) * 6)
            fig, ax = plt.subplots(cv_num + 1, 1, **subplot_kws)

        # 平均ROC曲線算出用のリスト
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        color_list = list(colors.TABLEAU_COLORS.values())
        # クロスバリデーション
        for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
            name = 'ROC fold {}'.format(i)
            # plot_roc_curveに渡す引数            
            if 'alpha' not in plot_roc_kws.keys():
                plot_roc_kws['alpha'] = 0.3
            if 'lw' not in plot_roc_kws.keys():
                plot_roc_kws['lw'] = 1
            # class_average_kwsに渡す引数
            if 'alpha' not in class_average_kws.keys():
                class_average_kws['alpha'] = 0.6
            if 'lw' not in class_average_kws.keys():
                class_average_kws['lw'] = 2
            if 'linestyle' not in class_average_kws.keys():
                class_average_kws['linestyle'] = ':'
            class_average_kws['color'] = color_list[i]
            # CVごとのROC曲線をプロット
            viz = plot_roc_curve_multiclass(clf, X[train], y_true[train], 
                                          X_test=X[test], y_test=y_true[test],
                                          name=name, ax=ax[i], fit_params=fit_params,
                                          plot_roc_kws=plot_roc_kws,
                                          class_average_kws=class_average_kws
                                          )
            ax[i].set_title(f'Cross Validation Fold{i}')
            # TPRとAUCを保持
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # データが存在しない部分を補完
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            # CVごとのROC曲線を全体図にプロット
            ax[cv_num].plot(mean_fpr, interp_tpr,
                            label=name, color=color_list[i],
                            **plot_roc_kws)
        
        # CV平均ROC曲線を計算
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        # CV平均ROC曲線plotに渡す引数
        if 'label' not in cv_mean_kws.keys():
                cv_mean_kws['label'] = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
        if 'alpha' not in cv_mean_kws.keys():
            cv_mean_kws['alpha'] = 0.8
        if 'lw' not in cv_mean_kws.keys():
            cv_mean_kws['lw'] = 2
        if 'color' not in cv_mean_kws.keys():
            cv_mean_kws['color'] = 'blue'
        # 平均ROC曲線プロット
        ax[cv_num].plot(mean_fpr, mean_tpr, **cv_mean_kws)
        ax[cv_num].set_title('All Cross Validations')

    # ランダム時の直線描画に渡す引数
    if 'label' not in chance_plot_kws.keys():
            chance_plot_kws['label'] = 'Chance'
    if 'alpha' not in chance_plot_kws.keys():
        chance_plot_kws['alpha'] = 0.8
    if 'lw' not in chance_plot_kws.keys():
        chance_plot_kws['lw'] = 2
    if 'color' not in chance_plot_kws.keys():
        chance_plot_kws['color'] = 'red'
    if 'linestyle' not in chance_plot_kws.keys():
        chance_plot_kws['linestyle'] = '--'
    # ランダム時の直線描画
    for ax_cv in ax if cv is not None else [ax]:
        ax_cv.plot([0, 1], [0, 1], **chance_plot_kws)
        ax_cv.legend(loc='lower right')
        ax_cv.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])

iris = sns.load_dataset("iris")
#iris = iris[iris['species'] != 'setosa']
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values

fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'rmse',
              'eval_set': [(X, y)]
              }

#estimator1 = LGBMClassifier(random_state=42, n_estimators=10000)
estimator1 = SVC(probability=True)
estimator2 = RandomForestClassifier(random_state=42)
fig, axes = plt.subplots(6, 2, figsize=(12, 36))
ax_pred = [[row[i] for row in axes] for i in range(2)]
roc_plot(estimator1, X, y, ax=ax_pred[0], cv=5)
roc_plot(estimator2, X, y, ax=ax_pred[1], cv=5)
# %%

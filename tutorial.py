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



# %% 2クラス分類でのROC曲線(RocCurveDisplay.from_estimator使用)
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
import numpy as np
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa'] 
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
# Plot ROC curve in binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
estimator = SVC(probability=True, random_state=42)
estimator.fit(X_train, y_train)
RocCurveDisplay.from_estimator(estimator, X_test, y_test)

# %% クロスバリデーション＆2クラス分類でのROC曲線
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
import numpy as np
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa'] 
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
# Plot ROC curve in binary classification
estimator = SVC(probability=True, random_state=42)
fig, axes = plt.subplots(4, 1, figsize=(6, 24))
cv = KFold(n_splits=3, shuffle=True, random_state=42)
classplot.roc_plot(estimator, X, y, ax=axes, cv=cv)

# %% 2クラス分類でのROC曲線(plot_roc_curve_multiclass使用)
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa'] 
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
# Plot ROC curve in binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
estimator = SVC(probability=True, random_state=42)
classplot.plot_roc_curve_multiclass(estimator, X_train, y_train, 
                                    X_test=X_test, y_test=y_test)

# %% 多クラス分類でのROC曲線(plot_roc_curve_multiclass使用)
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
# Plot ROC curve in multiclass classification
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
estimator = SVC(probability=True, random_state=42)
classplot.plot_roc_curve_multiclass(estimator, X_train, y_train, 
                                    X_test=X_test, y_test=y_test)
plt.plot([0, 1], [0, 1], label='Chance', alpha=0.8,
         lw=2, color='red', linestyle='--')
plt.legend(loc='lower right')

# %% クロスバリデーション＆多クラス分類でのROC曲線
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
import numpy as np
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
# Plot ROC curve with cross validation in binary classification
estimator = SVC(probability=True, random_state=42)
fig, axes = plt.subplots(4, 1, figsize=(6, 24))
classplot.roc_plot(estimator, X, y, ax=axes, cv=3)

# %% クロスバリデーション＆多クラス分類＆fit_params適用でのROC曲線
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'multi_logloss',
              'eval_set': [(X, y)]
              }
# Plot ROC curve with cross validation in multiclass classification
estimator = LGBMClassifier(random_state=42, n_estimators=10000)
fig, axes = plt.subplots(4, 1, figsize=(6, 24))
classplot.roc_plot(estimator, X, y, ax=axes, cv=3, fit_params=fit_params)

# %% 3つの分類アルゴリズムで2クラスROC曲線を描画
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa'] 
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'binary_logloss',
              'eval_set': [(X, y)]
              }
# Plot ROC curve with three classifiers
estimator1 = LGBMClassifier(random_state=42, n_estimators=10000)
estimator2 = SVC(probability=True, random_state=42)
estimator3 = RandomForestClassifier(random_state=42)
fig, axes = plt.subplots(4, 3, figsize=(18, 24))
ax_pred = [[row[i] for row in axes] for i in range(3)]
classplot.roc_plot(estimator1, X, y, ax=ax_pred[0], cv=3, fit_params=fit_params)
classplot.roc_plot(estimator2, X, y, ax=ax_pred[1], cv=3)
classplot.roc_plot(estimator3, X, y, ax=ax_pred[2], cv=3)
# Add etimator name to the graph
ax_pred[0][0].set_title(f'LightGBM\n\n{ax_pred[0][0].title._text}')
ax_pred[1][0].set_title(f'SVM\n\n{ax_pred[1][0].title._text}')
ax_pred[2][0].set_title(f'RandomForest\n\n{ax_pred[2][0].title._text}')

# %% 3つの分類アルゴリズムで多クラスROC曲線を描画（ノイズなし）
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'multi_logloss',
              'eval_set': [(X, y)]
              }
# Plot ROC curve with three classifiers
estimator1 = LGBMClassifier(random_state=42, n_estimators=10000)
estimator2 = SVC(probability=True, random_state=42)
estimator3 = RandomForestClassifier(random_state=42)
fig, axes = plt.subplots(4, 3, figsize=(18, 24))
ax_pred = [[row[i] for row in axes] for i in range(3)]
classplot.roc_plot(estimator1, X, y, ax=ax_pred[0], cv=3, fit_params=fit_params)
classplot.roc_plot(estimator2, X, y, ax=ax_pred[1], cv=3)
classplot.roc_plot(estimator3, X, y, ax=ax_pred[2], cv=3)
# Add etimator name to the graph
ax_pred[0][0].set_title(f'LightGBM\n\n{ax_pred[0][0].title._text}')
ax_pred[1][0].set_title(f'SVM\n\n{ax_pred[1][0].title._text}')
ax_pred[2][0].set_title(f'RandomForest\n\n{ax_pred[2][0].title._text}')

# %% 3つの分類アルゴリズムで多クラスROC曲線を描画（ノイズあり）
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from seaborn_analyzer import classplot
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Add random noise features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 10 * n_features)]
fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'multi_logloss',
              'eval_set': [(X, y)]
              }
# Plot ROC curve with three classifiers
estimator1 = LGBMClassifier(random_state=42, n_estimators=10000)
estimator2 = SVC(probability=True, random_state=42)
estimator3 = RandomForestClassifier(random_state=42)
fig, axes = plt.subplots(4, 3, figsize=(18, 24))
ax_pred = [[row[i] for row in axes] for i in range(3)]
classplot.roc_plot(estimator1, X, y, ax=ax_pred[0], cv=3, fit_params=fit_params)
classplot.roc_plot(estimator2, X, y, ax=ax_pred[1], cv=3)
classplot.roc_plot(estimator3, X, y, ax=ax_pred[2], cv=3)
# Add etimator name to the graph
ax_pred[0][0].set_title(f'LightGBM\n\n{ax_pred[0][0].title._text}')
ax_pred[1][0].set_title(f'SVM\n\n{ax_pred[1][0].title._text}')
ax_pred[2][0].set_title(f'RandomForest\n\n{ax_pred[2][0].title._text}')

# %%XGBoostのROC曲線描画
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn_analyzer import classplot
from xgboost import XGBClassifier
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # Objective variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explantory variables
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
NOT_OPT_PARAMS = {'random_state': 42, 'booster': 'gbtree', 'n_estimators': 10000, 'use_label_encoder': False}
BEST_PARAMS = {'learning_rate': 0.16380494993112965, 'min_child_weight': 1, 'max_depth': 9, 'colsample_bytree': 0.9885437044181186, 'subsample': 0.2762139608777358, 'reg_alpha': 0.09817943593356171, 'reg_lambda': 0.001895410328121496, 'gamma': 0.046138633197055}
params = {}
params.update(NOT_OPT_PARAMS)
params.update(BEST_PARAMS)
estimator = XGBClassifier()
estimator.set_params(**params)
FIT_PARAMS = {'verbose': 0, 'early_stopping_rounds': 10, 'eval_metric': 'logloss', 'eval_set': [(X, y)]}
fig, axes = plt.subplots(4, 1, figsize=(6, 24))
classplot.roc_plot(estimator, X, y,
                   ax=axes, cv=3, fit_params=FIT_PARAMS)


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

# %% average_plot
import seaborn as sns
from sklearn.svm import SVR
from seaborn_analyzer import regplot
iris = sns.load_dataset("iris")
svr = SVR()
features = ['petal_width', 'petal_length', 'sepal_width']
X = iris[features].values
y = iris['sepal_length'].values
regplot.average_plot(svr, X, y, x_colnames=features, cv=3)

# %% eval_set_selection引数の動作確認（regression_pred_true）
import seaborn as sns
iris = sns.load_dataset("iris")
from lightgbm import LGBMRegressor
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values

tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
fit_params={'verbose': 0,  # 学習中のコマンドライン出力
            'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'eval_metric': 'rmse',  # early_stopping_roundsの評価指標
            'eval_set': [(X, y)]
            }
estimator = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMRegressor())])
tuning.optuna_tuning(estimator=estimator)

params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = estimator.set_params(**params_after)
regplot.regression_pred_true(best_estimator, x=USE_EXPLANATORY,
                            y=OBJECTIVE_VARIABLE, data=df_reg,
                            scores='mse',
                            cv=tuning.cv,
                            fit_params=tuning.fit_params,
                            eval_set_selection='all'
                            )

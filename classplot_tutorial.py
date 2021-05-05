# %% 概要の「機能1」（決定境界）
import seaborn as sns
from sklearn.svm import SVC
from custom_scatter_plot import classplot
iris = sns.load_dataset("iris")
model = SVC()
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris)
# %% 概要の「機能2」（クラス確率）
import seaborn as sns
from sklearn.svm import SVC
from custom_scatter_plot import classplot
iris = sns.load_dataset("iris")
model = SVC(probability=True)
classplot.class_proba_plot(model, ['petal_width', 'petal_length'], 'species', iris,
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
model = SVC()  # SVM分類用インスタンス
model.fit(X, y)  # 学習
y_pred = model.predict(X)  # 推論
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
model.fit(X, y_int)
# mlxtendで決定境界可視化
plot_decision_regions(X, y_int[:, 0], clf=model,
                      colors='green,red,blue')
# %% 本ツールによる決定境界可視化
from custom_scatter_plot import classplot
model = SVC()
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris)
# %% ランダムフォレスト回帰での描画例
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris)
# %% 2分割クロスバリデーション
model = SVC()
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                               cv=2, display_cv_indices = [0, 1])
# %% 3次元特徴量
classplot.class_separator_plot(model, ['petal_width', 'petal_length', 'sepal_width'], 'species', iris,
                               x_chart=['petal_width', 'petal_length'],
                               pair_sigmarange = 1.0, pair_sigmainterval = 0.5)
# %% 4次元特徴量
classplot.class_separator_plot(model, ['petal_width', 'petal_length', 'sepal_width', 'sepal_length'], 'species', iris,
                               x_chart=['petal_width', 'petal_length'],
                               pair_sigmarange = 0.5, pair_sigmainterval = 0.5,
                               chart_scale=2)
# %% 本ツールによるクラス確率可視化（等高線表示）
model = SVC(probability=True)  # SVMでpredict_probaを有効にするため、引数"probability"をTrueに
classplot.class_proba_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='contourf')
# %% 本ツールによるクラス確率可視化（RGB画像表示）
classplot.class_proba_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='imshow')
# %% class_separator_plotの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from custom_scatter_plot import classplot
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
model = SVC()
classplot.class_separator_plot(model, x=['petal_width', 'petal_length'],
                               y='species', data=iris,
                               scatter_kws={'edgecolors': 'red'})
# %% class_proba_plotの参考図プロット用
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from xgboost import XGBClassifier
from custom_scatter_plot import classplot
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
model = SVC(probability=True)
classplot.class_proba_plot(model, x=['petal_width', 'petal_length'],
                           y='species', data=iris,
                           proba_type='imshow',
                           imshow_kws={'alpha':0.8})
# %% パイプライン
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])  # 標準化＋SVMパイプライン
classplot.class_separator_plot(pipe, x=['petal_width', 'petal_length'],
                             y='sepal_length', data=iris)
# %% 特徴量重要度
import seaborn as sns
iris = sns.load_dataset("iris")
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# モデルの学習
model = XGBClassifier()
features = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']
X = iris[features].values
y = iris['species'].values
model.fit(X, y)
# 特徴量重要度の取得と可視化
importances = list(model.feature_importances_)
plt.barh(features, importances)
# %% 主成分分析
import seaborn as sns
iris = sns.load_dataset("iris")
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from custom_scatter_plot import classplot
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
model = SVC()
classplot.class_separator_plot(model, x=['pc1', 'pc2'],
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
# %%

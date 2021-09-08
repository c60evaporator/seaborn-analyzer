# seaborn-analyzer
[![python](https://img.shields.io/pypi/pyversions/seaborn-analyzer)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/seaborn-analyzer?color=blue)](https://pypi.org/project/seaborn-analyzer/)
[![license](https://img.shields.io/pypi/l/seaborn-analyzer?color=blue)](https://github.com/c60evaporator/seaborn-analyzer/blob/master/LICENSE)

**A data analysis and visualization tool using Seaborn library.**

![image](https://user-images.githubusercontent.com/59557625/126887193-ceba9bdd-3653-4d58-a916-21dcfe9c38a0.png)

This documentation is Japanese language version.
**[English version is here](https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst)**

**[API reference is here](https://c60evaporator.github.io/seaborn-analyzer/)**

<br>

# 使用法
CustomPairPlotクラスの使用例
```python
from seaborn_analyzer import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')
```
※その他のクラスの使用法は[構成](https://github.com/c60evaporator/seaborn-analyzer#%E6%A7%8B%E6%88%90)の項を参照ください

※**[Qiitaにも記事を作成している](https://qiita.com/c60evaporator/items/8cb774e65d4423fd93ee)** ので、こちらもご覧ください

<br>

# 必要要件
* Python >=3.6
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Seaborn >=0.11.0
* Scipy >=1.6.3
* Scikit-learn >=0.24.2
<br>

# インストール方法
```
$ pip install seaborn-analyzer
```
<br>

# サポート
バグ等は[Issues](https://github.com/c60evaporator/seaborn-analyzer/issues)で報告してください

<br>

# 構成
seabornを利用して、各種データの可視化および評価指標の算出を実施します。
以下のクラスからなります
|クラス名|パッケージ名|概要|使用法|
|---|---|---|---|
|CustomPairPlot|custom_pair_plot.py|散布図行列と相関係数行列を同時に表示|[リンク](https://qiita.com/c60evaporator/items/20f11b6ee965cec48570)|
|hist|custom_hist_plot.py|ヒストグラムと各種分布のフィッティング|[リンク](https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42)|
|classplot|custom_scatter_plot.py|分類境界およびクラス確率の表示|[リンク](https://qiita.com/c60evaporator/items/43866a42e09daebb5cc0)|
|regplot|custom_scatter_plot.py|相関・回帰分析の散布図・ヒートマップ表示|[リンク](https://qiita.com/c60evaporator/items/c930c822b527f62796ee)|

<br>

## CustomPairPlotクラス (custom_pair_plot.py)
散布図行列と相関係数行列を同時に表示します。
1個のクラス「CustomPairPlot」からなります

**・CustomPairPlotクラス内のメソッド一覧**
|メソッド名|機能|
|---|---|
|pairanalyzer|散布図行列と相関係数行列を同時に表示します|
<br>

### pairanalyzerメソッド
#### 実行例
```python
from seaborn_analyzer import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')
```
![image](https://user-images.githubusercontent.com/59557625/115889860-4e8bde80-a48f-11eb-826a-cd3c79556a42.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|data|必須|pd.DataFrame|-|入力データ|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|palette|オプション|str|None|hueによる色分け用の[カラーパレット](https://matplotlib.org/stable/tutorials/colors/colormaps.html)|
|vars|オプション|list[str]|None|グラフ化するカラム名 (Noneなら全ての数値型＆Boolean型の列を使用)|
|lowerkind|オプション|str|'boxscatter'|左下に表示するグラフ種類 ('boxscatter', 'scatter', or 'reg')|
|diag_kind|オプション|str|'kde'|対角に表示するグラフ種類 ('kde' or 'hist')|
|markers|オプション|str or list[str]|None|hueで色分けしたデータの散布図プロット形状|
|height|オプション|float|2.5|グラフ1個の高さ|
|aspect|オプション|float|1|グラフ1個の縦横比|
|dropna|オプション|bool|True|[seaborn.PairGridのdropna引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|lower_kws|オプション|dict|{}|[seaborn.PairGrid.map_lowerの引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|diag_kws|オプション|dict|{}|[seaborn.PairGrid.map_diag引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|grid_kws|オプション|dict|{}|[seaborn.PairGridの上記以外の引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
<br>

### CustomPairPlotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/20f11b6ee965cec48570

<br>
<br>

## histクラス (custom_hist_plot.py)
ヒストグラム表示および各種分布のフィッティングを実行します。

**・histクラス内のメソッド一覧**
|メソッド名|機能|
|---|---|
|plot_normality|正規性検定とQQプロット|
|fit_dist|各種分布のフィッティングと、評価指標(RSS, AIC, BIC)の算出|

### plot_normalityメソッド
#### 実行例
```python
from seaborn_analyzer import hist
from sklearn.datasets import load_boston
import pandas as pd
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
hist.plot_normality(df, x='LSTAT', norm_hist=False, rounddigit=5)
```
![image](https://user-images.githubusercontent.com/59557625/117275256-cfd46f80-ae98-11eb-9da7-6f6e133846fa.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|data|必須|pd.DataFrame, pd.Series, or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|rounddigit|オプション|int|5|表示指標の小数丸め桁数|
|hist_kws|オプション|dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
|subplot_kws|オプション|dict|{}|[matplotlib.pyplot.subplotsに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)|
<br>

### fit_distメソッド
#### 実行例
```python
from seaborn_analyzer import hist
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
all_params, all_scores = hist.fit_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'])
df_scores = pd.DataFrame(all_scores).T
df_scores
```
![image](https://user-images.githubusercontent.com/59557625/115890066-81ce6d80-a48f-11eb-8390-f985d9e2b8b1.png)
![image](https://user-images.githubusercontent.com/59557625/115890108-8d219900-a48f-11eb-9896-38f7dedbb6e4.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|data|必須|pd.DataFrame, pd.Series, or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or list[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or list[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
<br>

### histクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42

<br>
<br>

## classplotクラス
分類の決定境界およびクラス確率の表示を実行します。<br>
Scikit-Learn APIに対応した分類モデル (例: XGBoostパッケージのXGBoostClassifierクラス)が表示対象となります

**・classplotクラス内のメソッド一覧**
|メソッド名|機能|
|---|---|
|class_separator_plot|決定境界プロット|
|class_proba_plot|クラス確率プロット|
|plot_roc_curve_multiclass|マルチクラス分類におけるROC曲線をプロット|
|roc_plot|クロスバリデーションにおけるROC曲線をプロット|

### class_separator_plotメソッド
#### 実行例
```python
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
clf = SVC()
classplot.class_separator_plot(clf, ['petal_width', 'petal_length'], 'species', iris)
```
![image](https://user-images.githubusercontent.com/59557625/117274234-d7474900-ae97-11eb-9de2-c8a74dc179a5.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|clf|必須|Scikit-learn API|-|表示対象の分類モデル|
|x|必須|list[str]|-|説明変数に指定するカラム名|
|y|必須|str|-|目的変数に指定するカラム名|
|data|必須|pd.DataFrame|-|入力データ|
|x_chart|オプション　　|list[str]|None|説明変数のうちグラフ表示対象のカラム名|
|pair_sigmarange|オプション|float|1.5|グラフ非使用変数の分割範囲|
|pair_sigmainterval|オプション|float|0.5|グラフ非使用変数の1枚あたり表示範囲|
|chart_extendsigma|オプション|float|0.5|グラフ縦軸横軸の表示拡張範囲|
|chart_scale|オプション|int|1|グラフの描画倍率|
|plot_scatter|オプション|str|'true'|散布図の描画種類|
|rounddigit_x3|オプション|int|2|グラフ非使用軸の小数丸め桁数|
|scatter_colors|オプション|list[str]|None|クラスごとのプロット色のリスト|
|true_marker|オプション|str|None|正解クラスの散布図プロット形状|
|false_marker|オプション|str|None|不正解クラスの散布図プロット形状|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|cv_group|オプション|str|None|GroupKFold,LeaveOneGroupOutのグルーピング対象カラム名|
|display_cv_indices|オプション|int|0|表示対象のクロスバリデーション番号|
|clf_params|オプション|dict|None|分類モデルに渡すパラメータ|
|fit_params|オプション|dict|None|学習時のパラメータ|
|subplot_kws|オプション|dict|None|[matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数|
|contourf_kws|オプション|dict|None|グラフ表示用の[matplotlib.pyplot.contourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)に渡す引数|
|scatter_kws|オプション|dict|None|散布図用の[matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### class_proba_plotメソッド
#### 実行例
```python
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
clf = SVC()
classplot.class_proba_plot(clf, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='imshow')
```
![image](https://user-images.githubusercontent.com/59557625/117276085-a1a35f80-ae99-11eb-8368-cdd1cfa78346.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|clf|必須|Scikit-learn API|-|表示対象の分類モデル|
|x|必須|list[str]|-|説明変数に指定するカラム名|
|y|必須|str|-|目的変数に指定するカラム名|
|data|必須|pd.DataFrame|-|入力データ|
|x_chart|オプション　　　|list[str]|None|説明変数のうちグラフ表示対象のカラム名|
|pair_sigmarange|オプション|float|1.5|グラフ非使用変数の分割範囲|
|pair_sigmainterval|オプション|float|0.5|グラフ非使用変数の1枚あたり表示範囲|
|chart_extendsigma|オプション|float|0.5|グラフ縦軸横軸の表示拡張範囲|
|chart_scale|オプション|int|1|グラフの描画倍率|
|plot_scatter|オプション|str|'true'|散布図の描画種類|
|rounddigit_x3|オプション|int|2|グラフ非使用軸の小数丸め桁数|
|scatter_colors|オプション|list[str]|None|クラスごとのプロット色のリスト|
|true_marker|オプション|str|None|正解クラスの散布図プロット形状|
|false_marker|オプション|str|None|不正解クラスの散布図プロット形状|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|cv_group|オプション|str|None|GroupKFold, LeaveOneGroupOutのグルーピング対象カラム名|
|display_cv_indices|オプション|int|0|表示対象のクロスバリデーション番号|
|clf_params|オプション|dict|None|分類モデルに渡すパラメータ|
|fit_params|オプション|dict|None|学習時のパラメータ|
|subplot_kws|オプション|dict|None|[matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数|
|contourf_kws|オプション|dict|None|proba_type='contour'のとき[matplotlib.pyplot.contourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)、 proba_type='contour'のとき[contour](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html))に渡す引数|
|scatter_kws|オプション|dict|None|散布図用の[matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)に渡す引数|
|plot_border|オプション|bool|True|決定境界線の描画有無|
|proba_class|オプション|str or list[str]|None|確率表示対象のクラス名|
|proba_cmap_dict|オプション|dict[str, str]|None|クラス確率図のカラーマップ(クラス名と[colormap](https://matplotlib.org/stable/tutorials/colors/colormaps.html)をdict指定)|
|proba_type|オプション|str|'contourf'|クラス確率図の描画種類<br>(等高線'contourf', 'contour', or RGB画像'imshow')|
|imshow_kws|オプション|dict|None|proba_type='imshow'のとき[matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### plot_roc_curve_multiclassメソッド
マルチクラス分類でのROC曲線をプロットします（2クラス分類もプロット可）
#### 実行例
```python
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
```
![outputmulti](https://user-images.githubusercontent.com/59557625/132558369-c6bfee32-156b-4043-bedb-5b1854b00660.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|必須|Scikit-learn API|-|表示対象の分類モデル(**未学習かつパラメータ入力済**の分類器を渡してください)|
|X_train|必須|np.ndarray|-|学習用データのうち説明変数|
|y_train|必須|np.ndarray|-|学習用データのうち目的変数|
|X_test|オプション|np.ndarray|-|ROC曲線評価用データのうち説明変数|
|y_test|オプション　　|np.ndarray|None|ROC曲線評価用データのうち目的変数|
|sample_weight|オプション|list[float]|None|ROC曲線算出時のクラスごとの重みづけ|
|drop_intermediate|オプション|bool|True|ROC曲線の形状に影響しない点の計算を省略するか|
|response_method|オプション|{'predict_proba', 'decision_function'}|predict_proba|クラス確率算出に使用するメソッド名|
|name|オプション|str|None|学習器の名称|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|pos_label|オプション|str or int|None|Positive判定するラベル番号、2クラス分類のみ有効|
|average|オプション|list[str]|None|クラスごとのプロット色のリスト|
|fit_params|オプション|dict|None|学習器の`fit()`メソッドに渡すパラメータ|
|plot_roc_kws|オプション|dict|None|クラスごとROC曲線描画用の`ax.plot()`メソッドに渡すパラメータ|
|class_average_kws|オプション|dict|None|全クラス平均ROC曲線描画用の`ax.plot()`メソッドに渡すパラメータ|

<br>

### roc_plotメソッド
クロスバリデーションにおけるROC曲線をプロットします。通常の多クラス分類では渡せないfit_paramsを渡すこともできます
#### 実行例
```python
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
              'eval_metric': 'rmse',
              'eval_set': [(X, y)]
              }
# Plot ROC curve with cross validation in multiclass classification
estimator = LGBMClassifier(random_state=42, n_estimators=10000)
fig, axes = plt.subplots(4, 1, figsize=(6, 24))
classplot.roc_plot(estimator, X, y, ax=axes, cv=3, fit_params=fit_params)
```
![outputcv](https://user-images.githubusercontent.com/59557625/132558249-77f742f7-7af2-4da7-9d22-5456b21f4234.png)

#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|clf|必須|Scikit-learn API|-|表示対象の分類モデル(**未学習**の分類器を渡してください)|
|X|必須|list[str] or np.ndarray|-|説明変数のカラム名、または入力データのうち説明変数|
|y|必須|str or np.ndarray|-|目的変数のカラム名、または入力データのうち目的変数|
|data|オプション　　.|pd.DataFrame|None|入力データ (`X`, `y`がstrのとき必須)|
|x_columns|オプション|list[str]|-|説明変数の名称リスト(`data`がNoneのときのみ有効)|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|cv_group|オプション|str or np.ndarray|None|GroupKFold, LeaveOneGroupOutのグルーピング対象カラム名、またはグルーピングのラベルデータ|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|sample_weight|オプション|list[float]|None|ROC曲線算出時のクラスごとの重みづけ|
|drop_intermediate|オプション|bool|True|ROC曲線の形状に影響しない点の計算を省略するか|
|response_method|オプション|{'predict_proba', 'decision_function'}|predict_proba|クラス確率算出に使用するメソッド名|
|pos_label|オプション|str or int|None|Positive判定するラベル番号、2クラス分類のみ有効|
|average|オプション|list[str]|None|クラスごとのプロット色のリスト|
|clf_params|オプション|dict|None|学習器に渡すパラメータ|
|fit_params|オプション|dict|None|学習器の`fit()`メソッドに渡すパラメータ|
|draw_grid|オプション|bool|True|グリッド線の描画有無|
|grid_kws|オプション|dict|None|グリッド描画用の[`matplotlib.pyplot.grid()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html)メソッドに渡すパラメータ|
|subplot_kws|オプション|dict|None|図作成用の[`matplotlib.pyplot.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数(ax=Noneのときのみ有効)|
|plot_roc_kws|オプション|dict|None|クラスごとROC曲線描画用の[`matplotlib.pyplot.plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)メソッドに渡すパラメータ|
|class_average_kws|オプション|dict|None|全クラス平均ROC曲線描画用の[`matplotlib.pyplot.plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)メソッドに渡すパラメータ|
|cv_mean_kws|オプション|dict|None|全クラス平均ROC曲線描画用の[`matplotlib.pyplot.plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)メソッドに渡すパラメータ|
|chance_plot_kws|オプション|dict|None|ランダム直線描画用の[`matplotlib.pyplot.plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)メソッドに渡すパラメータ|
<br>

### classplotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/43866a42e09daebb5cc0


<br>
<br>

## regplotクラス
相関・回帰分析の散布図・ヒートマップ表示を実行します。<br>
Scikit-Learn APIに対応した回帰モデル (例: XGBoostパッケージのXGBoostRegressorクラス)が表示対象となります

**・regplotクラス内のメソッド一覧**
|メソッド名|機能|
|---|---|
|linear_plot|ピアソン相関係数とP値を散布図と共に表示|
|regression_pred_true|予測値vs実測値プロット|
|regression_plot_1d|1次元説明変数で回帰線表示|
|regression_heat_plot|2～4次元説明変数で回帰予測値をヒートマップ表示|

### linear_plotメソッド
#### 実行例
```python
from seaborn_analyzer import regplot
import seaborn as sns
iris = sns.load_dataset("iris")
regplot.linear_plot(x='petal_length', y='sepal_length', data=iris)
```
![image](https://user-images.githubusercontent.com/59557625/117276994-65243380-ae9a-11eb-8ec8-fa1fb5d60a55.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|x|必須|str|-|横軸に指定するカラム名|
|y|必須|str|-|縦軸に指定するカラム名|
|data|必須|pd.DataFrame|-|入力データ|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のAxes (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|hue|オプション|str|None|色分けに指定するカラム名|
|linecolor|オプション|str|'red'|回帰直線の[色](https://matplotlib.org/stable/gallery/color/named_colors.html)|
|rounddigit|オプション|int|5|表示指標の小数丸め桁数|
|plot_scores|オプション|bool|True|回帰式、ピアソンの相関係数およびp値の表示有無|
|scatter_kws|オプション|dict|None|[seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### regression_pred_trueメソッド
#### 実行例
```python
import pandas as pd
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.linear_model import LinearRegression
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
regplot.regression_pred_true(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
```
![image](https://user-images.githubusercontent.com/59557625/117277036-6fdec880-ae9a-11eb-887a-5f8b2a93b0f9.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|必須|Scikit-learn API|-|表示対象の回帰モデル|
|x|必須|list[str]|-|説明変数に指定するカラム名のリスト|
|y|必須|str|-|目的変数に指定するカラム名|
|data|必須|pd.DataFrame|-|入力データ|
|hue|オプション　　　|str|None|色分けに指定するカラム名|
|linecolor|オプション|str|'red'|予測値=実測値の[線の色](https://matplotlib.org/stable/gallery/color/named_colors.html)|
|rounddigit|オプション|int|3|表示指標の小数丸め桁数|
|rank_number|オプション|int|None|誤差上位何番目までを文字表示するか|
|rank_col|オプション|str|None|誤差上位と一緒に表示するフィールド名|
|scores|オプション|str or list[str]|'mae'|文字表示する評価指標を指定 ('r2', 'mae', 'mse', 'rmse', 'rmsle', or 'max_error')|
|cv_stats|オプション|str|'mean'|クロスバリデーション時に表示する評価指標統計値 ('mean', 'median', 'max', or 'min')|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|estimator_params|オプション|dict|None|回帰モデルに渡すパラメータ|
|fit_params|オプション|dict|None|学習時のパラメータをdict指定|
|subplot_kws|オプション|dict|None|[matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数|
|scatter_kws|オプション|dict|None|[seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### regression_plot_1dメソッド
#### 実行例
```python
from seaborn_analyzer import regplot
import seaborn as sns
from sklearn.svm import SVR
iris = sns.load_dataset("iris")
regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris)
```
![image](https://user-images.githubusercontent.com/59557625/117277075-78cf9a00-ae9a-11eb-835c-01f635754f7b.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|必須|Scikit-learn API|-|表示対象の回帰モデル|
|x|必須|str|-|説明変数に指定するカラム名|
|y|必須|str|-|目的変数に指定するカラム名|
|data|必須|pd.DataFrame|-|入力するデータ（Pandasのデータフレーム）|
|hue|オプション　　　|str|None|色分けに指定するカラム名|
|linecolor|オプション|str|'red'|予測値=実測値の[線の色](https://matplotlib.org/stable/gallery/color/named_colors.html)|
|rounddigit|オプション|int|3|表示指標の小数丸め桁数|
|rank_number|オプション|int|None|誤差上位何番目までを文字表示するか|
|rank_col|オプション|str|None|誤差上位と一緒に表示するフィールド名|
|scores|オプション|str or list[str]|'mae'|文字表示する評価指標を指定 ('r2', 'mae', 'mse', 'rmse', 'rmsle', or 'max_error')|
|cv_stats|オプション|str|'mean'|クロスバリデーション時に表示する評価指標統計値 ('mean', 'median', 'max', or 'min')|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|estimator_params|オプション|dict|None|回帰モデルに渡すパラメータ|
|fit_params|オプション|dict|None|学習時のパラメータをdict指定|
|subplot_kws|オプション|dict|None|[matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数|
|scatter_kws|オプション|dict|None|[seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### regression_heat_plotメソッド
#### 実行例
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from seaborn_analyzer import regplot
df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
regplot.regression_heat_plot(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
```
![image](https://user-images.githubusercontent.com/59557625/115955837-1b4f5b00-a534-11eb-91b0-b913019d26ff.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|必須|Scikit-learn API|-|表示対象の回帰モデル|
|x|必須|list[str]|-|説明変数に指定するカラム名のリスト|
|y|必須|str|-|目的変数に指定するカラム名|
|data|必須|pd.DataFrame|-|入力データ|
|x_heat|オプション　　　|list[str]|None|説明変数のうちヒートマップ表示対象のカラム名|
|scatter_hue|オプション|str|None|散布図色分け指定カラム名 (plot_scatter='hue'時のみ有効)|
|pair_sigmarange|オプション|float|1.5|ヒートマップ非使用変数の分割範囲|
|pair_sigmainterval|オプション|float|0.5|ヒートマップ非使用変数の1枚あたり表示範囲|
|heat_extendsigma|オプション|float|0.5|ヒートマップ縦軸横軸の表示拡張範囲|
|heat_division|オプション|int|30|ヒートマップ縦軸横軸の解像度|
|value_extendsigma|オプション|float|0.5|ヒートマップの色分け最大最小値拡張範囲|
|plot_scatter|オプション|str|'true'|散布図の描画種類|
|rounddigit_rank|オプション|int|3|誤差上位表示の小数丸め桁数|
|rounddigit_x1|オプション|int|2|ヒートマップ横軸の小数丸め桁数|
|rounddigit_x2|オプション|int|2|ヒートマップ縦軸の小数丸め桁数|
|rounddigit_x3|オプション|int|2|ヒートマップ非使用軸の小数丸め桁数|
|rank_number|オプション|int|None|誤差上位何番目までを文字表示するか|
|rank_col|オプション|str|None|誤差上位と一緒に表示するフィールド名|
|cv|オプション|int or sklearn.model_selection.* |None|クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)|
|cv_seed|オプション|int|42|クロスバリデーションの乱数シード|
|display_cv_indices|オプション|int|0|表示対象のクロスバリデーション番号|
|estimator_params|オプション|dict|None|回帰モデルに渡すパラメータ|
|fit_params|オプション|dict|None|学習時のパラメータをdict指定|
|subplot_kws|オプション|dict|None|[matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)に渡す引数|
|heat_kws|オプション|dict|None|ヒートマップ用の[seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)に渡す引数|
|scatter_kws|オプション|dict|None|散布図用の[matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)に渡す引数|
|legend_kws|オプション|dict|None|凡例用の[matplotlib.axes.Axes.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)に渡す引数|
<br>

### regplotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/c930c822b527f62796ee

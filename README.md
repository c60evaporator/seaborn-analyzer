# seaborn-analyzer
[![python](https://img.shields.io/pypi/pyversions/seaborn-analyzer)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/seaborn-analyzer?color=blue)](https://img.shields.io/pypi/l/seaborn-analyzer?color=blue)
[![license](https://img.shields.io/pypi/l/seaborn-analyzer?color=blue)](https://github.com/c60evaporator/seaborn-analyzer/LICENSE)

**A data analysis and visualization tool using Seaborn library.**

![image](https://user-images.githubusercontent.com/59557625/126887193-ceba9bdd-3653-4d58-a916-21dcfe9c38a0.png)

This documentation is currently Japanese language only.

If there is a request for English version documantation, we will make it.

Therefore, please let us know if you have any requests for us.

<br>

# 使用法
```python
from seaborn_analyzer import CustomPairPlot
import seaborn as sns

titanic = sns.load_dataset("titanic")
cp = CustomPairPlot()
cp.pairanalyzer(titanic, hue='survived')
```
<br>

# 必要要件
* Python >=3.6
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Scipy >=1.6.3
* Scikit-learn >=0.24.2
<br>

# インストール方法
```
pip install seaborn-analyzer
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
|CustomPairPlot|custom_pair_plot.py|散布図行列と相関係数行列を同時に表示|[リンク](https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42)|
|hist|custom_hist_plot.py|ヒストグラムと各種分布のフィッティング|[リンク](https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42)|
|classplot|custom_scatter_plot.py|分類境界およびクラス確率の表示|[リンク](https://qiita.com/c60evaporator/items/43866a42e09daebb5cc0)|
|regplot|custom_scatter_plot.py|相関・回帰分析の散布図・ヒートマップ表示|[リンク](https://qiita.com/c60evaporator/items/c930c822b527f62796ee)|

<br>

## CustomPairPlotクラス (custom_pair_plot.py)
散布図行列と相関係数行列を同時に表示します。
1個のクラス「CustomPairPlot」からなります

**CustomPairPlotクラス内のメソッド一覧**
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
|vars|オプション|List[str]|None|グラフ化するカラム名 (Noneなら全ての数値型＆Boolean型の列を使用)|
|lowerkind|オプション|str|'boxscatter'|左下に表示するグラフ種類 ('boxscatter', 'scatter', or 'reg')|
|diag_kind|オプション|str|'kde'|対角に表示するグラフ種類 ('kde' or 'hist')|
|markers|オプション|str or List[str]|None|hueで色分けしたデータの散布図プロット形状|
|height|オプション|float|2.5|グラフ1個の高さ|
|aspect|オプション|float|1|グラフ1個の縦横比|
|dropna|オプション|str|True|[seaborn.PairGridのdropna引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|lower_kws|オプション|Dict|{}|[seaborn.PairGridのlower_kws引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|diag_kws|オプション|Dict|{}|[seaborn.PairGridのdiag_kws引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|grid_kws|オプション|Dict|{}|[seaborn.PairGridのgrid_kws引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|size|オプション|Dict|None|[seaborn.PairGridのsize引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
<br>

### CustomPairPlotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/20f11b6ee965cec48570

<br>
<br>

## histクラス (custom_hist_plot.py)
ヒストグラム表示および各種分布のフィッティングを実行します。

**histクラス内のメソッド一覧**
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|rounddigit|オプション|int|5|表示指標の小数丸め桁数|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
|subplot_kws|オプション|Dict|{}|[matplotlib.pyplot.subplotsに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)|
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
<br>

### histクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42

<br>
<br>

## classplotクラス
分類の決定境界およびクラス確率の表示を実行します。
Scikit-Learn APIに対応した分類モデル (例: XGBoostパッケージのXGBoostClassifierクラス)が表示対象となります
**classplotクラス内のメソッド一覧**
|メソッド名|機能|
|---|---|
|class_separator_plot|決定境界プロット|
|class_proba_plot|クラス確率プロット|

### class_separator_plotメソッド
#### 実行例
```python
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
model = SVC()
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris,)
```
![image](https://user-images.githubusercontent.com/59557625/117274234-d7474900-ae97-11eb-9de2-c8a74dc179a5.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
<br>

### class_proba_plotメソッド
####実行例
```python
import seaborn as sns
from sklearn.svm import SVC
from seaborn_analyzer import classplot
iris = sns.load_dataset("iris")
model = SVC()
classplot.class_proba_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                           proba_type='imshow')
```
![image](https://user-images.githubusercontent.com/59557625/117276085-a1a35f80-ae99-11eb-8368-cdd1cfa78346.png)
#### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
<br>

### classplotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/43866a42e09daebb5cc0

<br>
<br>

## regplotクラス
相関・回帰分析の散布図・ヒートマップ表示を実行します。
Scikit-Learn APIに対応した回帰モデル (例: XGBoostパッケージのXGBoostRegressorクラス)が表示対象となります
**regplotクラス内のメソッド一覧**
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
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
|data|必須|pd.DataFrame or pd.Series or pd.ndarray|-|入力データ|
|x|オプション|str|None|ヒストグラム作成対象のカラム名 (dataがpd.DataFrameのときは必須)|
|hue|オプション|str|None|色分けに指定するカラム名 (Noneなら色分けなし)|
|binwidth|オプション|float|None|ビンの幅 (binsと共存不可)|
|bins|オプション|int|'auto'|ビンの数 (bin_widthと共存不可、'auto'なら[スタージェスの公式](https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html)で自動決定)|
|norm_hist|オプション|bool|False|ヒストグラムを面積1となるよう正規化するか？|
|sigmarange|オプション|float|4|フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)|
|linesplit|オプション|float|200|フィッティング線の分割数 (カクカクしたら増やす)
|dist|オプション|str or List[str]|'norm'|分布の種類 ('norm', 'lognorm', 'gamma', 't', 'expon', 'uniform', 'chi2', 'weibull')|
|ax|オプション|matplotlib.axes._ subplots.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|linecolor|オプション|str or List[str]|'red'|フィッティング線の[色指定](https://matplotlib.org/stable/gallery/color/named_colors.html) (Listで複数指定可)|
|floc|オプション|float|None|フィッティング時のX方向オフセット (Noneなら指定なし(weibullとexponは0))|
|hist_kws|オプション|Dict|{}|[matplotlib.axes.Axes.histに渡す引数](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)|
<br>

### regplotクラス使用法詳細
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/c930c822b527f62796ee

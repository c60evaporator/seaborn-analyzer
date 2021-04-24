# seaborn_analyzer
seabornを利用して、各種データの可視化および評価指標の算出を実施します。
以下のモジュールからなります
|パッケージ名|概要|
|---|---|
|custom_pair_plot.py|散布図行列と相関係数行列を同時に表示|
|custom_hist_plot.py|ヒストグラムと各種分布のフィッティング|
|custom_scatter_plot.py|相関分析、回帰および分類結果の散布図表示|

## custom_pair_plot.py
散布図行列と相関係数行列を同時に表示するパッケージです。
1個のクラス「CustomPairPlot」からなります
![image](https://user-images.githubusercontent.com/59557625/115889860-4e8bde80-a48f-11eb-826a-cd3c79556a42.png)

### CustomPairPlotクラス内のメソッド一覧
|メソッド名|機能|
|---|---|
|pairanalyzer|散布図行列と相関係数行列を同時に表示します|

### 使用法
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/20f11b6ee965cec48570

<br>

## custom_hist_plot.py
ヒストグラム表示と各種分布のフィッティングを実行するパッケージです。
1個のクラス「hist」からなります
![image](https://user-images.githubusercontent.com/59557625/115890066-81ce6d80-a48f-11eb-8390-f985d9e2b8b1.png)
![image](https://user-images.githubusercontent.com/59557625/115890108-8d219900-a48f-11eb-9896-38f7dedbb6e4.png)

### histクラス内のメソッド一覧
|メソッド名|機能|
|---|---|
|plot_normality|正規性検定とQQプロット|
|fit_dist|各種分布のフィッティングと、評価指標(RSS, AIC, BIC)の算出|

### 使用法
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42

## custom_scatter_plot.py
相関分析、回帰および分類結果の散布図表示を実行するライブラリです。
2個のクラス「regplot」「classplot」からなります。
Scikit-Learn APIに対応した回帰・分類モデルが表示対象となります
|クラス名|機能|
|---|---|
|regplot|相関分析および回帰結果の可視化|
|classplot|分類結果の可視化|

**regplot**

![image](https://user-images.githubusercontent.com/59557625/115955837-1b4f5b00-a534-11eb-91b0-b913019d26ff.png)


### regplotクラス内のメソッド一覧
|メソッド名|機能|
|---|---|
|linear_plot|ピアソン相関係数とP値を散布図と共に表示|
|regression_pred_true|予測値vs実測値プロット|
|regression_plot_1d|1次元説明変数で回帰線表示|
|regression_heat_plot|2～4次元説明変数で回帰予測値をヒートマップ表示|

### 使用法
#### regplot
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/c930c822b527f62796ee

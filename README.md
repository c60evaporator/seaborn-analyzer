# seaborn_analyzer
seabornを利用して、
以下のパッケージからなります
|パッケージ名|概要|
|---|---|
|custom_pair_plot.py|散布図行列と相関係数行列を同時に表示|
|custom_hist_plot.py|ヒストグラムと各種分布のフィッティング|
|custom_scatter_plot.py|相関分析、回帰および分類結果の散布図表示|

## custom_pair_plot.py
散布図行列と相関係数行列を同時に表示するライブラリです。
1個のクラス「CustomPairPlot」からなります

### histクラス内のメソッド一覧
pairanalyzer：散布図行列と相関係数行列を同時に表示します

### 使用法
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/20f11b6ee965cec48570

<br>

## custom_hist_plot.py
ヒストグラム表示と各種分布のフィッティングを実行するライブラリです。
1個のクラス「hist」からなります
### histクラス内のメソッド一覧
plot_normality: 正規性検定とQQプロット<br>
fit_dist: 各種分布のフィッティングと、評価指標(RSS, AIC, BIC)の算出

### 使用法
こちらの記事にまとめました
https://qiita.com/c60evaporator/items/fc531aff0cdbafac0f42

## custom_scatter_plot.py
記事作成中

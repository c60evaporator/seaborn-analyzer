from typing import List, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class dist():
    # 正規分布のフィッティング
    def _fit_norm(X: np.ndarray, sigmarange, linesplit):
        # 平均と不偏標準偏差
        mean = np.mean(X)
        std = np.std(X, ddof=1)
        # シャピロウィルク検定
        shapiro=stats.shapiro(X)
        # フィッティング曲線の生成
        Xline = np.linspace(min(mean - std * sigmarange, np.amin(X)), max(mean + std * sigmarange, np.amax(X)), linesplit)
        Yline = stats.norm.pdf(Xline, mean, std)
        # 結果の保存と出力
        params = {'mean':mean,
                'std':std,
                'statistic':shapiro.statistic,
                'pvalue':shapiro.pvalue,
            }
        return params, Xline, Yline
    
    # QQプロット
    @classmethod
    def qqplot(cls, data: pd.Series, ax=None):
        print('a')

    @classmethod
    def hist_dist(cls, data: pd.Series, ax=None, dist='norm', sigmarange=4, linecolor='red', linesplit=50):
        # まずはヒストグラム描画
        sns.distplot(data, ax=ax, kde=False, norm_hist=True)

        # 分布をフィッティング
        X = data.values
        if dist == 'norm':
            params, Xline ,Yline = cls._fit_norm(X, sigmarange, linesplit)
            
        ax.plot(Xline, Yline, color=linecolor)

        # パラメータを記載

        return params

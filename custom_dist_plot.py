from typing import List, Dict, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class dist():
    def _fit_norm(X: np.ndarray, sigmarange: float, linesplit: int):
        """
        正規分布のフィッティング

        Parameters
        ----------
        X : ndarray
            フィッティング対象のデータ
        sigmarange : float
            フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
        linesplit : int
            フィッティング線の分割数（カクカクしたら増やす）
        """
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

    def _round_dict_digits(srcdict: Dict[str, float], rounddigit: int = None):
        dstdict = {}
        for k, v in srcdict.items():
            if isinstance(v, float):
                dstdict[k] = round(v, rounddigit)
            else:
                dstdict[k] = v
        return dstdict
    
    # QQプロット
    @classmethod
    def qqplot(cls, data: pd.Series, ax=None):
        print('a')

    @classmethod
    def hist_dist(cls, data: pd.Series, ax=None, dist='norm', sigmarange=4, linecolor='red', linesplit=50, rounddigit=None):
        """
        分布フィッティングと各指標の表示

        Parameters
        ----------
        data : pd.Series
            フィッティング対象のデータ
        ax : 
            表示対象のax（Noneなら新規作成）
        dist : str
            分布の種類 ("norm", "poisson", "exp", "chi2", "weibull", "lognorm", "gamma")
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # まずはヒストグラム描画
        sns.distplot(data, ax=ax, kde=False, norm_hist=True)

        # 分布をフィッティング
        X = data.values
        params = Xline = Yline = None
        # 正規分布
        if dist == 'norm':
            params, Xline ,Yline = cls._fit_norm(X, sigmarange, linesplit)
        # フィッティング線の描画
        ax.plot(Xline, Yline, color=linecolor)

        # パラメータを記載
        param_list = [f'{k}={v}' for k, v in cls._round_dict_digits(params, rounddigit).items()]
        param_text = "\n".join(param_list)
        ax.text(np.amax(Xline), np.amax(Yline), param_text, verticalalignment='top', horizontalalignment='right')

        return params

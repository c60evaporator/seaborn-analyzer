from typing import List, Dict, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import decimal

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
    
    def _round_digits(src: float, rounddigit: int = None, method='decimal'):
        """
        指定桁数で小数を丸める

        Parameters
        ----------
        srcdict : Dict[str, float]
            丸め対象のDict
        rounddigit : int
            フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
        method : int
            桁数決定手法（'decimal':小数点以下, 'sig':有効数字(Decimal指定), 'format':formatで有効桁数指定）
        """
        if method == 'decimal':
            return round(src, rounddigit)
        elif method == 'sig':
            with decimal.localcontext() as ctx:
                ctx.prec = rounddigit
                return ctx.create_decimal(src)
        elif method == 'format':
            return '{:.{width}g}'.format(src, width=rounddigit)

    @classmethod
    def _round_dict_digits(cls, srcdict: Dict[str, float], rounddigit: int = None, method='decimal'):
        """
        指定桁数でDictの値を丸める

        Parameters
        ----------
        srcdict : Dict[str, float]
            丸め対象のDict
        rounddigit : int
            フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
        method : int
            桁数決定手法（'decimal':小数点以下, 'sig':有効数字(Decimal指定), 'format':formatで有効桁数指定）
        """
        dstdict = {}
        for k, v in srcdict.items():
            if rounddigit is not None and isinstance(v, float):
                dstdict[k] = cls._round_digits(v, rounddigit=rounddigit, method=method)
            else:
                dstdict[k] = v
        return dstdict
    
    # QQプロット
    @classmethod
    def qqplot(cls, data: pd.Series, dist='norm', ax=None):
        stats.probplot(data, dist=dist, plot=ax)

    @classmethod
    def hist_dist(cls, data: pd.Series, dist='norm', ax=None, bin_width=None, norm_hist=True, hist_kws={}, sigmarange=4, linecolor='red', linesplit=50, rounddigit=None):
        """
        分布フィッティングと各指標の表示

        Parameters
        ----------
        data : pd.Series
            フィッティング対象のデータ
        dist : str
            分布の種類 ("norm", "poisson", "exp", "chi2", "weibull", "lognorm", "gamma")
        ax : matplotlib.axes._subplots.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        bin_width : float
            ビンの幅 (NoneならFreedman-Diaconis ruleで自動決定)
        norm_hist : bool
            ヒストグラムを面積1となるよう正規化するか？
        hist_kws : Dict
            ヒストグラム表示(matplotlib.axes.Axes.hist())の引数
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # まずはヒストグラム描画
        if bin_width is None:
            bins = None
        else:
            bins = np.arange(np.floor(data.min()), np.ceil(data.max()), bin_width)
        ax = sns.distplot(data, ax=ax, kde=False, bins=bins, norm_hist=norm_hist, hist_kws=hist_kws)

        # 分布をフィッティング
        X = data.values
        params = Xline = Yline = None
        # 正規分布
        if dist == 'norm':
            params, Xline ,Yline = cls._fit_norm(X, sigmarange, linesplit)
        # 標準化していないとき、ヒストグラムと最大値の8割を合わせるようフィッティング線の倍率調整
        if norm_hist is False:
            line_max = np.amax(Yline)
            hist_max = ax.get_ylim()[1]
            Yline = Yline * hist_max / line_max * 0.8
        # フィッティング線の描画
        ax.plot(Xline, Yline, color=linecolor)

        # パラメータを記載
        param_list = [f'{k}={v}' for k, v in cls._round_dict_digits(params, rounddigit, 'sig').items()]
        param_text = "\n".join(param_list)
        ax.text(np.amax(Xline), np.amax(Yline), param_text, verticalalignment='top', horizontalalignment='right')

        return params

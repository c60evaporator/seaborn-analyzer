from typing import List, Dict, Tuple
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import distributions
import decimal

class dist():
    DEFAULT_LINECOLORS = ['red', 'darkmagenta', 'hotpink', 'yellow', 'brown', 'blue', 'green', 'cyan', 'orange']
    
    def _fit_distribution(x: np.ndarray, distribution: distributions, sigmarange: float, linesplit: int):
        """
        分布のフィッティング

        Parameters
        ----------
        X : ndarray
            フィッティング対象のデータ
        dist : scipy.stats.distributions
            分布の種類
        sigmarange : float
            フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
        linesplit : int
            フィッティング線の分割数（カクカクしたら増やす）
        """
        # 表示範囲指定用に平均と不偏標準偏差計算(正規分布のときを基準に)
        mean = np.mean(x)
        std = np.std(x, ddof=1)

        # フィッティング実行
        params = distribution.fit(x)
        # フィッティング結果のパラメータを分割
        fit_params = {'arg': params[:-2],
                      'loc': params[-2],
                      'scale': params[-1]
                      }
        # フィッティング曲線の生成
        Xline = np.linspace(min(mean - std * sigmarange, np.amin(x)), max(mean + std * sigmarange, np.amax(x)), linesplit)
        Yline = distribution.pdf(Xline, loc=fit_params['loc'], scale=fit_params['scale'], *fit_params['arg'])

        # フィッティングの残差平方和を計算 (参考https://rmizutaa.hatenablog.com/entry/2020/02/24/191312)
        hist_y, hist_x = np.histogram(x, bins=20, density=True)  # ヒストグラム化して標準化
        hist_x = (hist_x + np.roll(hist_x, -1))[:-1] / 2.0  # ヒストグラムのxの値をビンの左端→中央に移動
        pred = distribution.pdf(hist_x, loc=fit_params['loc'], scale=fit_params['scale'], *fit_params['arg'])
        sum_squared_error = np.sum(np.power(pred - hist_y, 2.0))
        # AIC、BICを計算
        k = len(params)  # パラメータ数
        n = len(x)  # サンプル数
        # TODO: Fitterライブラリだと対数尤度はhist_xから求めているが、本来の定義ではxから求めるのが適切に見える
        logL = np.sum(distribution.logpdf(x, loc=fit_params['loc'], scale=fit_params['scale'], *fit_params['arg']))  # 対数尤度
        aic = -2 * logL + 2 * k  # AIC
        bic = -2 * logL + k * np.log(n)  # BIC
        # 評価指標()
        fit_scores = {'sum_squared_error': sum_squared_error,
                      'AIC': aic,
                      'BIC': bic
                      }

        return Xline, Yline, fit_params, fit_scores
    
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

    @classmethod
    def hist_dist(cls, data: pd.Series, dist='norm', ax=None, hue_data=None, bin_width=None, bins=None, norm_hist=True,
                  sigmarange=4, linecolor='red', linesplit=50, hist_kws={}):
        """
        分布フィッティングと各指標の表示

        Parameters
        ----------
        data : pd.Series
            フィッティング対象のデータ
        dist : str or List[str]
            分布の種類 ("norm", "lognorm", "gamma", "t", "expon", "uniform", "chi2", "weibull")
        ax : matplotlib.axes._subplots.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        hue_data : pd.Series
            積み上げ色分け指定対象のデータ (Noneなら色分けなし)
        bin_width : float
            ビンの幅 (NoneならFreedman-Diaconis ruleで自動決定)
        bins : int
            ビンの数 (bin_widthと共存不可)
        norm_hist : bool
            ヒストグラムを面積1となるよう正規化するか？
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        linesplit : str or List[str]
            フィッティング線の色指定 (複数分布フィッティング時は、List指定)
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        hist_kws : Dict
            ヒストグラム表示(matplotlib.axes.Axes.hist())の引数
        """

        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax == None:
            ax=plt.gca()

        # ビンサイズを設定
        if bin_width is not None:
            if bins is None:
                bins = np.arange(np.floor(data.min()), np.ceil(data.max()), bin_width)
            else: # binsとbin_widthは同時指定できない
                raise Exception('arguments "bins" and "bin_width" cannot coexist')
        # ヒストグラム描画
        if hue_data is None:  # 色分けないとき
            sns.distplot(data, ax=ax, kde=False, bins=bins, norm_hist=norm_hist, hist_kws=hist_kws)
        else:  # 色分けあるとき
            df_hue = pd.concat([data, hue_data], axis=1)
            grby_hue = df_hue.groupby(hue_data.name)
            data_list = [df_group[data.name] for key, df_group in grby_hue]
            hue_list = [key for key, df_group in grby_hue]
            ax.hist(data_list, stacked=True, bins=bins, density=norm_hist, **hist_kws)
            ax.legend(hue_list, loc='upper left')

        # distをList化
        dists = [dist] if isinstance(dist, str) else dist
        # フィッティング線の色指定をリスト化
        linecolor = [linecolor] if isinstance(linecolor, str) else linecolor
        # 2種類以上をプロットしており、かつ色指定がListでないとき、他の色を追加
        if len(dists) >= 2:
            if len(linecolor) == 1:
                linecolor = cls.DEFAULT_LINECOLORS
            elif len(dists) != len(linecolor):
                raise Exception('length of "linecolor" must be equal to length of "dist"')

        # 分布をフィッティング
        all_params = {}
        all_scores = {}
        line_legends = []
        for i, distribution in enumerate(dists):
            # 分布が文字列型のとき、scipy.stats.distributionsに変更
            if isinstance(distribution, str):
                if distribution == 'norm':
                    distribution = stats.norm
                elif distribution == 'lognorm':
                    distribution = stats.lognorm
                elif distribution == 'gamma':
                    distribution = stats.gamma
                elif distribution == 't':
                    distribution = stats.t
                elif distribution == 'expon':
                    distribution = stats.expon
                elif distribution == 'uniform':
                    distribution = stats.uniform
                elif distribution == 'chi2':
                    distribution = stats.chi2
                elif distribution == 'weibull':
                    distribution = stats.weibull_min
            # 分布フィット
            x = data.values
            xline, yline, fit_params, fit_scores = cls._fit_distribution(x, distribution, sigmarange, linesplit)

            # 標準化していないとき、ヒストグラムと最大値の8割を合わせるようフィッティング線の倍率調整
            if norm_hist is False:
                line_max = np.amax(yline)
                hist_max = ax.get_ylim()[1]
                yline = yline * hist_max / line_max * 0.8
            # フィッティング線の描画
            leg, = ax.plot(xline, yline, color=linecolor[i])
            line_legends.append(leg)

            # フィッティング結果パラメータをdict化
            params = {}
            # 正規分布
            if distribution == stats.norm:
                params['mean'] = fit_params['loc']
                params['std'] = fit_params['scale']
                all_params['norm'] = params
                all_scores['norm'] = fit_scores  # フィッティングの評価指標
            # 対数正規分布 (参考https://analytics-note.xyz/statistics/scipy-lognorm/)
            elif distribution == stats.lognorm:
                params['mu'] = np.log(fit_params['scale'])
                params['sigma'] = fit_params['arg'][0]
                params['offset'] = fit_params['loc']
                all_params['lognorm'] = params
                all_scores['lognorm'] = fit_scores  # フィッティングの評価指標
            # ガンマ分布 (参考https://qiita.com/kidaufo/items/2a5ba5a4bf100dc0f106)
            elif distribution == stats.gamma:
                params['theta'] = fit_params['scale']
                params['k'] = fit_params['arg'][0]
                params['offset'] = fit_params['loc']
                all_params['gamma'] = params
                all_scores['gamma'] = fit_scores  # フィッティングの評価指標
            elif dist == 't':
                distribution = stats.t
            elif dist == 'expon':
                distribution = stats.expon
            elif dist == 'uniform':
                distribution = stats.uniform
            elif dist == 'chi2':
                distribution = stats.chi2
            elif dist == 'weibull':
                distribution = stats.weibull_min
            
        # フィッティング線の凡例をプロット (2色以上のときのみ)
        if len(dists) >= 2:
            line_labels = [str(d) for d in dists]
            ax.legend(line_legends, line_labels, loc='upper right')

        print(all_scores)
        return all_params, all_scores

    @classmethod
    def plot_normality(cls, data: pd.Series, hue_data=None, bin_width=None, bins=None, norm_hist=True,
                        sigmarange=4, linecolor='red', linesplit=50, rounddigit=None,
                        hist_kws={}, subplot_kws={}):
        """
        正規性検定プロット

        Parameters
        ----------
        data : pd.Series
            フィッティング対象のデータ
        hue_data : pd.Series
            積み上げ色分け指定対象のデータ (Noneなら色分けなし)
        bin_width : float
            ビンの幅 (NoneならFreedman-Diaconis ruleで自動決定)
        bins : int
            ビンの数 (bin_widthと共存不可)
        norm_hist : bool
            ヒストグラムを面積1となるよう正規化するか？
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        hist_kws : Dict
            ヒストグラム表示(matplotlib.axes.Axes.hist())の引数
        subplot_kws : Dict
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        """

        # 描画用のsubplots作成
        if 'figsize' not in subplot_kws.keys():
            subplot_kws['figsize'] = (6, 12)
        fig, axes = plt.subplots(2, 1, **subplot_kws)

        # QQプロット描画
        stats.probplot(data, dist='norm', plot=axes[0])

        # ヒストグラムとフィッティング線を描画
        cls.hist_dist(data, dist='norm', ax=axes[1], hue_data=hue_data, bin_width=bin_width, bins=bins, norm_hist=norm_hist,
                      sigmarange=sigmarange, linecolor=linecolor, linesplit=linesplit, rounddigit=rounddigit, hist_kws=hist_kws)
        # 平均と不偏標準偏差を計算し、ヒストグラム図中に記載
        x = data.values
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        params = {'mean':mean,
                  'std':std
                  }
        param_list = [f'{k}={v}' for k, v in cls._round_dict_digits(params, rounddigit, 'sig').items()]
        param_text = "\n".join(param_list)
        axes[1].text(axes[1].get_xlim()[0] + (axes[1].get_xlim()[1] - axes[1].get_xlim()[0]) * 0.95,
                     axes[1].get_ylim()[1] * 0.9,
                     param_text, verticalalignment='top', horizontalalignment='right')

        # 正規性検定
        if len(x) <= 2000: # シャピロウィルク検定 (N<=2000のとき)
            method = 'shapiro-wilk'
            normality=stats.shapiro(x)
        else: # コルモゴロフ-スミルノフ検定 (N>2000のとき)
            method = 'kolmogorov-smirnov'
            normality = stats.kstest(x, stats.norm(loc=mean, scale=std).cdf)
        # 検定結果を図中に記載
        params = {'statistic':normality.statistic,
                  'pvalue':normality.pvalue,
                  }
        param_list = [f'{k}={v}' for k, v in cls._round_dict_digits(params, rounddigit, 'sig').items()]
        param_list.insert(0, f'method={method}')
        param_text = "\n".join(param_list)
        axes[0].text(axes[0].get_xlim()[0] + (axes[0].get_xlim()[1] - axes[0].get_xlim()[0]) * 0.95,
                     axes[0].get_ylim()[0] + (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) * 0.1,
                     param_text, verticalalignment='bottom', horizontalalignment='right')
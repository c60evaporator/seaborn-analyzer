from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score

import decimal

class dist():
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
    def _plot_real_pred(cls, y_real, y_pred, hue_data=None, ax=None, linecolor='red', linesplit=50, rounddigit=None,
                        score_dict=None, colname=None):
        """
        実測値と予測値を、回帰評価指標とともにプロット

        Parameters
        ----------
        y_real : ndarray
            目的変数実測値
        y_pred : ndarray
            目的変数予測値
        hue_data : ndarray
            色分け用ラベルデータ
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        linecolor : str
            予測値=実測値の線の色
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        score_dict : Dict[str, float]
            算出した評価指標一覧
        colname : str
            列名
        """
        # 実測値と予測値を合体
        data = np.stack([y_real, y_pred], axis=1)
        columns = ['y_real', 'y_pred']
        # 色分け指定しているとき
        if hue_data is not None:
            data = np.hstack([data, hue_data.reshape(len(hue_data), 1)])
            columns.append('hue')
        # DataFrame化
        data = pd.DataFrame(data, columns=columns)
        # 散布図プロット
        sns.scatterplot(x='y_real', y='y_pred', data=data, ax=ax, hue='hue')

        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax == None:
            ax=plt
        # score_dictがNoneのとき、空のDictを加瀬宇
        if score_dict is None:
            score_dict = {}

        # 予測値=実測値の線を作成
        real_min = np.amin(y_real)
        real_max = np.amax(y_real)
        real_line = np.linspace(real_min, real_max, linesplit)
        # 評価指標文字列作成
        score_list = [f'{k}={v}' for k, v in cls._round_dict_digits(score_dict, rounddigit, 'sig').items()]
        score_text = "\n".join(score_list)
        # 線と文字をプロット
        ax.plot(real_line, real_line, color=linecolor)
        ax.text(real_max, np.amin(y_pred), score_text, verticalalignment='bottom', horizontalalignment='right')
    
    @classmethod
    def regression_plot_pred(cls, model, x: List[str], y: str, data: pd.DataFrame, ax=None, hue=None, linecolor='red', linesplit=50, rounddigit=None,
                             scores='rmse', cv=None, cv_seed: int=42, model_params=None, fit_params=None):
        """
        回帰して予測値と実測値をプロットし、評価値を表示

        Parameters
        ----------
        model : 
            回帰の学習器
        x : str or List[str]
            説明変数カラム (列名指定 or 列名のリスト指定)
        y : str
            目的変数カラム (列名指定)
        data : pd.DataFrame
            フィッティング対象のデータ
        ax : matplotlib.axes._subplots.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        hue : str
            色分け指定カラム (列名指定)
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        scores : str or list[str]
            算出する評価指標（'r2', 'mae','rmse', 'rmsle', 'maxerror'）
        cv : None or int or KFold
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed : int
            クロスバリデーションの乱数シード
        params : Dict[str, float]
            学習器に使用するパラメータの値 (Noneならデフォルト)
        fit_params : Dict[str, float]
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        elif isinstance(x, str):
            X = data[[x]].values
        else:
            Exception('x must be str or list[str]')
        # yをndarray化
        if isinstance(y, str):
            y_real = data[y].values
        else:
            Exception('y msut be str')

        # scoresの型をListに統一
        score_dict = {}
        if scores is None:
            scores = []
        elif isinstance(scores, str):
            scores = [scores]
        elif ~isinstance(scores, list):
            Exception('scores must be str or list[str]')
        
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            model.fit(X, y_real, **fit_params)
            y_pred = model.predict(X)
            # 評価指標算出
            for score in scores:
                if score == 'r2':
                    score_dict['r2'] = r2_score(y_real, y_pred)
                elif score == 'mae':
                    score_dict['mae'] = mean_absolute_error(y_real, y_pred)
                elif score == 'rmse':
                    score_dict['rmse'] = mean_squared_error(y_real, y_pred)
                elif score == 'rmsle':
                    score_dict['rmsle'] = mean_squared_log_error(y_real, y_pred)
                elif score == 'maxerror':
                    score_dict['maxerror'] = max([abs(p - r) for r, p in zip(y_real, y_pred)])
            # 色分け用データ取得
            if hue is None:
                hue_data = None
            else:
                hue_data = data[hue].values
            # プロット
            cls._plot_real_pred(y_real, y_pred, hue_data, score_dict=score_dict, colname=None)
            
        # クロスバリデーション実施時
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
        return score_dict

from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

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
    def plot_real_pred(cls, y_real, y_pred, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=50, rounddigit=None,
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
        hue_name : str
            色分け用の列名
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
        # 実測値と予測値を合体してDataFrame化
        data = np.stack([y_real, y_pred], axis=1)
        data = pd.DataFrame(data, columns=['y_real', 'y_pred'])
        # 色分け指定しているとき、色分け用のフィールドを追加
        if hue_data is not None:
            if hue_name == None:
                hue_name = 'hue'
            data[hue_name] = pd.Series(hue_data)
        # 散布図プロット
        sns.scatterplot(x='y_real', y='y_pred', data=data, ax=ax, hue=hue_name)

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
    def _rank_display(cls, y_real, y_pred, rank_number, rank_field, rank_field_data, ax=None, rounddigit=None):
        """
        誤差上位をプロット

        Parameters
        ----------
        y_real : ndarray
            目的変数実測値
        y_pred : ndarray
            目的変数予測値
        rank_number : int
            誤差上位何番目までを文字表示するか
        rank_field : List[str]
            誤差上位と一緒に表示するフィールド (NoneならIndexを使用)
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax == None:
            ax=plt

        if rank_field == None:
            rank_field = 'index'
        y_error = np.abs(y_pred - y_real)
        rank_index  = np.argsort(-y_error)[:rank_number]
        for rank, i in enumerate(rank_index):
            error = cls._round_digits(y_error[i], rounddigit=rounddigit, method='decimal')
            rank_text = f'rank={rank+1}\n{rank_field}={rank_field_data[i]}\nerror={error}'
            ax.text(y_real[i], y_pred[i], rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def regression_plot_pred(cls, model, x: List[str], y: str, data: pd.DataFrame, hue=None, linecolor='red', rounddigit=None,
                             rank_number=None, rank_field=None, scores=['rmse'], plot_stats='mean', cv=None, cv_seed=42,
                             model_params=None, fit_params=None, subplot_kws={}):
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
        hue : str
            色分け指定カラム (列名指定)
        sigmarange : float
            フィッティング線の表示範囲 (標準偏差の何倍まで表示するか指定)
        rounddigit: int
            表示指標の小数丸め桁数
        rank_number : int
            誤差上位何番目までを文字表示するか
        rank_field : List[str]
            誤差上位と一緒に表示するフィールド (NoneならIndexを使用)
        scores : str or list[str]
            算出する評価指標（'r2', 'mae','rmse', 'rmsle', 'max_error'）
        plot_stats : Dict
            クロスバリデーション時に表示する統計値 ('mean', 'median', 'max', 'min')
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
        subplot_kws : Dict[str, float]
            プロット用のplt.subplots()に渡す引数 (例：figsize)
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
        elif not isinstance(scores, list):
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
            for scoring in scores:
                if scoring == 'r2':
                    score_dict['r2'] = r2_score(y_real, y_pred)
                elif scoring == 'mae':
                    score_dict['mae'] = mean_absolute_error(y_real, y_pred)
                elif scoring == 'rmse':
                    score_dict['rmse'] = mean_squared_error(y_real, y_pred)
                elif scoring == 'rmsle':
                    score_dict['rmsle'] = mean_squared_log_error(y_real, y_pred)
                elif scoring == 'max_error':
                    score_dict['max_error'] = max([abs(p - r) for r, p in zip(y_real, y_pred)])
            # 色分け用データ取得
            if hue is None:
                hue_data = None
                hue_name = None
            else:
                hue_data = data[hue].values
                hue_name = hue
            # 誤差上位表示用データ取得
            if rank_number is not None:
                if rank_field is None:  # 表示フィールド指定ないとき、Index使用
                    rank_field_data = data.index.values
                else:  # 表示フィールド指定あるとき
                    rank_field_data = data[rank_field].values
            # 予測値と実測値プロット
            cls.plot_real_pred(y_real, y_pred, hue_data=hue_data, hue_name=hue_name,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict, colname=None)
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_real, y_pred, rank_number, rank_field, rank_field_data, rounddigit=rounddigit)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)

            # スコア種類ごとにクロスバリデーションスコアの算出
            score_all_dict = {}
            for scoring in scores:
                # cross_val_scoreでクロスバリデーション
                if scoring == 'r2':
                    score_all_dict['r2'] = cross_val_score(model, X, y_real, cv=cv, scoring='r2',
                                                           fit_params=fit_params, n_jobs=-1)
                elif scoring == 'mae':
                    neg_mae = cross_val_score(model, X, y_real, cv=cv, scoring='neg_mean_absolute_error',
                                                           fit_params=fit_params, n_jobs=-1)
                    score_all_dict['mae'] = -neg_mae  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmse':
                    neg_rmse = cross_val_score(model, X, y_real, cv=cv, scoring='neg_root_mean_squared_error',
                                                           fit_params=fit_params, n_jobs=-1)
                    score_all_dict['rmse'] = -neg_rmse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmsle':
                    neg_msle = cross_val_score(model, X, y_real, cv=cv, scoring='neg_mean_squared_log_error',
                                                           fit_params=fit_params, n_jobs=-1)
                    score_all_dict['rmsle'] = np.sqrt(-neg_msle)  # 正負を逆にしてルートをとる
                elif scoring == 'max_error':
                    neg_max_error = cross_val_score(model, X, y_real, cv=cv, scoring='max_error',
                                                           fit_params=fit_params, n_jobs=-1)
                    score_all_dict['max_error'] = - neg_max_error  # scikit-learnの仕様に合わせ正負を逆に
            
            #LeaveOneOutかどうかを判定
            isLeaveOneOut = isinstance(cv, LeaveOneOut)
            cv_num = 1 if isLeaveOneOut else cv.n_splits
            # 表示用のaxes作成
            # LeaveOneOutのとき、クロスバリデーションごとの図は作成せず
            if isLeaveOneOut:
                if 'figsize' not in subplot_kws.keys():
                    subplot_kws['figsize'] = (6, 6)
                fig, axes = plt.subplots(1, 1, **subplot_kws)
            # LeaveOneOut以外のとき、クロスバリデーションごとに図作成
            else:
                if 'figsize' not in subplot_kws.keys():
                    subplot_kws['figsize'] = (6, (cv_num + 1) * 6)
                fig, axes = plt.subplots(cv_num + 1, 1, **subplot_kws)

            # 表示用にテストデータと学習データ分割
            y_real_all = []
            y_pred_all = []
            hue_all = []
            rank_field_all = []
            for i, (train, test) in enumerate(cv.split(X, y_real)):
                X_train = X[train]
                y_train = y_real[train]
                X_test = X[test]
                y_test = y_real[test]
                # 色分け用データ取得(していないときは、クロスバリデーション番号を使用、LeaveOuneOutのときは番号分けない)
                if hue is None:
                    hue_test = np.full(1 ,'leave_one_out') if isLeaveOneOut else np.full(len(test) ,f'cv_{i}')
                    hue_name = 'cv_number'  # 色分け名を'cv_number'に指定
                else:
                    hue_test = data[hue].values[test]
                    hue_name = hue
                # 誤差上位表示用データ取得
                if rank_number is not None:
                    if rank_field is None:  # 表示フィールド指定ないとき、Index使用
                        rank_field_test = data.index.values[test]
                    else:  # 表示フィールド指定あるとき
                        rank_field_test = data[rank_field].values[test]
                else:
                    rank_field_test = np.array([])
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # CV内結果をプロット(LeaveOneOutのときはプロットしない)
                if not isLeaveOneOut:
                    score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                    cls.plot_real_pred(y_test, y_pred, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                                    linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict, colname=None)
                    axes[i].set_title(f'Cross Validation No{i}')
                # 全体プロット用データに追加
                y_real_all.append(y_test)
                y_pred_all.append(y_pred)
                hue_all.append(hue_test)
                rank_field_all.append(rank_field_test)

            # 全体プロット用データを合体
            y_real_all = np.hstack(y_real_all)
            y_pred_all = np.hstack(y_pred_all)
            hue_all = np.hstack(hue_all)
            rank_field_all = np.hstack(rank_field_all)
            # 指標の統計値を計算
            if plot_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}            
            elif plot_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}            
            elif plot_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}            
            elif plot_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
            # 全体プロット
            ax_all = axes if isLeaveOneOut else axes[cv_num]
            cls.plot_real_pred(y_real_all, y_pred_all, hue_data=hue_all, hue_name=hue_name, ax=ax_all,
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict, colname=None)
            ax_all.set_title('All Cross Validations')
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_real_all, y_pred_all, rank_number, rank_field, rank_field_all,
                                  ax=ax_all, rounddigit=rounddigit)

        return score_dict

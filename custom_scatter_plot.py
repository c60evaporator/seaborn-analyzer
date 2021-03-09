from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression

import decimal

class regplot():
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

    def _make_score_dict(y_real, y_pred, scores):
        """
        評価指標を算出してdict化
        """
        score_dict = {}
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
        return score_dict

    @classmethod
    def _rank_display(cls, y_real, y_pred, rank_number, rank_col, rank_col_data, ax=None, rounddigit=None):
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
        rank_col : List[str]
            誤差上位と一緒に表示するフィールド名 (NoneならIndexを使用)
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax == None:
            ax=plt

        if rank_col == None:
            rank_col = 'index'
        y_error = np.abs(y_pred - y_real)
        rank_index  = np.argsort(-y_error)[:rank_number]
        for rank, i in enumerate(rank_index):
            error = cls._round_digits(y_error[i], rounddigit=rounddigit, method='decimal')
            rank_text = f'{rank+1}\n{rank_col}={rank_col_data[i]}\nerror={error}'
            ax.text(y_real[i], y_pred[i], rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def _scatterplot_ndarray(cls, x, x_name, y, y_name, hue_data, hue_name, ax):
        """
        ndarrayからscatterplot
        """
        # X値とY値を合体してDataFrame化
        data = np.stack([x, y], axis=1)
        data = pd.DataFrame(data, columns=[x_name, y_name])
        # 色分け指定しているとき、色分け用のフィールドを追加
        if hue_data is not None:
            if hue_name == None:
                hue_name = 'hue'
            data[hue_name] = pd.Series(hue_data)
        # 散布図プロット
        sns.scatterplot(x=x_name, y=y_name, data=data, ax=ax, hue=hue_name)

    @classmethod
    def plot_real_pred(cls, y_real, y_pred, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=50, rounddigit=None,
                        score_dict=None):
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
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        linecolor : str
            予測値=実測値の線の色
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        score_dict : Dict[str, float]
            算出した評価指標一覧
        """
        # 散布図プロット
        cls._scatterplot_ndarray(y_real, 'y_real', y_pred, 'y_pred', hue_data, hue_name, ax)

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
    def regression_plot_pred(cls, model, x: List[str], y: str, data: pd.DataFrame, hue=None, linecolor='red', rounddigit=None,
                             rank_number=None, rank_col=None, scores=['rmse'], plot_stats='mean', cv=None, cv_seed=42,
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
        linecolor : str
            予測値=実測値の線の色
        rounddigit: int
            表示指標の小数丸め桁数
        rank_number : int
            誤差上位何番目までを文字表示するか
        rank_col : List[str]
            誤差上位と一緒に表示するフィールド名 (NoneならIndexを使用)
        scores : str or list[str]
            算出する評価指標 ('r2', 'mae','rmse', 'rmsle', 'max_error')
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
            score_dict = cls._make_score_dict(y_real, y_pred, scores)
            # 色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 誤差上位表示用データ取得
            if rank_number is not None:
                if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                    rank_col_data = data.index.values
                else:  # 表示フィールド指定あるとき
                    rank_col_data = data[rank_col].values
            # 予測値と実測値プロット
            cls.plot_real_pred(y_real, y_pred, hue_data=hue_data, hue_name=hue_name,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict)
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_real, y_pred, rank_number, rank_col, rank_col_data, rounddigit=rounddigit)
            return score_dict
            
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

            # クロスバリデーション
            y_real_all = []
            y_pred_all = []
            hue_all = []
            rank_col_all = []
            for i, (train, test) in enumerate(cv.split(X, y_real)):
                # 表示用にテストデータと学習データ分割
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
                    if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                        rank_col_test = data.index.values[test]
                    else:  # 表示フィールド指定あるとき
                        rank_col_test = data[rank_col].values[test]
                else:
                    rank_col_test = np.array([])
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # CV内結果をプロット(LeaveOneOutのときはプロットしない)
                if not isLeaveOneOut:
                    score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                    cls.plot_real_pred(y_test, y_pred, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                                    linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict)
                    axes[i].set_title(f'Cross Validation No{i}')
                # 全体プロット用データに追加
                y_real_all.append(y_test)
                y_pred_all.append(y_pred)
                hue_all.append(hue_test)
                rank_col_all.append(rank_col_test)

            # 全体プロット用データを合体
            y_real_all = np.hstack(y_real_all)
            y_pred_all = np.hstack(y_pred_all)
            hue_all = np.hstack(hue_all)
            rank_col_all = np.hstack(rank_col_all)
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
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict)
            ax_all.set_title('All Cross Validations')
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_real_all, y_pred_all, rank_number, rank_col, rank_col_all,
                                  ax=ax_all, rounddigit=rounddigit)
            return score_stats_dict

    @classmethod
    def linear_plot(cls, x: str, y: str, data: pd.DataFrame, ax=None, hue=None, linecolor='red', linesplit=50, rounddigit=None):
        """
        1変数線形回帰してプロットし、p値と相関係数を表示

        Parameters
        ----------
        x : str
            説明変数カラム (列名指定)
        y : str
            目的変数カラム (列名指定)
        data : pd.DataFrame
            フィッティング対象のデータ
        ax : matplotlib.axes._subplots.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        hue : str
            色分け指定カラム (列名指定)
        linecolor : str
            予測値=実測値の線の色
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # xをndarray化
        if isinstance(x, str):
            X = data[[x]].values
        else:
            Exception('x must be str')
        # yをndarray化
        if isinstance(y, str):
            y_real = data[y].values
        else:
            Exception('y msut be str')

        # まずは散布図プロット
        ax = sns.scatterplot(x=x, y=y, data=data, ax=ax, hue=hue)

        # 線形回帰モデル作成
        lr = LinearRegression()
        lr.fit(X, y_real)
        xmin = np.amin(X)
        xmax = np.amax(X)
        Xline = np.linspace(xmin, xmax, linesplit)
        Xline = Xline.reshape(len(Xline), 1)
        # 回帰線を描画
        ax.plot(Xline, lr.predict(Xline), color=linecolor)

        # ピアソンの相関係数およびp値を表示
        pearsonr = stats.pearsonr(data[x], data[y])
        r = cls._round_digits(pearsonr[0], rounddigit=rounddigit, method="decimal")
        pvalue = cls._round_digits(pearsonr[1], rounddigit=rounddigit, method="decimal")
        rtext = f'r={r}\np={pvalue}'
        ax.text(xmax, np.amin(y_real), rtext, verticalalignment='bottom', horizontalalignment='right')

    @classmethod
    def _model_plot_1d(cls, trained_model, X, y_real, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=50, rounddigit=None,
                        score_dict=None):
        """
        1次説明変数回帰曲線を、回帰評価指標とともにプロット

        Parameters
        ----------
        trained_model : 
            学習済の回帰モデル(scikit-learn API)
        X : ndarray
            説明変数
        y_real : ndarray
            目的変数実測値
        hue_data : ndarray
            色分け用ラベルデータ
        hue_name : str
            色分け用の列名
        ax : matplotlib.axes._subplots.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)
        linecolor : str
            予測値=実測値の線の色
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        score_dict : Dict[str, float]
            算出した評価指標一覧
        """
        # 散布図プロット
        cls._scatterplot_ndarray(np.ravel(X), 'X', y_real, 'Y', hue_data, hue_name, ax)

        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax == None:
            ax=plt
        # score_dictがNoneのとき、空のDictを入力
        if score_dict is None:
            score_dict = {}

        # 回帰モデルの線を作成
        xmin = np.amin(X)
        xmax = np.amax(X)
        Xline = np.linspace(xmin, xmax, linesplit)
        Xline = Xline.reshape(len(Xline), 1)
        # 回帰線を描画
        ax.plot(Xline, trained_model.predict(Xline), color=linecolor)
        
        # 評価指標文字列作成
        score_list = [f'{k}={v}' for k, v in cls._round_dict_digits(score_dict, rounddigit, 'sig').items()]
        score_text = "\n".join(score_list)
        ax.text(xmax, np.amin(y_real), score_text, verticalalignment='bottom', horizontalalignment='right')

    @classmethod
    def regression_plot_1d(cls, model, x: str, y: str, data: pd.DataFrame, hue=None, linecolor='red', rounddigit=None,
                             scores=['rmse'], plot_stats='mean', cv=None, cv_seed=42,
                             model_params=None, fit_params=None, subplot_kws={}):
        """
        1次元説明変数の任意の回帰曲線をプロット

        Parameters
        ----------
        model : 
            使用する回帰モデル(scikit-learn API)
        x : str
            説明変数カラム (列名指定)
        y : str
            目的変数カラム (列名指定)
        data : pd.DataFrame
            フィッティング対象のデータ
        hue : str
            色分け指定カラム (列名指定)
        linecolor : str
            予測値=実測値の線の色
        rounddigit: int
            表示指標の小数丸め桁数
        scores : str or list[str]
            算出する評価指標 ('r2', 'mae','rmse', 'rmsle', 'max_error')
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
        if isinstance(x, str):
            X = data[[x]].values
        else:
            Exception('x must be str')
        # yをndarray化
        if isinstance(y, str):
            y_real = data[y].values
        else:
            Exception('y msut be str')

        # scoresの型をListに統一
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
            score_dict = cls._make_score_dict(y_real, y_pred, scores)
            # 色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 回帰線プロット
            cls._model_plot_1d(model, X, y_real, hue_data=hue_data, hue_name=hue_name,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict)
            return score_dict
            
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

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_real)):
                # 表示用にテストデータと学習データ分割
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
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                # CV内結果をプロット(LeaveOneOutのときはプロットしない)
                if not isLeaveOneOut:
                    score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                    cls._model_plot_1d(model, X_test, y_test, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict)
                    axes[i].set_title(f'Cross Validation No{i}')

            # 指標の統計値を計算
            if plot_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}
            elif plot_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}
            elif plot_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}
            elif plot_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
            # 全体データで学習＆評価指標算出
            model.fit(X, y_real, **fit_params)
            y_pred = model.predict(X)
            score_dict = cls._make_score_dict(y_real, y_pred, scores)
            # 全体データ指標を指標dictに追加
            score_dict = {f'{k}_train': np.mean(v) for k, v in score_dict.items()}
            score_stats_dict.update(score_dict)
            # 全体色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 全体プロット
            ax_all = axes if isLeaveOneOut else axes[cv_num]
            cls._model_plot_1d(model, X, y_real, hue_data=hue_data, hue_name=hue_name, ax=ax_all,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict)
            ax_all.set_title('All Cross Validations')
            return score_stats_dict

    @classmethod
    def _reg_heat_plot_2d(cls, trained_model, x_heat, y_real_col, y_pred_col, rank_col, data, x_heat_indices,
                          x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                          vmin, vmax, ax, plot_scatter, maxerror, rank_number,
                          rounddigit_rank, rounddigit_x1, rounddigit_x2,
                          heat_kws={}, scatter_kws={}):
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax == None:
            ax=plt.gca()

        # ヒートマップ用グリッドデータを作成
        xx = np.linspace(x1_start, x1_end, heat_division)
        yy = np.linspace(x2_start, x2_end, heat_division)
        X1, X2 = np.meshgrid(xx, yy)
        X_grid = np.c_[X1.ravel(), X2.ravel()]
        df_heat = pd.DataFrame(X_grid, columns=x_heat)
        # 推論用に全特徴量を保持したndarrayを作成 (ヒートマップ非使用特徴量は固定値other_xとして追加)
        n_rows = X_grid.shape[0]
        X_all = []
        other_add_flg = False
        for i in range(2 + len(other_x)):
            if i == x_heat_indices[0]: # ヒートマップ使用特徴量(1個目)を追加
                X_all.append(X_grid[:, 0].reshape(n_rows, 1))
            elif i == x_heat_indices[1]: # ヒートマップ使用特徴量(2個目)を追加
                X_all.append(X_grid[:, 1].reshape(n_rows, 1))
            elif len(other_x) >= 1 and not other_add_flg:  # ヒートマップ非使用特徴量(1個目)を固定値として追加
                X_all.append(np.full((n_rows, 1), other_x[0]))
                other_add_flg = True
            elif len(other_x) == 2:  # ヒートマップ非使用特徴量(2個目)を固定値として追加
                X_all.append(np.full((n_rows, 1), other_x[1]))
        X_all = np.hstack(X_all)
        # グリッドデータに対して学習し、推定値を作成
        y_pred_grid = trained_model.predict(X_all)
        df_heat['y_pred'] = pd.Series(y_pred_grid)
        # グリッドデータ縦軸横軸の表示桁数を調整
        df_heat[x_heat[0]] = df_heat[x_heat[0]].map(lambda x: cls._round_digits(x, rounddigit=rounddigit_x1))
        df_heat[x_heat[1]] = df_heat[x_heat[1]].map(lambda x: cls._round_digits(x, rounddigit=rounddigit_x2))
        # グリッドデータをピボット化
        df_heat_pivot = pd.pivot_table(data=df_heat, values='y_pred', 
                                  columns=x_heat[0], index=x_heat[1], aggfunc=np.mean)

        # ヒートマップのカラーマップ指定ないとき、YlGnを指定
        if 'cmap' not in heat_kws.keys():
            heat_kws['cmap'] = 'YlGn'
        # ヒートマップをプロット
        sns.heatmap(df_heat_pivot, ax=ax, vmax=vmax, vmin=vmin, center=(vmax+vmin)/2, **heat_kws)

        # 誤差散布図をプロット
        if plot_scatter:
            y_error = data[y_pred_col].values - data[y_real_col].values        
            # 軸範囲が0～heat_divisionになっているので、スケール変換
            x1_scatter = 0.5 + (data[x_heat[0]].values - x1_start) * (heat_division - 1) / (x1_end - x1_start)
            x2_scatter = 0.5 + (data[x_heat[1]].values - x2_start) * (heat_division - 1) / (x2_end - x2_start)
            # 散布図のカラーマップ指定ないとき、seismicを指定
            if 'cmap' not in scatter_kws.keys():
                scatter_kws['cmap'] = 'seismic'
            # 散布図プロット
            ax.scatter(x1_scatter, x2_scatter, vmin=-maxerror, vmax=maxerror, c=y_error, **scatter_kws)
    
    @classmethod
    def _reg_heat_plot(cls, trained_model, X, y_pred, y_real, x_heat, x_not_heat, x_heat_indices,
                       pair_sigmarange=2.0, pair_sigmainterval=0.5, heat_extendsigma=0.5, heat_division=30, 
                       vmin=None, vmax=None, plot_scatter=True, maxerror=None,
                       rank_number=None, rank_col=None, rank_col_data=None,
                       rounddigit_rank=None, rounddigit_x1=None, rounddigit_x2=None, rounddigit_x3=None,
                       subplot_kws={}, heat_kws={}, scatter_kws={}):
        # 説明変数の数
        x_num = X.shape[1]
        # ヒートマップ使用DataFrame
        df_heat = pd.DataFrame(X[:, x_heat_indices], columns=x_heat)
        # ヒートマップ非使用DataFrame
        X_not_heat = X[:, [i for i in range(X.shape[1]) if i not in x_heat_indices]]
        df_not_heat = pd.DataFrame(X_not_heat, columns=x_not_heat)
        # 結合＆目的変数実測値と予測値追加
        df_all = df_heat.join(df_not_heat)
        df_all = df_all.join(pd.DataFrame(y_real, columns=['y_real']))
        df_all = df_all.join(pd.DataFrame(y_pred, columns=['y_pred']))
        # ヒートップ非使用特徴量を標準化してDataFrameに追加
        if x_num >= 3:
            X_not_heat_norm = stats.zscore(X_not_heat)
            columns = [f'normalize_{c}' for c in x_not_heat]
            df_all = df_all.join(pd.DataFrame(X_not_heat_norm, columns=[f'normalize_{c}' for c in x_not_heat]))
        # 誤差上位表示用IDデータをDataFrameに追加
        df_all = df_all.join(pd.DataFrame(rank_col_data, columns=[rank_col]))

        # 誤差の順位を計算
        y_error = np.abs(y_pred - y_real)
        rank_index  = np.argsort(-y_error)[:rank_number]

        # ヒートマップのX1軸およびX2軸の表示範囲(最大最小値 + extendsigma)
        x1_min = np.min(X[:, x_heat_indices[0]])
        x1_max = np.max(X[:, x_heat_indices[0]])
        x1_std = np.std(X[:, x_heat_indices[0]])
        x1_start = x1_min - x1_std * heat_extendsigma
        x1_end = x1_max + x1_std * heat_extendsigma
        x2_min = np.min(X[:, x_heat_indices[1]])
        x2_max = np.max(X[:, x_heat_indices[1]])
        x2_std = np.std(X[:, x_heat_indices[1]])
        x2_start = x2_min - x2_std * heat_extendsigma
        x2_end = x2_max + x2_std * heat_extendsigma

        # プロットする図の数(sigmarange外「2枚」 + sigmarange内「int(pair_sigmarange / pair_sigmainterval) * 2枚」)
        pair_n = int(pair_sigmarange / pair_sigmainterval) * 2 + 2
        # ヒートップ非使用特徴量をプロットする範囲の下限(標準化後)
        pair_min = -(pair_n - 2) / 2 * pair_sigmainterval

        # 説明変数が2次元のとき (図は1枚のみ)
        if x_num == 2:
            pair_w = 1
            pair_h = 1
        # 説明変数が3次元のとき (図はpair_n × 1枚)
        elif x_num == 3:
            pair_w = 1
            pair_h = pair_n
        # 説明変数が4次元のとき (図はpair_n × pair_n枚)
        elif x_num == 4:
            pair_w = pair_n
            pair_h = pair_n

        # figsize (全ての図全体のサイズ)指定
        if 'figsize' not in subplot_kws.keys():
            subplot_kws['figsize'] = (pair_w * 6, pair_h * 5)
        # プロット用のaxes作成
        fig, axes = plt.subplots(pair_h, pair_w, **subplot_kws)

        # 図ごとにプロット
        for i in range(pair_h):
            for j in range(pair_w):
                # pair縦軸特徴量(標準化後)の最小値
                if i == 0:
                    h_min = -float('inf')
                    h_mean = pair_min - pair_sigmainterval / 2  # ヒートマップ外特徴量指定用の平均値
                else:
                    h_min = pair_min + (i - 1) * pair_sigmainterval
                    h_mean = pair_min + (i - 0.5) * pair_sigmainterval  # ヒートマップ外特徴量指定用の平均値
                # pair縦軸特徴量(標準化後)の最大値
                if i == pair_h - 1:
                    h_max = float('inf')
                else:
                    h_max = pair_min + i * pair_sigmainterval
                # pair横軸特徴量(標準化後)の最小値
                if j == 0:
                    w_min = -float('inf')
                    w_mean = pair_min - pair_sigmainterval / 2  # ヒートマップ外特徴量指定用の平均値
                else:
                    w_min = pair_min + (j - 1) * pair_sigmainterval
                    w_mean = pair_min + (j - 0.5) * pair_sigmainterval  # ヒートマップ外特徴量指定用の平均値
                # pair横軸特徴量(標準化後)の最大値
                if j == pair_w - 1:
                    w_max = float('inf')
                else:
                    w_max = pair_min + j * pair_sigmainterval

                # 説明変数が2次元のとき (図は1枚のみ)
                if x_num == 2:
                    ax = axes
                    df_pair = df_all.copy()
                    other_x = []
                # 説明変数が3次元のとき (図はpair_n × 1枚)
                elif x_num == 3:
                    ax = axes[i]
                    # 縦軸特徴量範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_heat[0]}'] >= h_min) & (df_all[f'normalize_{x_not_heat[0]}'] < h_max)]
                    # ヒートマップ外特徴量の標準化逆変換
                    x3_mean = np.mean(X_not_heat[:, 0])
                    x3_std = np.std(X_not_heat[:, 0])
                    other_x = [h_mean * x3_std + x3_mean]
                # 説明変数が4次元のとき (図はpair_n × pair_n枚)
                elif x_num == 4:
                    ax = axes[j, i]
                    # 縦軸特徴量範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_heat[0]}'] >= h_min) & (df_all[f'normalize_{x_not_heat[0]}'] < h_max)]
                    # 横軸特徴量範囲内のみのデータを抽出
                    df_pair = df_pair[(df_pair[f'normalize_{x_not_heat[1]}'] >= w_min) & (df_pair[f'normalize_{x_not_heat[1]}'] < w_max)]
                    # ヒートマップ外特徴量の標準化逆変換
                    x3_mean = np.mean(X_not_heat[:, 0])
                    x3_std = np.std(X_not_heat[:, 0])
                    x4_mean = np.mean(X_not_heat[:, 1])
                    x4_std = np.std(X_not_heat[:, 1])
                    other_x = [h_mean * x3_std + x3_mean, w_mean * x4_std + x4_mean]
                
                cls._reg_heat_plot_2d(trained_model, x_heat, 'y_real', 'y_pred', rank_col, df_pair, x_heat_indices,
                                      x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                                      vmin, vmax, ax, plot_scatter, maxerror, rank_number,
                                      rounddigit_rank, rounddigit_x1, rounddigit_x2,
                                      heat_kws=heat_kws, scatter_kws=scatter_kws)

                # グラフタイトルとして、ヒートマップ外特徴量の範囲を記載
                if x_num == 3:
                    if i == 0:
                        ax.set_title(f'{x_not_heat[0]}=- {cls._round_digits(h_max * x3_std + x3_mean, rounddigit=rounddigit_x3)} (- {h_max}σ)')
                    elif i == pair_h - 1:
                        ax.set_title(f'{x_not_heat[0]}={cls._round_digits(h_min * x3_std + x3_mean, rounddigit=rounddigit_x3)} - ({h_min}σ -)')
                    else:
                        ax.set_title(f'{x_not_heat[0]}={cls._round_digits(h_min * x3_std + x3_mean, rounddigit=rounddigit_x3)} - {cls._round_digits(h_max * x3_std + x3_mean, rounddigit=rounddigit_x3)} ({h_min}σ - {h_max}σ)')
                if x_num == 4:
                    ax.set_title(f'{x_not_heat[0]}= {h_min}σ - {h_max}σ  {x_not_heat[1]}= {w_min}σ - {w_max}σ')

        # 字が重なるのでtight_layoutにする
        plt.tight_layout()

    @classmethod
    def regression_heat_plot(cls, model, x: List[str], y: str, data: pd.DataFrame, x_heat: List[str] = None,
                             pair_sigmarange = 1.5, pair_sigmainterval = 0.5, heat_extendsigma = 0.5, value_extendsigma = 0.5, plot_scatter = True,
                             rank_number=None, rank_col=None, rounddigit_rank=3, rounddigit_x1=2, rounddigit_x2=2, rounddigit_x3=2,
                             cv=None, cv_seed=42, display_cv_indices = 0,
                             model_params=None, fit_params=None, subplot_kws={}, heat_kws={}, scatter_kws={}):
        # 説明変数xの次元が2～4以外ならエラーを出す
        if len(x) < 2 or len(x) > 4:
            Exception('length of x must be 2 to 4')
        
        # display_cv_indexをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x, list):
            Exception('cv_display_num must be int or List[int]')
        
        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        else:
            Exception('x must be str or str')
        # yをndarray化
        if isinstance(y, str):
            y_real = data[y].values
        else:
            Exception('y msut be str')
        
        # ヒートマップ表示用の列を抽出
        if x_heat == None:  # 列名指定していないとき、前から2列を抽出
            x_heat = x[:2]
            x_heat_indices = [0, 1]
        else:  # 列名指定しているとき、該当列のXにおけるインデックス(0～3)を保持
            if len(x_heat) != 2:
                Exception('length of x_heat must be 2')
            x_heat_indices = []
            for colname in x_heat:
                x_heat_indices.append(x.index(colname))
        # ヒートマップ表示以外の列
        x_not_heat = [colname for colname in x if colname not in x_heat]
        
        # ヒートマップの色分け最大最小値(y_realの最大最小値 ± y_realの標準偏差 × value_extendsigma)
        y_real_std = np.std(y_real)
        vmin = np.min(y_real) - y_real_std * value_extendsigma
        vmax = np.max(y_real) + y_real_std * value_extendsigma

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
            # 誤差上位表示用データ取得
            if rank_number is not None:
                if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                    rank_col_data = data.index.values
                else:  # 表示フィールド指定あるとき
                    rank_col_data = data[rank_col].values
            # 誤差最大値
            maxerror = np.max(np.abs(y_pred - y_real))
            # ヒートマップをプロット
            cls._reg_heat_plot(model, X, y_pred, y_real, x_heat, x_not_heat, x_heat_indices,
                               pair_sigmarange = pair_sigmarange, pair_sigmainterval=pair_sigmainterval, heat_extendsigma=heat_extendsigma,
                               vmin=vmin, vmax=vmax, plot_scatter=plot_scatter, maxerror=maxerror,
                               rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_data,
                               rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                               subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_real)):
                # 表示対象以外のCVなら飛ばす
                if i not in display_cv_indices:
                    continue
                print(f'cv_number={i}')
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_real[train]
                X_test = X[test]
                y_test = y_real[test]
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # 誤差上位表示用データ取得
                if rank_number is not None:
                    if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                        rank_col_test = data.index.values[test]
                    else:  # 表示フィールド指定あるとき
                        rank_col_test = data[rank_col].values[test]
                # 誤差最大値
                maxerror = np.max(np.abs(y_pred - y_test))
                # ヒートマップをプロット
                cls._reg_heat_plot(model, X, y_pred, y_real, x_heat, x_not_heat, x_heat_indices,
                                   pair_sigmarange = pair_sigmarange, pair_sigmainterval = pair_sigmainterval, heat_extendsigma=heat_extendsigma,
                                   vmin=vmin, vmax=vmax, plot_scatter = plot_scatter, maxerror=maxerror,
                                   rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_data,
                                   rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                                   subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws)
from typing import List, Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LinearRegression
import decimal

class regplot():
    # regression_heat_plotメソッド (回帰モデルヒートマップ表示)における、散布図カラーマップ
    HEAT_SCATTER_HUECOLORS = ['red', 'mediumblue', 'darkorange', 'darkmagenta', 'cyan',  'pink', 'brown', 'gold', 'grey']

    def _round_digits(src: float, rounddigit: int = None, method='decimal'):
        """
        指定桁数で小数を丸める

        Parameters
        ----------
        src : float
            丸め対象の数値
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

    def _make_score_dict(y_true, y_pred, scores):
        """
        回帰評価指標を算出してdict化
        """
        score_dict = {}
        for scoring in scores:
            if scoring == 'r2':
                score_dict['r2'] = r2_score(y_true, y_pred)
            elif scoring == 'mae':
                score_dict['mae'] = mean_absolute_error(y_true, y_pred)
            elif scoring == 'rmse':
                score_dict['rmse'] = mean_squared_error(y_true, y_pred)
            elif scoring == 'rmsle':
                score_dict['rmsle'] = mean_squared_log_error(y_true, y_pred)
            elif scoring == 'max_error':
                score_dict['max_error'] = max([abs(p - r) for r, p in zip(y_true, y_pred)])
        return score_dict

    @classmethod
    def _rank_display(cls, y_true, y_pred, rank_number, rank_col, rank_col_data, x=None, ax=None, rounddigit=None):
        """
        誤差上位を文字プロット

        Parameters
        ----------
        y_true : np.ndarray
            目的変数実測値
        y_pred : np.ndarray
            目的変数予測値
        rank_number : int
            誤差上位何番目までを文字表示するか
        rank_col : List[str]
            誤差上位と一緒に表示するフィールド名 (NoneならIndexを使用)
        x : np.ndarray
            説明変数の値 (Noneなら横軸y_true縦軸y_pred、Noneでなければ横軸x縦軸y_true)
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax is None:
            ax=plt

        if rank_col is None:
            rank_col = 'index'
        y_error = y_pred - y_true
        y_error_abs = np.abs(y_error)
        rank_index  = np.argsort(-y_error_abs)[:rank_number]
        for rank, i in enumerate(rank_index):
            error = cls._round_digits(y_error[i], rounddigit=rounddigit, method='decimal')
            rank_text = f'{rank+1}\n{rank_col}={rank_col_data[i]}\nerror={error}'
            if x is None:  # 横軸y_true縦軸y_pred (regression_pred_trueメソッド用)
                ax.text(y_true[i], y_pred[i], rank_text, verticalalignment='center', horizontalalignment='left')
            else:  # 横軸x縦軸y_true (regression_plot_1dメソッド用)
                ax.text(x[i], y_true[i], rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def _scatterplot_ndarray(cls, x, x_name, y, y_name, hue_data, hue_name, ax):
        """
        np.ndarrayを入力として散布図表示(scatterplot)
        """
        # X値とY値を合体してDataFrame化
        data = np.stack([x, y], axis=1)
        data = pd.DataFrame(data, columns=[x_name, y_name])
        # 色分け指定しているとき、色分け用のフィールドを追加
        if hue_data is not None:
            if hue_name is None:
                hue_name = 'hue'
            data[hue_name] = pd.Series(hue_data)
        # 散布図プロット
        sns.scatterplot(x=x_name, y=y_name, data=data, ax=ax, hue=hue_name)

    @classmethod
    def _plot_pred_true(cls, y_true, y_pred, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=200, rounddigit=None,
                        score_dict=None):
        """
        予測値と実測値を、回帰評価指標とともにプロット

        Parameters
        ----------
        y_true : ndarray
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
        cls._scatterplot_ndarray(y_true, 'y_true', y_pred, 'y_pred', hue_data, hue_name, ax)

        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax is None:
            ax=plt
        # score_dictがNoneのとき、空のDictを加瀬宇
        if score_dict is None:
            score_dict = {}

        # 予測値=実測値の線を作成
        true_min = np.amin(y_true)
        true_max = np.amax(y_true)
        true_line = np.linspace(true_min, true_max, linesplit)
        # 評価指標文字列作成
        score_list = [f'{k}={v}' for k, v in cls._round_dict_digits(score_dict, rounddigit, 'sig').items()]
        score_text = "\n".join(score_list)
        # 線と文字をプロット
        ax.plot(true_line, true_line, color=linecolor)
        ax.text(true_max, np.amin(y_pred), score_text, verticalalignment='bottom', horizontalalignment='right')
    
    @classmethod
    def regression_pred_true(cls, model, x: List[str], y: str, data: pd.DataFrame, hue=None, linecolor='red', rounddigit=3,
                             rank_number=None, rank_col=None, scores='mae', cv_stats='mean', cv=None, cv_seed=42,
                             model_params=None, fit_params=None, subplot_kws=None):

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
        cv_stats : Dict
            クロスバリデーション時に表示する統計値 ('mean', 'median', 'max', 'min')
        cv : None or int or KFold
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed : int
            クロスバリデーションの乱数シード
        model_params : Dict[str, float]
            回帰モデルに渡すパラメータ (チューニング後のパラメータがgood、Noneならデフォルト)
        fit_params : Dict[str, float]
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        subplot_kws : Dict[str, float]
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        """
        # scoresの型をListに統一
        if scores is None:
            scores = []
        elif isinstance(scores, str):
            scores = [scores]
        elif not isinstance(scores, list):
            raise Exception('the "scores" argument must be str or list[str]')
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}

        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        elif isinstance(x, str):
            X = data[[x]].values
        else:
            raise Exception('the "x" argument must be str or list[str]')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            # 評価指標算出
            score_dict = cls._make_score_dict(y_true, y_pred, scores)
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
            cls._plot_pred_true(y_true, y_pred, hue_data=hue_data, hue_name=hue_name,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict)
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_true, y_pred, rank_number, rank_col, rank_col_data, rounddigit=rounddigit)
            return score_dict
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            #LeaveOneOutかどうかを判定
            isLeaveOneOut = isinstance(cv, LeaveOneOut)
            # GroupKFold、LeaveOneGroupOutのとき、hueをグルーピング対象に指定
            split_kws={}
            if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                if hue is not None:
                    split_kws['groups'] = data[hue].values
                else:
                    raise Exception('"GroupKFold" cross validation needs "hue" argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をhueの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[hue].values))
            elif isLeaveOneOut:
                cv_num = 1
            else:
                cv_num = cv.n_splits

            # スコア種類ごとにクロスバリデーションスコアの算出
            score_all_dict = {}
            for scoring in scores:
                # cross_val_scoreでクロスバリデーション
                if scoring == 'r2':
                    score_all_dict['r2'] = cross_val_score(model, X, y_true, cv=cv, scoring='r2',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                elif scoring == 'mae':
                    neg_mae = cross_val_score(model, X, y_true, cv=cv, scoring='neg_mean_absolute_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mae'] = -neg_mae  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmse':
                    neg_rmse = cross_val_score(model, X, y_true, cv=cv, scoring='neg_root_mean_squared_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmse'] = -neg_rmse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmsle':
                    neg_msle = cross_val_score(model, X, y_true, cv=cv, scoring='neg_mean_squared_log_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmsle'] = np.sqrt(-neg_msle)  # 正負を逆にしてルートをとる
                elif scoring == 'max_error':
                    neg_max_error = cross_val_score(model, X, y_true, cv=cv, scoring='max_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['max_error'] = - neg_max_error  # scikit-learnの仕様に合わせ正負を逆に
            
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
            y_true_all = []
            y_pred_all = []
            hue_all = []
            rank_col_all = []
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                X_test = X[test]
                y_test = y_true[test]
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
                    cls._plot_pred_true(y_test, y_pred, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                                    linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict)
                    axes[i].set_title(f'Cross Validation No{i}')
                # 全体プロット用データに追加
                y_true_all.append(y_test)
                y_pred_all.append(y_pred)
                hue_all.append(hue_test)
                rank_col_all.append(rank_col_test)

            # 全体プロット用データを合体
            y_true_all = np.hstack(y_true_all)
            y_pred_all = np.hstack(y_pred_all)
            hue_all = np.hstack(hue_all)
            rank_col_all = np.hstack(rank_col_all)
            # 指標の統計値を計算
            if cv_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}
            elif cv_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}            
            elif cv_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}            
            elif cv_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
            # 全体データを学習＆評価データとして評価指標算出
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            score_dict = cls._make_score_dict(y_true, y_pred, scores)
            # 学習データ指標を指標dictに追加
            score_dict = {f'{k}_train': np.mean(v) for k, v in score_dict.items()}
            score_stats_dict.update(score_dict)
            # 全体プロット
            ax_all = axes if isLeaveOneOut else axes[cv_num]
            cls._plot_pred_true(y_true_all, y_pred_all, hue_data=hue_all, hue_name=hue_name, ax=ax_all,
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict)
            ax_all.set_title('All Cross Validations')
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_true_all, y_pred_all, rank_number, rank_col, rank_col_all,
                                  ax=ax_all, rounddigit=rounddigit)
            return score_stats_dict

    @classmethod
    def linear_plot(cls, x: str, y: str, data: pd.DataFrame, ax=None, hue=None, linecolor='red',
                    rounddigit=5, plot_scores=True):
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
            表示対象のaxes (Noneならplt.plotで1枚ごとにプロット)
        hue : str
            色分け指定カラム (列名指定)
        linecolor : str
            回帰直線の色
        rounddigit: int
            表示指標の小数丸め桁数
        plot_scores: bool
            回帰式、ピアソンの相関係数およびp値の表示有無 (Trueなら表示あり)
        """
        # xをndarray化
        if isinstance(x, str):
            X = data[[x]].values
        else:
            raise Exception('the "x" argument must be str')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')

        # まずは散布図プロット
        ax = sns.scatterplot(x=x, y=y, data=data, ax=ax, hue=hue)

        # 線形回帰モデル作成
        lr = LinearRegression()
        lr.fit(X, y_true)
        xmin = np.amin(X)
        xmax = np.amax(X)
        linesplit=200
        Xline = np.linspace(xmin, xmax, linesplit)
        Xline = Xline.reshape(len(Xline), 1)
        # 回帰線を描画
        ax.plot(Xline, lr.predict(Xline), color=linecolor)

        # 回帰式、ピアソンの相関係数およびp値を表示
        if plot_scores == True:
            # 回帰式
            coef = cls._round_digits(lr.coef_[0], rounddigit=rounddigit, method="decimal")
            intercept = cls._round_digits(lr.intercept_, rounddigit=rounddigit, method="decimal")
            equation = f'y={coef}x+{intercept}' if intercept >= 0 else f'y={coef}x-{-intercept}'
            # ピアソン相関係数
            pearsonr = stats.pearsonr(data[x], data[y])
            r = cls._round_digits(pearsonr[0], rounddigit=rounddigit, method="decimal")
            pvalue = cls._round_digits(pearsonr[1], rounddigit=rounddigit, method="decimal")            
            # プロット
            rtext = f'{equation}\nr={r}\np={pvalue}'
            ax.text(xmax, np.amin(y_true), rtext, verticalalignment='bottom', horizontalalignment='right')

    @classmethod
    def _model_plot_1d(cls, trained_model, X, y_true, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=1000, rounddigit=None,
                       score_dict=None):
        """
        1次説明変数回帰曲線を、回帰評価指標とともにプロット

        Parameters
        ----------
        trained_model : 
            学習済の回帰モデル(scikit-learn API)
        X : ndarray
            説明変数
        y_true : ndarray
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
        cls._scatterplot_ndarray(np.ravel(X), 'X', y_true, 'Y', hue_data, hue_name, ax)

        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax is None:
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
        ax.text(xmax, np.amin(y_true), score_text, verticalalignment='bottom', horizontalalignment='right')

    @classmethod
    def regression_plot_1d(cls, model, x: str, y: str, data: pd.DataFrame, hue=None, linecolor='red', rounddigit=3,
                           rank_number=None, rank_col=None, scores='mae', cv_stats='mean', cv=None, cv_seed=42,
                           model_params=None, fit_params=None, subplot_kws=None):
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
        rank_number : int
            誤差上位何番目までを文字表示するか
        rank_col : List[str]
            誤差上位と一緒に表示するフィールド名 (NoneならIndexを使用)
        scores : str or list[str]
            算出する評価指標 ('r2', 'mae', 'rmse', 'rmsle', 'max_error')
        cv_stats : Dict
            クロスバリデーション時に表示する統計値 ('mean', 'median', 'max', 'min')
        cv : None or int or KFold
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed : int
            クロスバリデーションの乱数シード
        model_params: Dict, optional
            回帰モデルに渡すパラメータ (チューニング後のパラメータがgood、Noneならデフォルト)
        fit_params : Dict[str, float]
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        subplot_kws : Dict[str, float]
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        """
        # scoresの型をListに統一
        if scores is None:
            scores = []
        elif isinstance(scores, str):
            scores = [scores]
        elif not isinstance(scores, list):
            raise Exception('the "scores" argument must be str or list[str]')
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        
        # xをndarray化
        if isinstance(x, str):
            X = data[[x]].values
        else:
            raise Exception('the "x" argument must be str')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            # 評価指標算出
            score_dict = cls._make_score_dict(y_true, y_pred, scores)
            # 色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 誤差上位表示用データ取得
            if rank_number is not None:
                if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                    rank_col_data = data.index.values
                else:  # 表示フィールド指定あるとき
                    rank_col_data = data[rank_col].values
            # 回帰線プロット
            cls._model_plot_1d(model, X, y_true, hue_data=hue_data, hue_name=hue_name,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict)
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_true, y_pred, rank_number, rank_col, rank_col_data, x=X, rounddigit=rounddigit)
            return score_dict
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            #LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_plot_1d" method does not support "LeaveOneOut" cross validation')
            # GroupKFold、LeaveOneGroupOutのとき、hueをグルーピング対象に指定
            split_kws={}
            if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                if hue is not None:
                    split_kws['groups'] = data[hue].values                    
                else:
                    raise Exception('"GroupKFold" cross validation needs "hue" argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をhueの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[hue].values))
            else:
                cv_num = cv.n_splits

            # スコア種類ごとにクロスバリデーションスコアの算出
            score_all_dict = {}
            for scoring in scores:
                # cross_val_scoreでクロスバリデーション
                if scoring == 'r2':
                    score_all_dict['r2'] = cross_val_score(model, X, y_true, cv=cv, scoring='r2',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                elif scoring == 'mae':
                    neg_mae = cross_val_score(model, X, y_true, cv=cv, scoring='neg_mean_absolute_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mae'] = -neg_mae  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmse':
                    neg_rmse = cross_val_score(model, X, y_true, cv=cv, scoring='neg_root_mean_squared_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmse'] = -neg_rmse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmsle':
                    neg_msle = cross_val_score(model, X, y_true, cv=cv, scoring='neg_mean_squared_log_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmsle'] = np.sqrt(-neg_msle)  # 正負を逆にしてルートをとる
                elif scoring == 'max_error':
                    neg_max_error = cross_val_score(model, X, y_true, cv=cv, scoring='max_error',
                                                           fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['max_error'] = - neg_max_error  # scikit-learnの仕様に合わせ正負を逆に
            
            # 表示用のaxes作成
            # クロスバリデーションごとに図作成
            if 'figsize' not in subplot_kws.keys():
                subplot_kws['figsize'] = (6, (cv_num + 1) * 6)
            fig, axes = plt.subplots(cv_num + 1, 1, **subplot_kws)

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                X_test = X[test]
                y_test = y_true[test]
                # 色分け用データ取得(していないときは、クロスバリデーション番号を使用、LeaveOuneOutのときは番号分けない)
                if hue is None:
                    hue_test = np.full(len(test) ,f'cv_{i}')
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
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                # CV内結果をプロット
                score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                cls._model_plot_1d(model, X_test, y_test, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                            linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict)
                # 誤差上位を文字表示
                if rank_number is not None:
                    cls._rank_display(y_test, model.predict(X_test), rank_number, rank_col, rank_col_test, x=X_test, ax=axes[i], rounddigit=rounddigit)
                axes[i].set_title(f'Cross Validation No{i}')

            # 指標の統計値を計算
            if cv_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}
            elif cv_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}
            elif cv_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}
            elif cv_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
            # 全体データを学習＆評価データとして評価指標算出
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            score_dict = cls._make_score_dict(y_true, y_pred, scores)
            # 学習データ指標を指標dictに追加
            score_dict = {f'{k}_train': np.mean(v) for k, v in score_dict.items()}
            score_stats_dict.update(score_dict)
            # 全体色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 全体プロット
            ax_all = axes[cv_num]
            cls._model_plot_1d(model, X, y_true, hue_data=hue_data, hue_name=hue_name, ax=ax_all,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict)
            ax_all.set_title('All Cross Validations')
            return score_stats_dict

    @classmethod
    def _reg_heat_plot_2d(cls, trained_model, x_heat, y_true_col, y_pred_col, rank_col, data, x_heat_indices, hue_name,
                          x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                          vmin, vmax, ax, plot_scatter, maxerror, rank_dict, scatter_hue_dict,
                          rounddigit_rank, rounddigit_x1, rounddigit_x2,
                          heat_kws=None, scatter_kws=None):
        """
        回帰予測値ヒートマップと各種散布図の表示
        (regression_heat_plotメソッドの描画処理部分)
        """
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()

        # ヒートマップ用グリッドデータを作成
        xx = np.linspace(x1_start, x1_end, heat_division)
        yy = np.linspace(x2_start, x2_end, heat_division)
        X1, X2 = np.meshgrid(xx, yy)
        X_grid = np.c_[X1.ravel(), X2.ravel()]
        df_heat = pd.DataFrame(X_grid, columns=x_heat)
        # 推論用に全説明変数を保持したndarrayを作成 (ヒートマップ非使用変数は固定値other_xとして追加)
        n_rows = X_grid.shape[0]
        X_all = []
        other_add_flg = False
        for i in range(2 + len(other_x)):
            if i == x_heat_indices[0]: # ヒートマップ使用変数(1個目)を追加
                X_all.append(X_grid[:, 0].reshape(n_rows, 1))
            elif i == x_heat_indices[1]: # ヒートマップ使用変数(2個目)を追加
                X_all.append(X_grid[:, 1].reshape(n_rows, 1))
            elif len(other_x) >= 1 and not other_add_flg:  # ヒートマップ非使用変数(1個目)を固定値として追加
                X_all.append(np.full((n_rows, 1), other_x[0]))
                other_add_flg = True
            elif len(other_x) == 2:  # ヒートマップ非使用変数(2個目)を固定値として追加
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
        # 横軸の列数がheat_divisionに満たない時、分解能不足のためrounddigit_x1桁数を増やすようエラー表示
        if len(df_heat_pivot.columns) < heat_division:
            raise Exception(f'the "rounddigit_x1" argument must be bigger than {rounddigit_x1} because of the shortage of the "{x_heat[0]}" resolution')
        # 縦軸の列数がheat_divisionに満たない時、分解能不足のためrounddigit_x2桁数を増やすようエラー表示
        if len(df_heat_pivot) < heat_division:
            raise Exception(f'the "rounddigit_x2" argument must be bigger than {rounddigit_x2} because of the shortage of the "{x_heat[1]}" resolution')

        # ヒートマップのカラーマップ指定ないとき、YlGnを指定
        if 'cmap' not in heat_kws.keys():
            heat_kws['cmap'] = 'YlGn'
        # ヒートマップをプロット
        sns.heatmap(df_heat_pivot, ax=ax, vmax=vmax, vmin=vmin, center=(vmax+vmin)/2, **heat_kws)

        # 誤差散布図をプロット
        if plot_scatter is not None:
            # 軸範囲が0～heat_divisionになっているので、スケール変換
            x1_scatter = 0.5 + (data[x_heat[0]].values - x1_start) * (heat_division - 1) / (x1_end - x1_start)
            x2_scatter = 0.5 + (data[x_heat[1]].values - x2_start) * (heat_division - 1) / (x2_end - x2_start)
            # 色分け
            if plot_scatter == 'error':  # 誤差で色分け
                scatter_c = data[y_pred_col].values - data[y_true_col].values
                scatter_vmin = -maxerror
                scatter_vmax = maxerror
                if 'cmap' not in scatter_kws.keys():  # 散布図のカラーマップ指定ないとき、seismicを指定
                    scatter_kws['cmap'] = 'seismic'
            elif plot_scatter == 'true':  # 真値で色分け
                scatter_c = data[y_true_col].values
                scatter_vmin = vmin
                scatter_vmax = vmax
                if 'cmap' not in scatter_kws.keys():  # 散布図のカラーマップ指定ないとき、ヒートマップと同cmap使用
                    scatter_kws['cmap'] = heat_kws['cmap']
                if 'edgecolors' not in scatter_kws.keys():  # 線の色指定ないとき、ブラウンを指定
                    scatter_kws['edgecolors'] = 'brown'
            # 散布図プロット (誤差or真値で色分けしたとき)
            if plot_scatter == 'error' or plot_scatter == 'true':
                ax.scatter(x1_scatter, x2_scatter, vmin=scatter_vmin, vmax=scatter_vmax, c=scatter_c, **scatter_kws)
            # 散布図プロット (hue列名で色分けしたとき)
            if plot_scatter == 'hue':
                scatter_data = pd.DataFrame(np.stack([x1_scatter, x2_scatter, data[hue_name]], 1), columns=['x1', 'x2', hue_name])
                for name, group in scatter_data.groupby(hue_name):
                    ax.scatter(group['x1'].values, group['x2'].values, label=name, c=scatter_hue_dict[name], **scatter_kws)
                ax.legend()
        
        # 誤差上位を文字表示
        df_rank = data[data.index.isin(rank_dict.keys())]
        for index, row in df_rank.iterrows():
            # rank_col指定ないとき、indexがfloat型に変換されてしまうので、int型に戻す
            rank_col_value = int(row[rank_col]) if rank_col == 'index' else row[rank_col]
            # 誤差を計算してテキスト化
            error = cls._round_digits(row['y_pred'] - row['y_true'], rounddigit=rounddigit_rank)
            rank_text = f'{rank_dict[index]+1}\n{rank_col}={rank_col_value}\nerror={error}'
            # 軸範囲が0～heat_divisionになっているので、スケール変換してプロット
            x1_text = 0.5 + (row[x_heat[0]] - x1_start) * (heat_division - 1) / (x1_end - x1_start)
            x2_text = 0.5 + (row[x_heat[1]] - x2_start) * (heat_division - 1) / (x2_end - x2_start)
            ax.text(x1_text, x2_text, rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def _reg_heat_plot(cls, trained_model, X, y_pred, y_true, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                       pair_sigmarange=2.0, pair_sigmainterval=0.5, heat_extendsigma=0.5, heat_division=30, 
                       vmin=None, vmax=None, plot_scatter='true', maxerror=None,
                       rank_number=None, rank_col=None, rank_col_data=None, scatter_hue_dict=None,
                       rounddigit_rank=None, rounddigit_x1=None, rounddigit_x2=None, rounddigit_x3=None,
                       cv_index=None, subplot_kws=None, heat_kws=None, scatter_kws=None):
        """
        回帰予測値ヒートマップ表示の、説明変数の数に応じた分岐処理
        (regression_heat_plotメソッド処理のうち、説明変数の数に応じたデータ分割等を行う)
        """
        # 説明変数の数
        x_num = X.shape[1]
        # ヒートマップ使用DataFrame
        df_heat = pd.DataFrame(X[:, x_heat_indices], columns=x_heat)
        # ヒートマップ非使用DataFrame
        X_not_heat = X[:, [i for i in range(X.shape[1]) if i not in x_heat_indices]]
        df_not_heat = pd.DataFrame(X_not_heat, columns=x_not_heat)
        # 結合＆目的変数実測値と予測値追加
        df_all = df_heat.join(df_not_heat)
        df_all = df_all.join(pd.DataFrame(y_true, columns=['y_true']))
        df_all = df_all.join(pd.DataFrame(y_pred, columns=['y_pred']))
        # ヒートップ非使用変数を標準化してDataFrameに追加
        if x_num >= 3:
            X_not_heat_norm = stats.zscore(X_not_heat)
            df_all = df_all.join(pd.DataFrame(X_not_heat_norm, columns=[f'normalize_{c}' for c in x_not_heat]))
        # 誤差上位表示用IDデータをDataFrameに追加
        rank_col = 'index' if rank_col is None else rank_col
        df_all = df_all.join(pd.DataFrame(rank_col_data, columns=[rank_col]))
        # 散布図色分け用列をDataFrameに追加(hue_nameがNoneでないときのみ))
        if hue_name is not None:
            df_all = df_all.join(pd.DataFrame(hue_data, columns=[hue_name]))

        # 誤差の順位を計算
        if rank_number is not None:
            y_error_abs = np.abs(y_pred - y_true)
            rank_index  = np.argsort(-y_error_abs)[:rank_number]
            rank_dict = dict(zip(rank_index.tolist(), range(rank_number)))
        else:
            rank_dict = {}

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
        # ヒートップ非使用変数をプロットする範囲の下限(標準化後)
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
        if cv_index is not None:
            fig.suptitle(f'CV No.{cv_index}')

        # 図ごとにプロット
        for i in range(pair_h):
            for j in range(pair_w):
                # pair縦軸変数(標準化後)の最小値
                if i == 0:
                    h_min = -float('inf')
                    h_mean = pair_min - pair_sigmainterval / 2  # ヒートマップ非使用変数指定用の平均値
                else:
                    h_min = pair_min + (i - 1) * pair_sigmainterval
                    h_mean = pair_min + (i - 0.5) * pair_sigmainterval  # ヒートマップ非使用変数指定用の平均値
                # pair縦軸変数(標準化後)の最大値
                if i == pair_h - 1:
                    h_max = float('inf')
                else:
                    h_max = pair_min + i * pair_sigmainterval
                # pair横軸変数(標準化後)の最小値
                if j == 0:
                    w_min = -float('inf')
                    w_mean = pair_min - pair_sigmainterval / 2  # ヒートマップ非使用変数指定用の平均値
                else:
                    w_min = pair_min + (j - 1) * pair_sigmainterval
                    w_mean = pair_min + (j - 0.5) * pair_sigmainterval  # ヒートマップ非使用変数指定用の平均値
                # pair横軸変数(標準化後)の最大値
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
                    # 縦軸変数範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_heat[0]}'] >= h_min) & (df_all[f'normalize_{x_not_heat[0]}'] < h_max)].copy()
                    # ヒートマップ非使用変数の標準化逆変換
                    x3_mean = np.mean(X_not_heat[:, 0])
                    x3_std = np.std(X_not_heat[:, 0])
                    other_x = [h_mean * x3_std + x3_mean]
                # 説明変数が4次元のとき (図はpair_n × pair_n枚)
                elif x_num == 4:
                    ax = axes[j, i]
                    # 縦軸変数範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_heat[0]}'] >= h_min) & (df_all[f'normalize_{x_not_heat[0]}'] < h_max)].copy()
                    # 横軸変数範囲内のみのデータを抽出
                    df_pair = df_pair[(df_pair[f'normalize_{x_not_heat[1]}'] >= w_min) & (df_pair[f'normalize_{x_not_heat[1]}'] < w_max)]
                    # ヒートマップ非使用変数の標準化逆変換
                    x3_mean = np.mean(X_not_heat[:, 0])
                    x3_std = np.std(X_not_heat[:, 0])
                    x4_mean = np.mean(X_not_heat[:, 1])
                    x4_std = np.std(X_not_heat[:, 1])
                    other_x = [h_mean * x3_std + x3_mean, w_mean * x4_std + x4_mean]
                
                cls._reg_heat_plot_2d(trained_model, x_heat, 'y_true', 'y_pred', rank_col, df_pair, x_heat_indices, hue_name,
                                      x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                                      vmin, vmax, ax, plot_scatter, maxerror, rank_dict, scatter_hue_dict,
                                      rounddigit_rank, rounddigit_x1, rounddigit_x2,
                                      heat_kws=heat_kws, scatter_kws=scatter_kws)

                # グラフタイトルとして、ヒートマップ非使用変数の範囲を記載（説明変数が3次元以上のとき）
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
    def regression_heat_plot(cls, model, x: List[str], y: str, data: pd.DataFrame, x_heat: List[str] = None, scatter_hue=None,
                             pair_sigmarange = 1.5, pair_sigmainterval = 0.5, heat_extendsigma = 0.5, heat_division = 30, color_extendsigma = 0.5,
                             plot_scatter = 'true', rounddigit_rank=3, rounddigit_x1=2, rounddigit_x2=2, rounddigit_x3=2, rank_number=None, rank_col=None,
                             cv=None, cv_seed=42, display_cv_indices = 0,
                             model_params=None, fit_params=None, subplot_kws=None, heat_kws=None, scatter_kws=None):
        """
        2～4次元説明変数の回帰モデルをヒートマップで可視化

        Parameters
        ----------
        model:
            使用する回帰モデル(scikit-learn API)
        x: List[str]
            説明変数カラム (列名指定)
        y: str
            目的変数カラム (列名指定)
        data: pd.DataFrame
            フィッティング対象のデータ
        x_heat: List[str], optional
            説明変数のうち、ヒートマップ表示対象のカラム (Noneなら前から2カラム自動選択)
        scatter_hue : str, optional
            散布図色分け指定カラム (列名指定, plot_scatter='hue'のときのみ有効)
        pair_sigmarange: float, optional
            ヒートマップ非使用変数の分割範囲 (pair_sigmarange=2なら、-2σ~2σの範囲でpair_sigmaintervalに従いヒートマップ分割)
        pair_sigmainterval: float, optional
            ヒートマップ非使用変数の1枚あたり表示範囲 (pair_sigmainterval=0.5なら、‥1σ~-0.5σ, 0.5σ~-0σ, 0σ~0.5σ, 0.5σ~1σ‥というようにヒートマップ分割)
        heat_extendsigma: float, optional
            ヒートマップ縦軸横軸の表示拡張範囲 (ヒートマップ使用変数の最大最小値 + extendsigmaが横軸範囲となる)
        heat_division: int, optional
            ヒートマップ縦軸横軸の解像度
        color_extendsigma: float, optional
            ヒートマップの色分け最大最小値拡張範囲(y_trueの最大最小値 ± y_trueの標準偏差 × color_extendsigma)
        plot_scatter: str, optional
            散布図の描画種類('error':誤差で色分け, 'true':真値で色分け, 'hue':引数hue指定列で色分け, None:散布図表示なし)
        rounddigit_rank: int, optional
            表示指標の小数丸め桁数
        rounddigit_x1: int, optional
            ヒートマップ横軸の小数丸め桁数
        rounddigit_x2: int, optional
            ヒートマップ縦軸の小数丸め桁数
        rounddigit_x3: int, optional
            ヒートマップ非表示軸の小数丸め桁数
        rank_number: int, optional
            誤差上位何番目までを文字表示するか
        rank_col: str, optional
            誤差上位と一緒に表示するフィールド名 (NoneならIndexを使用)
        cv: int or KFold, optional
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed: int, optional
            クロスバリデーションの乱数シード
        display_cv_indices: int, optional
            表示対象のクロスバリデーション番号 (指定したCV番号での回帰結果が表示される。リスト指定も可)
        model_params: Dict, optional
            回帰モデルに渡すパラメータ (チューニング後のパラメータがgood、Noneならデフォルト)
        fit_params: Dict, optional
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        subplot_kws: dict, optional
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        heat_kws: dict, optional
            ヒートマップ用のsns.heatmap()に渡す引数
        scatter_kws: dict, optional
            散布図用のax.scatter()に渡す引数
        """
        # 説明変数xの次元が2～4以外ならエラーを出す
        if len(x) < 2 or len(x) > 4:
            raise Exception('length of x must be 2 to 4')
        
        # display_cv_indicesをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x, list):
            raise Exception('the "cv_display_num" argument must be int or List[int]')
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # heat_kwsがNoneなら空のdictを入力
        if heat_kws is None:
            heat_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
            
        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        else:
            raise Exception('the "x" argument must be str or str')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')
        
        # ヒートマップ表示用の列を抽出
        if x_heat is None:  # 列名指定していないとき、前から2列を抽出
            x_heat = x[:2]
            x_heat_indices = [0, 1]
        else:  # 列名指定しているとき、該当列のXにおけるインデックス(0～3)を保持
            if len(x_heat) != 2:
                raise Exception('length of x_heat must be 2')
            x_heat_indices = []
            for colname in x_heat:
                x_heat_indices.append(x.index(colname))
        # ヒートマップ表示以外の列
        x_not_heat = [colname for colname in x if colname not in x_heat]        
        # ヒートマップの色分け最大最小値(y_trueの最大最小値 ± y_trueの標準偏差 × color_extendsigma)
        y_true_std = np.std(y_true)
        vmin = np.min(y_true) - y_true_std * color_extendsigma
        vmax = np.max(y_true) + y_true_std * color_extendsigma

        # 引数plot_scatter='hue'とscatter_hueが同時指定されていないとき、エラーを出す
        if scatter_hue is not None:
            if plot_scatter != 'hue' and not isinstance(cv, GroupKFold) and not isinstance(cv, LeaveOneGroupOut):
                raise Exception('the "plot_scatter" argument must be "hue" when the argument "scatter_hue" is not None')
        elif plot_scatter == 'hue':
            raise Exception('the "scatter_hue" argument is required when the argument "plot_scatter" is "hue"')
        # 引数plot_scatter='hue'のとき、色分け対象列とカラーマップを紐づけ(色分けを全ての図で統一用)
        if plot_scatter == 'hue':
            hue_list = data[scatter_hue].values.tolist()
            hue_list = sorted(set(hue_list), key=hue_list.index)
            scatter_hue_dict = dict(zip(hue_list, cls.HEAT_SCATTER_HUECOLORS[0:len(hue_list)]))
        else:
            scatter_hue_dict = None
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            # 誤差上位表示用データ取得
            if rank_number is not None:
                if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                    rank_col_data = data.index.values
                else:  # 表示フィールド指定あるとき
                    rank_col_data = data[rank_col].values
            else:
                rank_col_data = None
            # 誤差最大値
            maxerror = np.max(np.abs(y_pred - y_true))
            # 散布図色分け用データ取得(plot_scatter='hue'のときのみ有効)
            hue_data = data[scatter_hue] if scatter_hue is not None and plot_scatter=='hue' else None
            hue_name = scatter_hue if scatter_hue is not None and plot_scatter=='hue' else None
            # ヒートマップをプロット
            cls._reg_heat_plot(model, X, y_pred, y_true, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                               pair_sigmarange = pair_sigmarange, pair_sigmainterval=pair_sigmainterval, heat_extendsigma=heat_extendsigma, heat_division=heat_division,
                               vmin=vmin, vmax=vmax, plot_scatter=plot_scatter, maxerror=maxerror,
                               rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_data, scatter_hue_dict=scatter_hue_dict,
                               rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                               cv_index=None, subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
            # GroupKFold、LeaveOneGroupOutのとき、scatter_hueをグルーピング対象に指定
            split_kws={}
            if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                if scatter_hue is not None:
                    split_kws['groups'] = data[scatter_hue].values
                else:
                    raise Exception('"GroupKFold" cross validation needs "hue" argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をscatter_hueの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[scatter_hue].values))
            else:
                cv_num = cv.n_splits

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示対象以外のCVなら飛ばす
                if i not in display_cv_indices:
                    continue
                print(f'cv_number={i}/{cv_num}')
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                X_test = X[test]
                y_test = y_true[test]
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # 誤差上位表示用データ取得
                if rank_number is not None:
                    if rank_col is None:  # 表示フィールド指定ないとき、Index使用
                        rank_col_test = data.index.values[test]
                    else:  # 表示フィールド指定あるとき
                        rank_col_test = data[rank_col].values[test]
                else:
                    rank_col_test = None
                # 誤差最大値
                maxerror = np.max(np.abs(y_pred - y_test))
                # 散布図色分け用データ取得(plot_scatter='hue'のときのみ有効))
                hue_data = data[scatter_hue].values[test] if scatter_hue is not None and plot_scatter=='hue' else None
                hue_name = scatter_hue if scatter_hue is not None and plot_scatter=='hue' else None
                # ヒートマップをプロット
                cls._reg_heat_plot(model, X_test, y_pred, y_test, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                                   pair_sigmarange = pair_sigmarange, pair_sigmainterval = pair_sigmainterval, heat_extendsigma=heat_extendsigma, heat_division=heat_division,
                                   vmin=vmin, vmax=vmax, plot_scatter = plot_scatter, maxerror=maxerror,
                                   rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_test, scatter_hue_dict=scatter_hue_dict,
                                   rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                                   cv_index=i, subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws)


class classplot():
    # 散布図カラーリスト
    SCATTER_COLORS = ['green', 'red', 'mediumblue', 'brown', 'darkmagenta', 'darkorange', 'gold', 'grey']
    # クラス確率図カラーマップ
    PROB_CMAP = ['Greens', 'Reds', 'Blues', 'YlOrBr', 'Purples', 'OrRd', 'Wistia', 'Greys']
    # デフォルトでの決定境界図の透明度(alpha)
    DEFAULT_SEPARATOR_ALPHA = 0.3
    # デフォルトでのクラス確率図等高線モードの透明度(alpha)
    DEFAULT_PROBA_CONTOURF_ALPHA = 0.5
    # デフォルトでのクラス確率図透明度補正シグモイド関数のゲイン
    DEFAULT_PROBA_CONTOURF_SIG_GAIN = 0.5
    # デフォルトでのクラス確率図の等高線段階数
    DEFAULT_PROBA_CONTOURF_LEVELS = 10
    # デフォルトでのクラス確率図RGB画像モードの透明度(alpha)
    DEFAULT_PROBA_RGB_ALPHA = 0.45

    def _round_digits(src: float, rounddigit: int = None, method='decimal'):
        """
        指定桁数で小数を丸める

        Parameters
        ----------
        src : float
            丸め対象の数値
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
    def _chart_plot_2d(cls, trained_model, x_chart, y_true_col, y_pred_col, data, x_chart_indices,
                       x1_start, x1_end, x2_start, x2_end, other_x, chart_scale,
                       proba_pred_col, proba_class_indices, ax, plot_border, plot_scatter,
                       scatter_color_dict, scatter_marker_dict, proba_cmap_dict, proba_type,
                       contourf_kws=None, imshow_kws=None, scatter_kws=None):
        """
        分類チャート（決定境界図 or クラス確率図）と各種散布図の表示
        (class_separator_plotあるいはclass_prob_plotメソッドの描画処理部分)
        """
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()

        # 図のサイズからグリッド数を取得
        xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
        # チャート用グリッドデータを作成
        xx = np.linspace(x1_start, x1_end, num=int(xnum/chart_scale))
        yy = np.linspace(x2_start, x2_end, num=int(ynum/chart_scale))
        X1, X2 = np.meshgrid(xx, yy)
        X_grid = np.c_[X1.ravel(), X2.ravel()]
        # 推論用に全説明変数を保持したndarrayを作成 (チャート非使用変数は固定値other_xとして追加)
        n_rows = X_grid.shape[0]
        X_all = []
        other_add_flg = False
        for i in range(2 + len(other_x)):
            if i == x_chart_indices[0]: # チャート使用変数(1個目)を追加
                X_all.append(X_grid[:, 0].reshape(n_rows, 1))
            elif i == x_chart_indices[1]: # チャート使用変数(2個目)を追加
                X_all.append(X_grid[:, 1].reshape(n_rows, 1))
            elif len(other_x) >= 1 and not other_add_flg:  # チャート非使用変数(1個目)を固定値として追加
                X_all.append(np.full((n_rows, 1), other_x[0]))
                other_add_flg = True
            elif len(other_x) == 2:  # チャート非使用変数(2個目)を固定値として追加
                X_all.append(np.full((n_rows, 1), other_x[1]))
        X_all = np.hstack(X_all)

        # グリッドデータに対して推論し、推定値を作成
        y_pred_grid = trained_model.predict(X_all)
        # 推定値をint型に変換
        class_int_dict = dict(zip(scatter_color_dict.keys(), range(len(scatter_color_dict))))
        y_pred_grid_int = np.vectorize(lambda x: class_int_dict[x])(y_pred_grid)
        # グリッドデータをピボット化
        y_pred_pivot = y_pred_grid_int.reshape(X1.shape)

        # 決定境界図をプロット
        if proba_pred_col is None:
            # 決定境界色分けプロット
            ax.contourf(X1, X2, y_pred_pivot,
                        levels=np.arange(y_pred_pivot.max() + 2) - 0.5,
                        **contourf_kws)

        # クラス確率図をプロット
        else:
            # クラス数
            nclass = len(proba_class_indices)
            # グリッドデータに対してクラス確率算出
            y_proba_grid = trained_model.predict_proba(X_all)[:, proba_class_indices]

            # contourfで等高線プロットするとき
            if proba_type == 'contourf':
                # alpha値を保持(描画終了後に更新前に戻すため)
                src_alpha = contourf_kws['alpha']
                # シグモイド関数(クラス数1のときalphaで、クラス数∞のとき1に漸近)でalphaを補正
                contourf_kws['alpha'] = 2*(1-src_alpha)/(1+np.exp(-cls.DEFAULT_PROBA_CONTOURF_SIG_GAIN*(nclass-1)))+2*src_alpha-1
                # クラスごとに処理
                for i in range(nclass):
                    # グリッドデータから該当クラスのみ抜き出してピボット化
                    y_proba_pivot = y_proba_grid[:, i].reshape(X1.shape)
                    # カラーマップをproba_cmap_dictの値から取得
                    cmap = list(proba_cmap_dict.values())[i]
                    # クラス確率図プロット
                    ax.contourf(X1, X2, y_proba_pivot,
                                cmap=cmap,
                                **contourf_kws)
                    # alpha値を更新(alpha/(1+alpha))
                    old_alpha = contourf_kws['alpha']
                    contourf_kws['alpha'] = old_alpha / (1 + old_alpha)
                # alpha値を更新前に戻す
                contourf_kws['alpha'] = src_alpha

            # imshowでRGB画像プロットするとき
            elif proba_type == 'imshow':
                # いったんRGB各色ゼロで埋める
                proba_g = np.zeros(X1.shape)  # 緑
                proba_r = np.zeros(X1.shape)  # 赤
                proba_b = np.zeros(X1.shape)  # 青
                # RGBいずれかのカラーマップを持つクラスが存在すれば、そのクラスの確率を格納
                for i, cmap in enumerate(proba_cmap_dict.values()):
                    if cmap == 'Greens':
                        proba_g = y_proba_grid[:, i].reshape(X1.shape)
                    elif cmap == 'Reds':
                        proba_r = y_proba_grid[:, i].reshape(X1.shape)
                    elif cmap == 'Blues':
                        proba_b = y_proba_grid[:, i].reshape(X1.shape)
                    else:
                        # imshowのとき、Greens, Reds, Blues以外のカラーマップを指定したらエラーを出す(4クラス以上は描画不可)
                        raise Exception('only "Greens, Reds, Blues" cmap are allowd if the "proba_type" argument is "imshow"')
                # RGBのデータを合体して上下反転
                im_grid = np.flip(np.stack([proba_r, proba_g, proba_b], 2), axis=0)
                # RGB画像をプロット
                ax.imshow(im_grid,
                          aspect="auto", extent=(x1_start, x1_end, x2_start, x2_end),
                          **imshow_kws)
            else:
                raise Exception('the "proba_type" argument must be "contourf" or "imshow"')

        # 境界線をプロット
        if plot_border:
            ax.contour(X1, X2, y_pred_pivot,
                       levels=np.arange(y_pred_pivot.max() + 2) - 0.5,
                       colors='k',
                       linewidths=0.5,
                       antialiased=True)

        # 散布図をプロット
        if plot_scatter is not None:
            # マーカの縁の色未指定のとき、dimgreyを指定
            if 'edgecolors' not in scatter_kws.keys():
                scatter_kws['edgecolors'] = 'dimgrey'
            # 正誤を判定
            data['error'] = (data[y_true_col] == data[y_pred_col])
            # 色分け
            if plot_scatter == 'error':  # 正誤で色分け
                cdict = {True:'blue', False:'red'}
                for name, group in data.groupby('error'):
                    ax.scatter(group[x_chart[0]].values, group[x_chart[1]].values,
                               label=name, c=cdict[name],
                               marker=scatter_marker_dict[name],
                               **scatter_kws)
            elif plot_scatter == 'class':  # クラスで色分け
                for name, group in data.groupby(y_true_col):
                    ax.scatter(group[x_chart[0]].values, group[x_chart[1]].values,
                               label=name, c=scatter_color_dict[name],
                               **scatter_kws)
            elif plot_scatter == 'class_error':  # クラスと正誤で色分け
                for name, group in data.groupby([y_true_col, 'error']):
                    ax.scatter(group[x_chart[0]].values, group[x_chart[1]].values,
                               label=f'{name[0]}   {name[1]}', c=scatter_color_dict[name[0]],
                               marker=scatter_marker_dict[name[1]],
                               **scatter_kws)
            # 凡例表示
            ax.legend()

        # 軸ラベルを追加
        ax.set_xlabel(x_chart[0])
        ax.set_ylabel(x_chart[1])

    @classmethod
    def _class_chart_plot(cls, trained_model, X, y_pred, y_true, x_chart, x_not_chart, x_chart_indices,
                       pair_sigmarange=2.0, pair_sigmainterval=0.5, chart_extendsigma=0.5, chart_scale=1,
                       proba_pred = None, proba_class_indices = None, plot_border=True, plot_scatter='class', 
                       scatter_color_dict=None, scatter_marker_dict=None, proba_cmap_dict=None,  proba_type=None,
                       rounddigit_x3=None,
                       cv_index=None, subplot_kws=None, contourf_kws=None, imshow_kws=None, scatter_kws=None):
        """
        分類チャート（決定境界図 or クラス確率図）表示の、説明変数の数に応じた分岐処理
        (class_separator_plotあるいはclass_prob_plotメソッド処理のうち、説明変数の数に応じたデータ分割等を行う)
        """
        # 説明変数の数
        x_num = X.shape[1]
        # チャート（決定境界図 or クラス確率図）使用DataFrame
        df_chart = pd.DataFrame(X[:, x_chart_indices], columns=x_chart)
        # チャート非使用DataFrame
        X_not_chart = X[:, [i for i in range(X.shape[1]) if i not in x_chart_indices]]
        df_not_chart = pd.DataFrame(X_not_chart, columns=x_not_chart)
        # 結合＆目的変数実測値と予測値追加
        df_all = df_chart.join(df_not_chart)
        df_all = df_all.join(pd.DataFrame(y_true, columns=['y_true']))
        df_all = df_all.join(pd.DataFrame(y_pred, columns=['y_pred']))
        # クラス確率追加（クラス確率図プロット時のみ）
        if proba_pred is not None:
            proba_pred_col = list(proba_cmap_dict.keys())
            df_all = df_all.join(pd.DataFrame(proba_pred, columns=[proba_pred_col]))
        else:
            proba_pred_col = None
        # チャート非使用変数を標準化してDataFrameに追加
        if x_num >= 3:
            X_not_chart_norm = stats.zscore(df_not_chart)
            df_all = df_all.join(pd.DataFrame(X_not_chart_norm, columns=[f'normalize_{c}' for c in df_not_chart]))

        # チャートのX1軸およびX2軸の表示範囲(最大最小値 + extendsigma)
        x1_min = np.min(X[:, x_chart_indices[0]])
        x1_max = np.max(X[:, x_chart_indices[0]])
        x1_std = np.std(X[:, x_chart_indices[0]])
        x1_start = x1_min - x1_std * chart_extendsigma
        x1_end = x1_max + x1_std * chart_extendsigma
        x2_min = np.min(X[:, x_chart_indices[1]])
        x2_max = np.max(X[:, x_chart_indices[1]])
        x2_std = np.std(X[:, x_chart_indices[1]])
        x2_start = x2_min - x2_std * chart_extendsigma
        x2_end = x2_max + x2_std * chart_extendsigma

        # プロットする図の数(sigmarange外「2枚」 + sigmarange内「int(pair_sigmarange / pair_sigmainterval) * 2枚」)
        pair_n = int(pair_sigmarange / pair_sigmainterval) * 2 + 2
        # チャート非使用変数をプロットする範囲の下限(標準化後)
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
        if cv_index is not None:
            fig.suptitle(f'CV No.{cv_index}')

        # 図ごとにプロット
        for i in range(pair_h):
            for j in range(pair_w):
                # pair縦軸変数(標準化後)の最小値
                if i == 0:
                    h_min = -float('inf')
                    h_mean = pair_min - pair_sigmainterval / 2  # チャート非使用変数指定用の平均値
                else:
                    h_min = pair_min + (i - 1) * pair_sigmainterval
                    h_mean = pair_min + (i - 0.5) * pair_sigmainterval  # チャート非使用変数指定用の平均値
                # pair縦軸変数(標準化後)の最大値
                if i == pair_h - 1:
                    h_max = float('inf')
                else:
                    h_max = pair_min + i * pair_sigmainterval
                # pair横軸変数(標準化後)の最小値
                if j == 0:
                    w_min = -float('inf')
                    w_mean = pair_min - pair_sigmainterval / 2  # チャート非使用変数指定用の平均値
                else:
                    w_min = pair_min + (j - 1) * pair_sigmainterval
                    w_mean = pair_min + (j - 0.5) * pair_sigmainterval  # チャート非使用変数指定用の平均値
                # pair横軸変数(標準化後)の最大値
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
                    # 縦軸変数範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_chart[0]}'] >= h_min) & (df_all[f'normalize_{x_not_chart[0]}'] < h_max)].copy()
                    # 決定境界図非使用変数の標準化逆変換
                    x3_mean = np.mean(X_not_chart[:, 0])
                    x3_std = np.std(X_not_chart[:, 0])
                    other_x = [h_mean * x3_std + x3_mean]
                # 説明変数が4次元のとき (図はpair_n × pair_n枚)
                elif x_num == 4:
                    ax = axes[j, i]
                    # 縦軸変数範囲内のみのデータを抽出
                    df_pair = df_all[(df_all[f'normalize_{x_not_chart[0]}'] >= h_min) & (df_all[f'normalize_{x_not_chart[0]}'] < h_max)].copy()
                    # 横軸変数範囲内のみのデータを抽出
                    df_pair = df_pair[(df_pair[f'normalize_{x_not_chart[1]}'] >= w_min) & (df_pair[f'normalize_{x_not_chart[1]}'] < w_max)]
                    # チャート非使用変数の標準化逆変換
                    x3_mean = np.mean(X_not_chart[:, 0])
                    x3_std = np.std(X_not_chart[:, 0])
                    x4_mean = np.mean(X_not_chart[:, 1])
                    x4_std = np.std(X_not_chart[:, 1])
                    other_x = [h_mean * x3_std + x3_mean, w_mean * x4_std + x4_mean]
                
                cls._chart_plot_2d(trained_model, x_chart, 'y_true', 'y_pred', df_pair, x_chart_indices,
                                      x1_start, x1_end, x2_start, x2_end, other_x, chart_scale,
                                      proba_pred_col, proba_class_indices, ax, plot_border, plot_scatter,
                                      scatter_color_dict, scatter_marker_dict, proba_cmap_dict,  proba_type,
                                      contourf_kws=contourf_kws, imshow_kws=imshow_kws, scatter_kws=scatter_kws)

                # グラフタイトルとして、チャート非使用変数の範囲を記載（説明変数が3次元以上のとき）
                if x_num == 3:
                    if i == 0:
                        ax.set_title(f'{x_not_chart[0]}=- {cls._round_digits(h_max * x3_std + x3_mean, rounddigit=rounddigit_x3)} (- {h_max}σ)')
                    elif i == pair_h - 1:
                        ax.set_title(f'{x_not_chart[0]}={cls._round_digits(h_min * x3_std + x3_mean, rounddigit=rounddigit_x3)} - ({h_min}σ -)')
                    else:
                        ax.set_title(f'{x_not_chart[0]}={cls._round_digits(h_min * x3_std + x3_mean, rounddigit=rounddigit_x3)} - {cls._round_digits(h_max * x3_std + x3_mean, rounddigit=rounddigit_x3)} ({h_min}σ - {h_max}σ)')
                if x_num == 4:
                    ax.set_title(f'{x_not_chart[0]}= {h_min}σ - {h_max}σ  {x_not_chart[1]}= {w_min}σ - {w_max}σ')

        # 字が重なるのでtight_layoutにする
        plt.tight_layout()

    @classmethod
    def class_separator_plot(cls, model, x: List[str], y: str, data: pd.DataFrame, x_chart: List[str] = None,
                             pair_sigmarange = 1.5, pair_sigmainterval = 0.5, chart_extendsigma = 0.5, chart_scale = 1,
                             plot_scatter = 'class_error', rounddigit_x3 = 2,
                             scatter_colors = None, true_marker = 'o', false_marker = 'x',
                             cv=None, cv_seed=42, cv_group=None, display_cv_indices = 0,
                             model_params=None, fit_params=None, subplot_kws=None, contourf_kws=None, scatter_kws=None):
        """
        2～4次元説明変数の分類決定境界可視化

        Parameters
        ----------
        model:
            使用する回帰モデル(scikit-learn API)
        x: List[str]
            説明変数カラム (列名指定)
        y: str
            目的変数カラム (列名指定)
        data: pd.DataFrame
            フィッティング対象のデータ
        x_chart: List[str], optional
            説明変数のうち、決定境界図表示対象のカラム (Noneなら前から2カラム自動選択)
        pair_sigmarange: float, optional
            決定境界図非使用変数の分割範囲 (pair_sigmarange=2なら、-2σ~2σの範囲でpair_sigmaintervalに従い決定境界図分割)
        pair_sigmainterval: float, optional
            決定境界図非使用変数の1枚あたり表示範囲 (pair_sigmainterval=0.5なら、‥1σ~-0.5σ, 0.5σ~-0σ, 0σ~0.5σ, 0.5σ~1σ‥というようにヒートマップ分割)
        chart_extendsigma: float, optional
            決定境界図縦軸横軸の表示拡張範囲 (決定境界図使用変数の最大最小値 + extendsigmaが横軸範囲となる)
        chart_scale: int, optional
            決定境界図の表示倍率 (cv指定時等で表示が遅い時は大きくする)
        plot_scatter: str, optional
            散布図の描画種類('error':正誤で色分け, 'class':クラスで色分け, None:散布図表示なし)        
        rounddigit_x3: int, optional
            決定境界図非表示軸の小数丸め桁数
        scatter_colors: List[str], optional
            クラスごとのプロット色のリスト
        true_marker: str, optional
            正解クラスの散布図プロット形状
        false_marker: str, optional
            不正解クラスの散布図プロット形状
        cv: int or KFold, optional
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed: int, optional
            クロスバリデーションの乱数シード
        cv_group: str, optional
            GroupKFold、LeaveOneGroupOutのグルーピング対象カラム (列名指定)
        display_cv_indices: int, optional
            表示対象のクロスバリデーション番号 (指定したCV番号での回帰結果が表示される。リスト指定も可)
        model_params: Dict, optional
            回帰モデルに渡すパラメータ (チューニング後のパラメータがgood、Noneならデフォルト)
        fit_params: Dict, optional
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        subplot_kws: dict, optional
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        contourf_kws: dict, optional
            決定境界図用のax.contourf()に渡す引数
        scatter_kws: dict, optional
            散布図用のax.scatter()に渡す引数
        """
        # 説明変数xの次元が2～4以外ならエラーを出す
        if len(x) < 2 or len(x) > 4:
            raise Exception('length of x must be 2 to 4')
        
        # display_cv_indicesをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x, list):
            raise Exception('the "cv_display_num" argument must be int or List[int]')
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # contourf_kwsがNoneなら空のdictを入力
        if contourf_kws is None:
            contourf_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        
        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        else:
            raise Exception('the "x" argument must be str or str')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')
        
        # 決定境界図表示用の列を抽出
        if x_chart is None:  # 列名指定していないとき、前から2列を抽出
            x_chart = x[:2]
            x_chart_indices = [0, 1]
        else:  # 列名指定しているとき、該当列のXにおけるインデックス(0～3)を保持
            if len(x_chart) != 2:
                raise Exception('length of x_chart must be 2')
            x_chart_indices = []
            for colname in x_chart:
                x_chart_indices.append(x.index(colname))
        # 決定境界図表示以外の列
        x_not_chart = [colname for colname in x if colname not in x_chart]

        # クラス名と散布図色を紐づけ(色分けを全ての図で統一用)
        if scatter_colors is None:
            scatter_colors = cls.SCATTER_COLORS
        class_list = data[y].values.tolist()
        class_list = sorted(set(class_list), key=class_list.index)
        scatter_color_dict = dict(zip(class_list, scatter_colors[0:len(class_list)]))
        # 散布図マーカー形状をdict化
        scatter_marker_dict = {True: true_marker, False: false_marker}
        # contourf_kwsにcolors指定ないとき、scatter_color_dictの値を使用
        if 'colors' not in contourf_kws.keys():
            contourf_kws['colors'] = list(scatter_color_dict.values())
        # contourf_kwsにalphat指定ないとき、DEFAULT_SEPARATOR_ALPHAを使用
        if 'alpha' not in contourf_kws.keys():
            contourf_kws['alpha'] = cls.DEFAULT_SEPARATOR_ALPHA
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データから学習してプロット)
        if cv is None:
            # 学習と推論
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            # 決定境界図をプロット
            cls._class_chart_plot(model, X, y_pred, y_true, x_chart, x_not_chart, x_chart_indices,
                               pair_sigmarange = pair_sigmarange, pair_sigmainterval=pair_sigmainterval, chart_extendsigma=chart_extendsigma, chart_scale=chart_scale,
                               proba_pred = None, proba_class_indices = None, plot_border = True, plot_scatter = plot_scatter,
                               scatter_color_dict=scatter_color_dict, scatter_marker_dict=scatter_marker_dict, proba_cmap_dict=None, proba_type=None,
                               rounddigit_x3=rounddigit_x3,
                               cv_index=None, subplot_kws=subplot_kws, contourf_kws=contourf_kws, imshow_kws=None, scatter_kws=scatter_kws)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
            # GroupKFold、LeaveOneGroupOutのとき、y_trueをグルーピング対象に指定
            split_kws={}
            if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                if cv_group is not None:
                    split_kws['groups'] = data[cv_group].values
                else:
                    raise Exception('"GroupKFold" cross validation needs "cv_group" argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group].values))
            else:
                cv_num = cv.n_splits

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示対象以外のCVなら飛ばす
                if i not in display_cv_indices:
                    continue
                print(f'cv_number={i}/{cv_num}')
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                X_test = X[test]
                y_test = y_true[test]
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # 決定境界図をプロット
                cls._class_chart_plot(model, X_test, y_pred, y_test, x_chart, x_not_chart, x_chart_indices,
                                   pair_sigmarange = pair_sigmarange, pair_sigmainterval = pair_sigmainterval, chart_extendsigma=chart_extendsigma, chart_scale=chart_scale,
                                   proba_pred = None, proba_class_indices = None, plot_border = True, plot_scatter = plot_scatter,
                                   scatter_color_dict=scatter_color_dict, scatter_marker_dict=scatter_marker_dict, proba_cmap_dict=None, proba_type=None,
                                   rounddigit_x3=rounddigit_x3,
                                   cv_index=i, subplot_kws=subplot_kws, contourf_kws=contourf_kws, imshow_kws=None, scatter_kws=scatter_kws)

    @classmethod
    def class_proba_plot(cls, model, x: List[str], y: str, data: pd.DataFrame, x_chart: List[str] = None,
                         pair_sigmarange = 1.5, pair_sigmainterval = 0.5, chart_extendsigma = 0.5, chart_scale = 1,
                         plot_border = True, plot_scatter = 'class', rounddigit_x3 = 2,
                         proba_class = None, proba_cmap_dict = None, proba_type = 'contourf',
                         scatter_colors = None, true_marker = 'o', false_marker = 'x',
                         cv=None, cv_seed=42, cv_group=None, display_cv_indices = 0,
                         model_params=None, fit_params=None, subplot_kws=None, contourf_kws=None, imshow_kws=None, scatter_kws=None):
        """
        2～4次元説明変数のクラス確率図可視化

        Parameters
        ----------
        model:
            使用する回帰モデル(scikit-learn API)
        x: List[str]
            説明変数カラム (列名指定)
        y: str
            目的変数カラム (列名指定)
        data: pd.DataFrame
            フィッティング対象のデータ
        x_chart: List[str], optional
            説明変数のうち、クラス確率図表示対象のカラム (Noneなら前から2カラム自動選択)
        pair_sigmarange: float, optional
            クラス確率図非使用変数の分割範囲 (pair_sigmarange=2なら、-2σ~2σの範囲でpair_sigmaintervalに従い決定境界図分割)
        pair_sigmainterval: float, optional
            クラス確率図非使用変数の1枚あたり表示範囲 (pair_sigmainterval=0.5なら、‥1σ~-0.5σ, 0.5σ~-0σ, 0σ~0.5σ, 0.5σ~1σ‥というようにヒートマップ分割)
        chart_extendsigma: float, optional
            クラス確率図縦軸横軸の表示拡張範囲 (クラス確率図使用変数の最大最小値 + extendsigmaが横軸範囲となる)
        chart_scale: int, optional
            決定境界図の表示倍率 (cv指定時等で表示が遅い時は大きくする)
        plot_border: bool, optional
            クラス境界線の描画有無
        plot_scatter: str, optional
            散布図の描画種類('error':正誤で色分け, 'class':クラスで色分け, None:散布図表示なし)
        rounddigit_x3: int, optional
            クラス確率図非表示軸の小数丸め桁数
        proba_class: str or List[str], optional
            確率表示対象のクラス名
        proba_cmap_dict: dict[str, str], optional
            クラス確率図のカラーマップ (キーがクラス名、値がカラーマップのdictで指定)
        proba_type: str, optional
            クラス確率図の描画種類 ('contourf':ax.contourfでプロット, 'imshow':ax.imshowでプロット(3クラスまで))
        scatter_colors: List[str], optional
            クラスごとのプロット色のリスト
        true_marker: str, optional
            正解クラスの散布図プロット形状
        false_marker: str, optional
            不正解クラスの散布図プロット形状
        cv: int or KFold, optional
            クロスバリデーション分割法 (Noneのとき学習データから指標算出、int入力時はkFoldで分割)
        cv_seed: int, optional
            クロスバリデーションの乱数シード
        cv_group: str, optional
            GroupKFold、LeaveOneGroupOutのグルーピング対象カラム (列名指定)
        display_cv_indices: int, optional
            表示対象のクロスバリデーション番号 (指定したCV番号での回帰結果が表示される。リスト指定も可)
        model_params: Dict, optional
            回帰モデルに渡すパラメータ (チューニング後のパラメータがgood、Noneならデフォルト)
        fit_params: Dict, optional
            学習時のパラメータをdict指定 (例: XGBoostのearly_stopping_rounds)
            Noneならデフォルト
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        subplot_kws: dict, optional
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        contourf_kws: dict, optional
            クラス確率図(等高線)用のax.contourf()に渡す引数
        imshow_kws: dict, optional
            クラス確率図(RGB画像)用のax.imshow()に渡す引数
        scatter_kws: dict, optional
            散布図用のax.scatter()に渡す引数
        """
        # 説明変数xの次元が2～4以外ならエラーを出す
        if len(x) < 2 or len(x) > 4:
            raise Exception('length of x must be 2 to 4')
        
        # display_cv_indicesをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x, list):
            raise Exception('the "cv_display_num" argument must be int or List[int]')
        # 学習器パラメータがあれば適用
        if model_params is not None:
            model.set_params(**model_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # contourf_kwsがNoneなら空のdictを入力
        if contourf_kws is None:
            contourf_kws = {}
        # imshow_kwsがNoneなら空のdictを入力
        if imshow_kws is None:
            imshow_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}

        # xをndarray化
        if isinstance(x, list):
            X = data[x].values
        else:
            raise Exception('the "x" argument must be str or str')
        # yをndarray化
        if isinstance(y, str):
            y_true = data[y].values
        else:
            raise Exception('the "y" argument must be str')

        # クラス確率図表示用の列を抽出
        if x_chart is None:  # 列名指定していないとき、前から2列を抽出
            x_chart = x[:2]
            x_chart_indices = [0, 1]
        else:  # 列名指定しているとき、該当列のXにおけるインデックス(0～3)を保持
            if len(x_chart) != 2:
                raise Exception('length of x_chart must be 2')
            x_chart_indices = []
            for colname in x_chart:
                x_chart_indices.append(x.index(colname))
        # クラス確率図表示以外の列
        x_not_chart = [colname for colname in x if colname not in x_chart]

        # scatter_colors未指定のとき、デフォルト値を使用
        if scatter_colors is None:
            scatter_colors = cls.SCATTER_COLORS
        # クラス名と散布図色を紐づけ(色分けを全ての図で統一用)
        class_list = data[y].values.tolist()
        class_list = sorted(set(class_list), key=class_list.index)
        scatter_color_dict = dict(zip(class_list, scatter_colors[0:len(class_list)]))
        # 散布図マーカー形状をdict化
        scatter_marker_dict = {True: true_marker, False: false_marker}

        # proba_class未指定のとき、全てのクラスを使用
        if proba_class is None:
            proba_class = [c for c in class_list]
        # proba_classをList化
        if isinstance(proba_class, int) or isinstance(proba_class, str) or isinstance(proba_class, bool):
            proba_class = [proba_class]
        elif not isinstance(x, list):
            raise Exception('the "proba_class" argument must be int, str, bool or List')
        # List化したproba_classを走査してデータ上でのインデックスを取得
        proba_class_indices = []
        for pc in proba_class:
            if pc not in class_list:  # 指定したproba_classがデータ上に存在しないとき、エラーを出す
                raise Exception(f'"{proba_class}"" is not in the "{y}" column')
            proba_class_indices.append(class_list.index(pc))
        # proba_cmap_dict未指定のとき、デフォルト値を使用
        if proba_cmap_dict is None:
            proba_cmap_dict = dict(zip(proba_class,
                                       [cls.PROB_CMAP[pci] for pci in proba_class_indices]))
        # proba_cmap_dictがproba_classと一致していないとき、エラーを出す
        if list(proba_cmap_dict.keys()) != proba_class:
            raise Exception(f'the keys of the "proba_cmap_dict" argument must be equal to the argument "proba_class"')

        # contourf_kwsにalpha指定ないとき、DEFAULT_PROBA_CONTOURF_ALPHAを使用
        if 'alpha' not in contourf_kws.keys():
            contourf_kws['alpha'] = cls.DEFAULT_PROBA_CONTOURF_ALPHA
        # contourf_kwsにlevels指定ないとき、DDEFAULT_PROBA_CONTOURF_LEVELSを使用
        if 'levels' not in contourf_kws.keys():
            contourf_kws['levels'] = cls.DEFAULT_PROBA_CONTOURF_LEVELS
        # imshow_kwsにalpha指定ないとき、DEFAULT_PROBA_RGB_ALPHAを使用
        if 'alpha' not in imshow_kws.keys():
            imshow_kws['alpha'] = cls.DEFAULT_PROBA_RGB_ALPHA
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データから学習してプロット)
        if cv is None:
            # 学習と推論
            model.fit(X, y_true, **fit_params)
            y_pred = model.predict(X)
            # クラス確率を推定
            proba_pred = model.predict_proba(X)[:, proba_class_indices]
            # TODO:クラス確率の順番が逆転した時の対策をメソッドとして実装する必要あり
            # https://qiita.com/rawHam/items/3bcb6a68a533f2b82a85
            # クラス確率図をプロット
            cls._class_chart_plot(model, X, y_pred, y_true, x_chart, x_not_chart, x_chart_indices,
                               pair_sigmarange = pair_sigmarange, pair_sigmainterval=pair_sigmainterval, chart_extendsigma=chart_extendsigma, chart_scale=chart_scale,
                               proba_pred = proba_pred, proba_class_indices = proba_class_indices, plot_border = plot_border, plot_scatter = plot_scatter,
                               scatter_color_dict=scatter_color_dict, scatter_marker_dict=scatter_marker_dict, proba_cmap_dict=proba_cmap_dict, proba_type = proba_type,
                               rounddigit_x3=rounddigit_x3,
                               cv_index=None, subplot_kws=subplot_kws, contourf_kws=contourf_kws, imshow_kws=imshow_kws, scatter_kws=scatter_kws)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
            # GroupKFold、LeaveOneGroupOutのとき、y_trueをグルーピング対象に指定
            split_kws={}
            if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                if cv_group is not None:
                    split_kws['groups'] = data[cv_group].values
                else:
                    raise Exception('"GroupKFold" cross validation needs "cv_group" argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group].values))
            else:
                cv_num = cv.n_splits

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示対象以外のCVなら飛ばす
                if i not in display_cv_indices:
                    continue
                print(f'cv_number={i}/{cv_num}')
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                X_test = X[test]
                y_test = y_true[test]
                # proba_class_indicesを学習データから再取得（存在しなければ飛ばす）
                class_list_train = y_train.tolist()
                class_list_train = sorted(set(class_list_train), key=class_list_train.index)
                proba_class_indices = [class_list_train.index(pc) for pc in proba_class if pc in class_list_train]
                if len(proba_class_indices) == 0:
                    print(f'there is no assigned "proba_class" in the train data')
                    continue
                # proba_cmap_dictも学習データから再取得
                proba_cmap_dict = {k: v for k, v in proba_cmap_dict.items() if k in class_list_train}
                # 学習と推論
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_test)
                # クラス確率を推定
                proba_pred = model.predict_proba(X_test)[:, proba_class_indices]
                # クラス確率図をプロット
                cls._class_chart_plot(model, X_test, y_pred, y_test, x_chart, x_not_chart, x_chart_indices,
                                   pair_sigmarange = pair_sigmarange, pair_sigmainterval = pair_sigmainterval, chart_extendsigma=chart_extendsigma, chart_scale=chart_scale,
                                   proba_pred = proba_pred, proba_class_indices = proba_class_indices, plot_border = plot_border, plot_scatter = plot_scatter,
                                   scatter_color_dict=scatter_color_dict, scatter_marker_dict=scatter_marker_dict, proba_cmap_dict=proba_cmap_dict, proba_type = proba_type,
                                   rounddigit_x3=rounddigit_x3,
                                   cv_index=i, subplot_kws=subplot_kws, contourf_kws=contourf_kws, imshow_kws=imshow_kws, scatter_kws=scatter_kws)
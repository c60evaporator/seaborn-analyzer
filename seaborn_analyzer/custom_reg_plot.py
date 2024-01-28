from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
import decimal

from ._cv_eval_set import _make_transformer, _eval_set_selection, cross_val_score_eval_set

class regplot():
    # regression_heat_plotメソッド (回帰モデルヒートマップ表示)における、散布図カラーマップ
    _HEAT_SCATTER_HUECOLORS = ['red', 'mediumblue', 'darkorange', 'darkmagenta', 'cyan',  'pink', 'brown', 'gold', 'grey']

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
        指定桁数でdictの値を丸める

        Parameters
        ----------
        srcdict : dict[str, float]
            丸め対象のdict
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
            elif scoring == 'mse':
                score_dict['mse'] = mean_squared_error(y_true, y_pred, squared=True)
            elif scoring == 'rmse':
                score_dict['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
            elif scoring == 'rmsle':
                score_dict['rmsle'] = mean_squared_log_error(y_true, y_pred)
            elif scoring == 'mape':
                score_dict['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            elif scoring == 'max_error':
                score_dict['max_error'] = max([abs(p - r) for r, p in zip(y_true, y_pred)])
        return score_dict

    def _reshape_input_data(x, y, data, x_colnames, cv_group):
        """
        入力データの形式統一(pd.DataFrame or np.ndarray)
        """
        # dataがpd.DataFrameのとき
        if isinstance(data, pd.DataFrame):
            if not isinstance(x, list):
                raise Exception('`x` argument should be list[str] if `data` is pd.DataFrame')
            if not isinstance(y, str):
                raise Exception('`y` argument should be str if `data` is pd.DataFrame')
            if x_colnames is not None:
                raise Exception('`x_colnames` argument should be None if `data` is pd.DataFrame')
            X = data[x].values
            y_true = data[y].values
            x_colnames = x
            y_colname = y
            cv_group_colname = cv_group
            
        # dataがNoneのとき(x, y, cv_groupがnp.ndarray)
        elif data is None:
            if not isinstance(x, np.ndarray):
                raise Exception('`x` argument should be np.ndarray if `data` is None')
            if not isinstance(y, np.ndarray):
                raise Exception('`y` argument should be np.ndarray if `data` is None')
            X = x if len(x.shape) == 2 else x.reshape([x.shape[0], 1])
            y_true = y.ravel()
            # x_colnameとXの整合性確認
            if x_colnames is None:
                x_colnames = list(range(X.shape[1]))
            elif X.shape[1] != len(x_colnames):
                raise Exception('width of X must be equal to length of x_colnames')
            else:
                x_colnames = x_colnames
            y_colname = 'objective_variable'
            if cv_group is not None:  # cv_group指定時
                cv_group_colname = 'group'
                data = pd.DataFrame(np.column_stack((X, y_true, cv_group)),
                                    columns=x_colnames + [y_colname] + [cv_group_colname])
            else:
                cv_group_colname = None
                data = pd.DataFrame(np.column_stack((X, y)),
                                    columns=x_colnames + [y_colname])
        else:
            raise Exception('`data` argument should be pd.DataFrame or None')

        return X, y_true, data, x_colnames, y_colname, cv_group_colname

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
        ax : matplotlib.axes.Axes
            表示対象のax（Noneならmatplotlib.pyplot.plotで1枚ごとにプロット）
        rounddigit: int
            表示指標の小数丸め桁数
        """
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()

        if rank_col is None:
            rank_col = 'index'
        y_error = y_pred - y_true
        y_error_abs = np.abs(y_error)
        rank_index  = np.argsort(-y_error_abs)[:rank_number]
        for rank, i in enumerate(rank_index):
            error = cls._round_digits(y_error[i], rounddigit=rounddigit, method='decimal')
            rank_text = f'      no{rank+1}\n-<-error={error}\n      {rank_col}={rank_col_data[i]}'
            if x is None:  # 横軸y_true縦軸y_pred (regression_pred_trueメソッド用)
                ax.text(y_true[i], y_pred[i], rank_text, verticalalignment='center', horizontalalignment='left')
            else:  # 横軸x縦軸y_true (regression_plot_1dメソッド用)
                ax.text(x[i], y_true[i], rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def _scatterplot_ndarray(cls, x, x_name, y, y_name, hue_data, hue_name, ax, scatter_kws, legend_kws):
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
        sns.scatterplot(x=x_name, y=y_name, data=data, ax=ax, hue=hue_name, **scatter_kws)
        # 凡例追加
        if 'title' not in legend_kws.keys():
            legend_kws['title'] = hue_name 
        ax.legend(**legend_kws)

    @classmethod
    def _plot_pred_true(cls, y_true, y_pred, hue_data=None, hue_name=None, ax=None,
                        linecolor='red', linesplit=200, rounddigit=None,
                        score_dict=None, scatter_kws=None, legend_kws=None):
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
        ax : matplotlib.axes.Axes
            表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)
        linecolor : str
            予測値=実測値の線の色
        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)
        rounddigit: int
            表示指標の小数丸め桁数
        score_dict : dict[str, float]
            算出した評価指標一覧
        scatter_kws : dict
            Additional parameters passed to sns.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        """
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()
        # score_dictがNoneのとき、空のDictを加瀬宇
        if score_dict is None:
            score_dict = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        
        # 散布図プロット
        cls._scatterplot_ndarray(y_true, 'y_true', y_pred, 'y_pred', hue_data, hue_name, ax, scatter_kws, legend_kws)

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
    def regression_pred_true(cls, estimator, x: List[str], y: str, data: pd.DataFrame = None,
                             x_colnames: List[str] = None, hue=None, linecolor='red', rounddigit=3,
                             rank_number=None, rank_col=None, scores='mae', 
                             cv_stats='mean', cv=None, cv_seed=42, cv_group=None, ax=None,
                             estimator_params=None, fit_params=None, validation_fraction=None,
                             subplot_kws=None, scatter_kws=None, legend_kws=None):

        """
        Plot prediction vs. true scatter plots of any scikit-learn regression estimator

        Parameters
        ----------
        estimator : estimator object implementing ``fit``
            Regression estimator. This is assumed to implement the scikit-learn estimator interface.

        x : str or list[str]
            Explanatory variables.

        y : str
            Objective variable.

        data : pd.DataFrame
            Input data structure.

        x_colnames: list[str], optional
            Names of explanatory variables. Available only if ``data`` is NOT pd.DataFrame

        hue : str, optional
            Grouping variable that will produce points with different colors.

        linecolor : str, optional
            Color of prediction = true line. See https://matplotlib.org/stable/gallery/color/named_colors.html

        rounddigit: int, optional
            Round a number of score to a given precision in decimal digits.

        rank_number : int, optional
            Number of emphasized data that are in the top posiotions for regression error.

        rank_col : list[str], optional
            Variables that are displayed with emphasized data that are in the top posiotions for regression error.

        scores : {'r2', 'mae', 'mse', 'rmse', 'rmsle', 'mape', 'max_error'} or list, optional
            Regression score that are displayed at the lower right of the graph.

        cv_stats : {'mean', 'median', 'max', 'min'}, optional
            Statistical method of cross validation score that are displayed at the lower right of the graph.

        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. If None, no cross-validation is used and the training data is displayed. If int, to specify the number of folds in a KFold.

        cv_seed : int, optional
            Seed for random number generator of cross validation.

        cv_group: str, optional
            Group variable for the samples used while splitting the dataset into train/test set. This argument is passed to ``groups`` argument of cv.split().

        ax : {matplotlib.axes.Axes, list[matplotlib.axes.Axes]}, optional
            Pre-existing axes for the plot or list of it. Otherwise, call matplotlib.pyplot.subplot() internally.

        estimator_params : dict, optional
            Parameters passed to the regression estimator. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        fit_params : dict, optional
            Parameters passed to the fit() method of the regression estimator, e.g. ``early_stopping_round`` and ``eval_set`` of XGBRegressor. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        subplot_kws : dict, optional
            Additional parameters passed to matplotlib.pyplot.subplots(), e.g. figsize. Available only if ``axes`` is None. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

        scatter_kws: dict, optional
            Additional parameters passed to sns.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html

        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html

        Returns
        ----------
        score_dict : dict
            Validation scores, e.g. r2, mae and rmse
        """

        # 入力データの形式統一
        X, y_true, data, x_colnames, y_colname, cv_group_colname = cls._reshape_input_data([x] if isinstance(x, str) else x, 
                                                                                           y, data,
                                                                                           x_colnames,
                                                                                           cv_group)
        # scoresの型をListに統一
        if scores is None:
            scores = []
        elif isinstance(scores, str):
            scores = [scores]
        elif not isinstance(scores, list):
            raise Exception('the "scores" argument must be str or list[str]')
        # 学習器パラメータがあれば適用
        if estimator_params is not None:
            estimator.set_params(**estimator_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            estimator.fit(X, y_true, **fit_params)
            y_pred = estimator.predict(X)
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
            cls._plot_pred_true(y_true, y_pred, hue_data=hue_data, hue_name=hue_name, ax=ax,
                                linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict,
                                scatter_kws=scatter_kws, legend_kws=legend_kws)
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
            # cv_groupをグルーピング対象に指定(GroupKFold、LeaveOneGroupOut等)
            split_kws={}
            if cv_group_colname is not None:
                split_kws['groups'] = data[cv_group_colname].values
            elif isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need ``cv_group`` argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group_colname].values))
            elif isLeaveOneOut:
                cv_num = 1
            else:
                cv_num = cv.n_splits

            # fit_paramsにeval_metricが入力されており、eval_setが入力されていないときの処理(eval_setにテストデータを使用)
            if validation_fraction is None:
                validation_fraction = 'cv'
            # 最終学習器以外の前処理変換器作成
            transformer = _make_transformer(validation_fraction, estimator)

            # スコア種類ごとにクロスバリデーションスコアの算出
            score_all_dict = {}
            for scoring in scores:
                # cross_val_scoreでクロスバリデーション
                if scoring == 'r2':
                    score_all_dict['r2'] = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='r2',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                elif scoring == 'mae':
                    neg_mae = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_absolute_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mae'] = -neg_mae  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'mse':
                    neg_mse = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_squared_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mse'] = -neg_mse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmse':
                    neg_rmse = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_root_mean_squared_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmse'] = -neg_rmse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmsle':
                    neg_msle = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_squared_log_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmsle'] = np.sqrt(-neg_msle)  # 正負を逆にしてルートをとる
                elif scoring == 'mape':
                    neg_mape = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_absolute_percentage_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mape'] = -neg_mape  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'max_error':
                    neg_max_error = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='max_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['max_error'] = - neg_max_error  # scikit-learnの仕様に合わせ正負を逆に
            
            # 表示用のax作成
            if ax is None:
                # LeaveOneOutのとき、クロスバリデーションごとの図は作成せず
                if isLeaveOneOut:
                    if 'figsize' not in subplot_kws.keys():
                        subplot_kws['figsize'] = (6, 6)
                    fig, ax = plt.subplots(1, 1, **subplot_kws)
                # LeaveOneOut以外のとき、クロスバリデーションごとに図作成
                else:
                    if 'figsize' not in subplot_kws.keys():
                        subplot_kws['figsize'] = (6, (cv_num + 1) * 6)
                    fig, ax = plt.subplots(cv_num + 1, 1, **subplot_kws)

            # クロスバリデーション
            y_true_all = []
            y_pred_all = []
            hue_all = []
            rank_col_all = []
            score_train_dict = {}
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

                # random_state取得
                if isinstance(estimator, Pipeline):
                    random_state = estimator.steps[-1][1].random_state if 'random_state' in estimator.steps[-1][1].__dict__.keys() else None
                else:
                    random_state = estimator.random_state if 'random_state' in estimator.__dict__.keys() else None

                # eval_setの中から学習データ or テストデータのみを抽出
                fit_params_modified, train_divided = _eval_set_selection(
                    validation_fraction, 
                    transformer, 
                    X,
                    y,
                    fit_params, 
                    train, 
                    test,
                    random_state
                    )

                # 学習と推論
                estimator.fit(X[train_divided], y_true[train_divided], **fit_params_modified)
                y_pred = estimator.predict(X_test)
                # 学習データスコア算出
                y_pred_train = estimator.predict(X[train_divided])
                score_dict = cls._make_score_dict(y_true[train_divided], y_pred_train, scores)
                for score in scores:
                    if f'{score}_train' not in score_train_dict:
                        score_train_dict[f'{score}_train'] = []
                    score_train_dict[f'{score}_train'].append(score_dict[score])
                # CV内結果をプロット(LeaveOneOutのときはプロットしない)
                if not isLeaveOneOut:
                    score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                    score_cv_dict.update({f'{k}_train': v for k, v in score_dict.items()})
                    cls._plot_pred_true(y_test, y_pred, hue_data=hue_test, hue_name=hue_name, ax=ax[i],
                                        linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict,
                                        scatter_kws=scatter_kws, legend_kws=legend_kws)
                    ax[i].set_title(f'Cross Validation Fold{i}')
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
            # スコアの統計値を計算
            if cv_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.mean(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.median(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.amin(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.amax(v) for k, v in score_train_dict.items()}
            # 学習データスコアをdictに追加
            score_stats_dict.update(train_stats_dict)
            # 全体プロット
            ax_all = ax if isLeaveOneOut else ax[cv_num]
            cls._plot_pred_true(y_true_all, y_pred_all, hue_data=hue_all, hue_name=hue_name, ax=ax_all,
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict,
                               scatter_kws=scatter_kws, legend_kws=legend_kws)
            ax_all.set_title('All Cross Validations')
            # 誤差上位を文字表示
            if rank_number is not None:
                cls._rank_display(y_true_all, y_pred_all, rank_number, rank_col, rank_col_all,
                                  ax=ax_all, rounddigit=rounddigit)
            return score_stats_dict
    
    def _average_plot(estimator, data, x_colnames, y_colname, hue,
                      aggregate, subplot_kws, plot_kws, scatter_kws, legend_kws,
                      cv_index, x_range=200):
        # figsize (全ての図全体のサイズ)指定
        if 'figsize' not in subplot_kws.keys():
            subplot_kws['figsize'] = (6, len(x_colnames) * 5)
        if 'color' not in plot_kws:
            plot_kws['color'] = 'red'
        # プロット用のaxes作成
        fig, axes = plt.subplots(len(x_colnames), 1, **subplot_kws)
        if cv_index is not None:
            fig.suptitle(f'CV No.{cv_index}')
        # 全列を走査
        for i, colname in enumerate(x_colnames):
            # 該当列（グラフのX軸）の値を作成
            x_max = data[colname].max()
            x_min = data[colname].min()
            x_array = np.linspace(x_min, x_max, x_range)
            # 該当列以外を抽出して平均値算出
            if aggregate == 'mean':
                other_x_agg = data[[col for col in x_colnames if col != colname]].mean()
            elif aggregate == 'median':
                other_x_agg = data[[col for col in x_colnames if col != colname]].median()
            else:
                raise ValueError('the `aggregate` argument should be "mean" or "median"')
            X_mean = np.tile(other_x_agg, (x_range, 1))
            # 該当列を挿入して説明変数とし、モデルで推論
            X_mean = np.insert(X_mean, i, x_array, axis=1)
            y_pred = estimator.predict(X_mean)
            # 実測値を散布図プロット
            ax = axes if len(x_colnames) == 1 else axes[i]
            sns.scatterplot(x=colname, y=y_colname, hue=hue, data=data, ax=ax, **scatter_kws)
            # 推測値曲線をプロット
            ax.plot(x_array, y_pred, **plot_kws)
            # 色分け時は凡例表示
            if hue is not None:
                ax.legend(**legend_kws)

        fig.tight_layout(rect=[0, 0, 1, 0.98])          


    @classmethod
    def average_plot(cls, estimator, x: List[str], y: str, data: pd.DataFrame = None,
                     x_colnames: List[str] = None, hue=None,
                     aggregate='mean',
                     cv=None, cv_seed=42, cv_group=None, display_cv_indices = 0,
                     estimator_params=None, fit_params=None, validation_fraction=None,
                     subplot_kws=None, plot_kws=None, scatter_kws=None, legend_kws=None):
        """
        Plot relationship between one explanatory variable and predicted value by line graph.

        Other explanatory variables are fixed to aggregated values such as mean values or median values.

        Parameters
        ----------
        estimator : estimator object implementing ``fit``
            Regression estimator. This is assumed to implement the scikit-learn estimator interface.
        x : list[str] or np.ndarray
            Explanatory variables. Should be list[str] if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        y : str or np.ndarray
            Objective variable. Should be str if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        data: pd.DataFrame
            Input data structure.

        x_colnames: list[str], optional
            Names of explanatory variables. Available only if ``data`` is NOT pd.DataFrame

        hue : str, optional
            Grouping variable that will produce points with different colors.

        aggregate : {'mean', 'median'}, optional
            Statistic method of aggregating explanatory variables except x_axis variable.

        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. If None, no cross-validation is used and the training data is displayed. If int, to specify the number of folds in a KFold.

        cv_seed : int, optional
            Seed for random number generator of cross validation.

        cv_group: str, optional
            Group variable for the samples used while splitting the dataset into train/test set. This argument is passed to ``groups`` argument of cv.split().

        display_cv_indices : int or list, optional
            Cross validation index or indices to display.

        estimator_params : dict, optional
            Parameters passed to the regression estimator. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        fit_params : dict, optional
            Parameters passed to the fit() method of the regression estimator, e.g. ``early_stopping_round`` and ``eval_set`` of XGBRegressor. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.
        
        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        subplot_kws: dict, optional
            Additional parameters passed to matplotlib.pyplot.subplots(), e.g. ``figsize``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

        plot_kws: dict, optional
            Additional parameters passed to matplotlib.axes.Axes.plot(), e.g. ``alpha``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

        scatter_kws: dict, optional
            Additional parameters passed to seaborn.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html

        legend_kws : dict
            Additional parameters passed to matplotlib.axes.Axes.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        """

        # 入力データの形式統一
        X, y_true, data, x_colnames, y_colname, cv_group_colname = cls._reshape_input_data(x, y, data,
                                                                                           x_colnames,
                                                                                           cv_group)
        
        # display_cv_indicesをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x_colnames, list):
            raise Exception('the "cv_display_indices" argument should be int or List[int]')
        # 学習器パラメータがあれば適用
        if estimator_params is not None:
            estimator.set_params(**estimator_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # plot_kwsがNoneなら空のdictを入力
        if plot_kws is None:
            plot_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            estimator.fit(X, y_true, **fit_params)
            # 平均値
            cls._average_plot(estimator, data, x_colnames, y_colname, hue,
                              aggregate=aggregate,
                              subplot_kws=subplot_kws, plot_kws=plot_kws,
                              scatter_kws=scatter_kws, legend_kws=legend_kws,
                              cv_index=None)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
            # cv_groupをグルーピング対象に指定(GroupKFold、LeaveOneGroupOut等)
            split_kws={}
            if cv_group_colname is not None:
                split_kws['groups'] = data[cv_group_colname].values
            elif isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need ``cv_group`` argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group_colname].values))
            else:
                cv_num = cv.n_splits

            # fit_paramsにeval_metricが入力されており、eval_setが入力されていないときの処理(eval_setにテストデータを使用)
            if validation_fraction is None:
                validation_fraction = 'cv'
            # 最終学習器以外の前処理変換器作成
            transformer = _make_transformer(validation_fraction, estimator)

            # クロスバリデーション
            for i, (train, test) in enumerate(cv.split(X, y_true, **split_kws)):
                # 表示対象以外のCVなら飛ばす
                if i not in display_cv_indices:
                    continue
                print(f'cv_number={i}/{cv_num}')
                # 表示用にテストデータと学習データ分割
                X_train = X[train]
                y_train = y_true[train]
                data_test = data.iloc[test]
                
                # eval_setの中から学習データ or テストデータのみを抽出
                fit_params_modified, train_divided = _eval_set_selection(
                    validation_fraction, 
                    transformer, 
                    X,
                    y,
                    fit_params, 
                    train, 
                    test,
                    estimator.steps[-1][1].random_state if isinstance(estimator, Pipeline) else estimator.random_state
                    )

                # 学習と推論
                estimator.fit(X[train_divided], y_true[train_divided], **fit_params_modified)
                # ヒートマップをプロット
                cls._average_plot(estimator, data_test, x_colnames, y_colname, hue,
                                  aggregate=aggregate,
                                  subplot_kws=subplot_kws, plot_kws=plot_kws,
                                  scatter_kws=scatter_kws, legend_kws=legend_kws,
                                  cv_index=i)

        
    @classmethod
    def linear_plot(cls, x: str, y: str, data: pd.DataFrame = None,
                    x_colname: str = None,
                    ax=None, hue=None, linecolor='red',
                    rounddigit=5, plot_scores=True, scatter_kws=None, legend_kws=None):
        """
        Plot linear regression line and calculate Pearson correlation coefficient.

        Parameters
        ----------
        x : str
            Variable that specify positions on the x.

        y : str
            Variable that specify positions on the y.

        data : pd.DataFrame
            Input data structure.

        x_colname: str, optional
            Names of explanatory variable. Available only if ``data`` is NOT pd.DataFrame

        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot. Otherwise, call matplotlib.pyplot.gca() internally.

        hue : str, optional
            Grouping variable that will produce points with different colors.

        linecolor : str, optional
            Color of regression line. See https://matplotlib.org/stable/gallery/color/named_colors.html

        rounddigit: int, optional
            Round a number of score to a given precision in decimal digits.

        plot_scores: bool, optional
            If True, display Pearson correlation coefficient and the p-value.

        scatter_kws: dict, optional
            Additional parameters passed to sns.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html

        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html

        Returns
        ----------
        ax : matplotlib.axes.Axes
            Returns the Axes object with the plot drawn onto it.
        """

        # 入力データの形式統一
        X, y_true, data, x_colnames, y_colname, cv_group_colname = cls._reshape_input_data([x] if isinstance(x, str) else x, 
                                                                                           y, data,
                                                                                           [x_colname] if x_colname is not None else x_colname,
                                                                                           cv_group=None)
        if x_colname is None:
            x_colname = x_colnames[0]
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}

        # まずは散布図プロット
        ax = sns.scatterplot(x=x_colname, y=y_colname, data=data, ax=ax, hue=hue, **scatter_kws)
        # 凡例追加
        if hue is not None and 'title' not in legend_kws.keys():
            legend_kws['title'] = hue
        if len(legend_kws.keys()) > 0:
            ax.legend(**legend_kws)

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
            pearsonr = stats.pearsonr(data[x_colname], data[y_colname])
            r = cls._round_digits(pearsonr[0], rounddigit=rounddigit, method="decimal")
            pvalue = cls._round_digits(pearsonr[1], rounddigit=rounddigit, method="decimal")            
            # プロット
            rtext = f'{equation}\nr={r}\np={pvalue}'
            ax.text(xmax, np.amin(y_true), rtext, verticalalignment='bottom', horizontalalignment='right')

        return ax

    @classmethod
    def _estimator_plot_1d(cls, trained_estimator, X, y_true, hue_data=None, hue_name=None, ax=None, linecolor='red', linesplit=1000, rounddigit=None,
                       score_dict=None, scatter_kws=None, legend_kws=None):
        """
        1次説明変数回帰曲線を、回帰評価指標とともにプロット

        Parameters
        ----------
        trained_estimator : 
            学習済の回帰モデル(scikit-learn API)

        X : ndarray
            説明変数

        y_true : ndarray
            目的変数実測値

        hue_data : ndarray
            色分け用ラベルデータ

        hue_name : str
            色分け用の列名

        ax : matplotlib.axes.Axes
            表示対象のax (Noneならplt.plotで1枚ごとにプロット)

        linecolor : str
            予測値=実測値の線の色

        linesplit : int
            フィッティング線の分割数 (カクカクしたら増やす)

        rounddigit: int
            表示指標の小数丸め桁数

        score_dict : dict[str, float]
            算出した評価指標一覧

        scatter_kws: dict, optional
            Additional parameters passed to sns.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html

        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        """
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()
        # score_dictがNoneのとき、空のDictを入力
        if score_dict is None:
            score_dict = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}
        
        # 散布図プロット
        cls._scatterplot_ndarray(np.ravel(X), 'X', y_true, 'Y', hue_data, hue_name, ax, scatter_kws, legend_kws)

        # 回帰モデルの線を作成
        xmin = np.amin(X)
        xmax = np.amax(X)
        Xline = np.linspace(xmin, xmax, linesplit)
        Xline = Xline.reshape(len(Xline), 1)
        # 回帰線を描画
        ax.plot(Xline, trained_estimator.predict(Xline), color=linecolor)
        
        # 評価指標文字列作成
        score_list = [f'{k}={v}' for k, v in cls._round_dict_digits(score_dict, rounddigit, 'sig').items()]
        score_text = "\n".join(score_list)
        ax.text(xmax, np.amin(y_true), score_text, verticalalignment='bottom', horizontalalignment='right')

    @classmethod
    def regression_plot_1d(cls, estimator, x: str, y: str, data: pd.DataFrame = None, x_colname: str = None,
                           hue=None, linecolor='red', rounddigit=3,
                           rank_number=None, rank_col=None, scores='mae',
                           cv_stats='mean', cv=None, cv_seed=42, cv_group=None,
                           estimator_params=None, fit_params=None, validation_fraction=None,
                           subplot_kws=None, scatter_kws=None, legend_kws=None):
        """
        Plot regression lines of any scikit-learn regressor with 1D explanatory variable.

        Parameters
        ----------
        estimator : estimator object implementing ``fit``
            Regression estimator. This is assumed to implement the scikit-learn estimator interface.

        x : str, or np.ndarray
            Explanatory variables. Should be str if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        y : str or np.ndarray
            Objective variable. Should be str if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        data: pd.DataFrame
            Input data structure.

        x_colname: str, optional
            Names of explanatory variable. Available only if ``data`` is NOT pd.DataFrame

        hue : str, optional
            Grouping variable that will produce points with different colors.

        linecolor : str, optional
            Color of prediction = true line. See https://matplotlib.org/stable/gallery/color/named_colors.html

        rounddigit: int, optional
            Round a number of score to a given precision in decimal digits.

        rank_number : int, optional
            Number of emphasized data that are in the top positions for regression error.

        rank_col : list[str], optional
            Variables that are displayed with emphasized data that are in the top posiotions for regression error.

        scores : {'r2', 'mae', 'mse', 'rmse', 'rmsle', 'mape', 'max_error'} or list,, optional
            Regression score that are displayed at the lower right of the graph.

        cv_stats : {'mean', 'median', 'max', 'min'}, optional
            Statistical method of cross validation score that are displayed at the lower right of the graph.

        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. If None, no cross-validation is used and the training data is displayed. If int, to specify the number of folds in a KFold.

        cv_seed : int, optional
            Seed for random number generator of cross validation.

        cv_group: str, optional
            Group variable for the samples used while splitting the dataset into train/test set. This argument is passed to ``groups`` argument of cv.split().

        estimator_params : dict, optional
            Parameters passed to the regression estimator. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        fit_params : dict, optional
            Parameters passed to the fit() method of the regression estimator, e.g. ``early_stopping_round`` and ``eval_set`` of XGBRegressor. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        subplot_kws : dict, optional
            Additional parameters passed to matplotlib.pyplot.subplots(), e.g. ``figsize``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        scatter_kws: dict, optional
            Additional parameters passed to sns.scatterplot(), e.g. ``alpha``. See https://seaborn.pydata.org/generated/seaborn.scatterplot.html

        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html

        Returns
        ----------
        score_dict : dict
            Validation scores, e.g. r2, mae and rmse
        """

        # 入力データの形式統一
        X, y_true, data, x_colnames, y_colname, cv_group_colname = cls._reshape_input_data([x] if isinstance(x, str) else x,
                                                                                           y, data,
                                                                                           [x_colname] if x_colname is not None else x_colname,
                                                                                           cv_group)
        # scoresの型をListに統一
        if scores is None:
            scores = []
        elif isinstance(scores, str):
            scores = [scores]
        elif not isinstance(scores, list):
            raise Exception('the "scores" argument must be str or list[str]')
        # 学習器パラメータがあれば適用
        if estimator_params is not None:
            estimator.set_params(**estimator_params)
        # 学習時パラメータがNoneなら空のdictを入力
        if fit_params is None:
            fit_params = {}
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            estimator.fit(X, y_true, **fit_params)
            y_pred = estimator.predict(X)
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
            cls._estimator_plot_1d(estimator, X, y_true, hue_data=hue_data, hue_name=hue_name,
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_dict,
                               scatter_kws=scatter_kws, legend_kws=legend_kws)
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
            # cv_groupをグルーピング対象に指定(GroupKFold、LeaveOneGroupOut等)
            split_kws={}
            if cv_group_colname is not None:
                split_kws['groups'] = data[cv_group_colname].values
            elif isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need ``cv_group`` argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group_colname].values))
            else:
                cv_num = cv.n_splits

            # fit_paramsにeval_metricが入力されており、eval_setが入力されていないときの処理(eval_setにテストデータを使用)
            if validation_fraction is None:
                validation_fraction = 'cv'
            # 最終学習器以外の前処理変換器作成
            transformer = _make_transformer(validation_fraction, estimator)

            # スコア種類ごとにクロスバリデーションスコアの算出
            score_all_dict = {}
            for scoring in scores:
                # cross_val_scoreでクロスバリデーション
                if scoring == 'r2':
                    score_all_dict['r2'] = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='r2',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                elif scoring == 'mae':
                    neg_mae = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_absolute_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mae'] = -neg_mae  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'mse':
                    neg_mse = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_squared_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mse'] = -neg_mse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmse':
                    neg_rmse = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_root_mean_squared_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmse'] = -neg_rmse  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'rmsle':
                    neg_msle = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_squared_log_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['rmsle'] = np.sqrt(-neg_msle)  # 正負を逆にしてルートをとる
                elif scoring == 'mape':
                    neg_mape = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='neg_mean_absolute_percentage_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['mape'] = -neg_mape  # scikit-learnの仕様に合わせ正負を逆に
                elif scoring == 'max_error':
                    neg_max_error = cross_val_score_eval_set(estimator, X, y_true, validation_fraction,
                                                    cv=cv, scoring='max_error',
                                                    fit_params=fit_params, n_jobs=-1, **split_kws)
                    score_all_dict['max_error'] = - neg_max_error  # scikit-learnの仕様に合わせ正負を逆に
            
            # 表示用のaxes作成
            # クロスバリデーションごとに図作成
            if 'figsize' not in subplot_kws.keys():
                subplot_kws['figsize'] = (6, (cv_num + 1) * 6)
            fig, axes = plt.subplots(cv_num + 1, 1, **subplot_kws)

            # クロスバリデーション
            score_train_dict = {}
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
                
                # eval_setの中から学習データ or テストデータのみを抽出
                fit_params_modified, train_divided = _eval_set_selection(
                    validation_fraction, 
                    transformer, 
                    X,
                    y,
                    fit_params, 
                    train, 
                    test,
                    estimator.steps[-1][1].random_state if isinstance(estimator, Pipeline) else estimator.random_state
                    )

                # 学習と推論
                estimator.fit(X[train_divided], y_true[train_divided], **fit_params_modified)
                # 学習データスコア算出
                y_pred_train = estimator.predict(X[train_divided])
                score_dict = cls._make_score_dict(y_true[train_divided], y_pred_train, scores)
                for score in scores:
                    if f'{score}_train' not in score_train_dict:
                        score_train_dict[f'{score}_train'] = []
                    score_train_dict[f'{score}_train'].append(score_dict[score])
                # CV内結果をプロット
                score_cv_dict = {k: v[i] for k, v in score_all_dict.items()}
                score_cv_dict.update({f'{k}_train': v for k, v in score_dict.items()})
                cls._estimator_plot_1d(estimator, X_test, y_test, hue_data=hue_test, hue_name=hue_name, ax=axes[i],
                                   linecolor=linecolor, rounddigit=rounddigit, score_dict=score_cv_dict,
                                   scatter_kws=scatter_kws, legend_kws=legend_kws)
                # 誤差上位を文字表示
                if rank_number is not None:
                    cls._rank_display(y_test, estimator.predict(X_test), rank_number, rank_col, rank_col_test, x=X_test, ax=axes[i], rounddigit=rounddigit)
                axes[i].set_title(f'Cross Validation Fold{i}')

            # スコアの統計値を計算
            if cv_stats == 'mean':
                score_stats_dict = {f'{k}_mean': np.mean(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.mean(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'median':
                score_stats_dict = {f'{k}_median': np.median(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.median(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'min':
                score_stats_dict = {f'{k}_min': np.amin(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.amin(v) for k, v in score_train_dict.items()}
            elif cv_stats == 'max':
                score_stats_dict = {f'{k}_max': np.amax(v) for k, v in score_all_dict.items()}
                train_stats_dict = {k: np.amax(v) for k, v in score_train_dict.items()}
            # 学習データスコアをdictに追加
            score_stats_dict.update(train_stats_dict)
            # 全体色分け用データ取得
            hue_data = None if hue is None else data[hue]
            hue_name = None if hue is None else hue
            # 全体プロット
            ax_all = axes[cv_num]
            cls._estimator_plot_1d(estimator, X, y_true, hue_data=hue_data, hue_name=hue_name, ax=ax_all,
                               linecolor=linecolor, rounddigit=rounddigit, score_dict=score_stats_dict,
                               scatter_kws=scatter_kws, legend_kws=legend_kws)
            ax_all.set_title('All Cross Validations')
            return score_stats_dict

    @classmethod
    def _reg_heat_plot_2d(cls, trained_estimator, x_heat, y_true_col, y_pred_col, rank_col, data, x_heat_indices, hue_name,
                          x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                          vmin, vmax, ax, plot_scatter, maxerror, rank_dict, scatter_hue_dict,
                          rounddigit_rank, rounddigit_x1, rounddigit_x2,
                          heat_kws=None, scatter_kws=None, legend_kws=None):
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
        y_pred_grid = trained_estimator.predict(X_all)
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
                ax.legend(**legend_kws)
        
        # 誤差上位を文字表示
        df_rank = data[data.index.isin(rank_dict.keys())]
        for index, row in df_rank.iterrows():
            # rank_col指定ないとき、indexがfloat型に変換されてしまうので、int型に戻す
            rank_col_value = int(row[rank_col]) if rank_col == 'index' else row[rank_col]
            # 誤差を計算してテキスト化
            error = cls._round_digits(row['y_pred'] - row['y_true'], rounddigit=rounddigit_rank)
            rank_text = f'     no{rank_dict[index]+1}\n-<-error={error}\n     {rank_col}={rank_col_value}'
            # 軸範囲が0～heat_divisionになっているので、スケール変換してプロット
            x1_text = 0.5 + (row[x_heat[0]] - x1_start) * (heat_division - 1) / (x1_end - x1_start)
            x2_text = 0.5 + (row[x_heat[1]] - x2_start) * (heat_division - 1) / (x2_end - x2_start)
            ax.text(x1_text, x2_text, rank_text, verticalalignment='center', horizontalalignment='left')
    
    @classmethod
    def _reg_heat_plot(cls, trained_estimator, X, y_pred, y_true, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                       pair_sigmarange=1.0, pair_sigmainterval=0.5, heat_extendsigma=0.5, heat_division=30, 
                       vmin=None, vmax=None, plot_scatter='true', maxerror=None,
                       rank_number=None, rank_col=None, rank_col_data=None, scatter_hue_dict=None,
                       rounddigit_rank=None, rounddigit_x1=None, rounddigit_x2=None, rounddigit_x3=None,
                       cv_index=None, subplot_kws=None, heat_kws=None, scatter_kws=None, legend_kws=None):
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
                
                cls._reg_heat_plot_2d(trained_estimator, x_heat, 'y_true', 'y_pred', rank_col, df_pair, x_heat_indices, hue_name,
                                      x1_start, x1_end, x2_start, x2_end, heat_division, other_x,
                                      vmin, vmax, ax, plot_scatter, maxerror, rank_dict, scatter_hue_dict,
                                      rounddigit_rank, rounddigit_x1, rounddigit_x2,
                                      heat_kws=heat_kws, scatter_kws=scatter_kws, legend_kws=legend_kws)

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
        plt.tight_layout(rect=[0, 0, 1, 0.98])

    @classmethod
    def regression_heat_plot(cls, estimator, x: List[str], y: str, data: pd.DataFrame = None,
                             x_colnames: List[str] = None, x_heat: List[str] = None, scatter_hue=None,
                             pair_sigmarange = 1.0, pair_sigmainterval = 0.5, heat_extendsigma = 0.5, 
                             heat_division = 30, color_extendsigma = 0.5,
                             plot_scatter = 'true', rounddigit_rank=3, rounddigit_x1=2, rounddigit_x2=2, rounddigit_x3=2,
                             rank_number=None, rank_col=None,
                             cv=None, cv_seed=42, cv_group=None, display_cv_indices = 0,
                             estimator_params=None, fit_params=None, validation_fraction=None,
                             subplot_kws=None, heat_kws=None, scatter_kws=None, legend_kws=None):
        """
        Plot regression heatmaps of any scikit-learn regressor with 2 to 4D explanatory variables.

        Parameters
        ----------
        estimator : estimator object implementing ``fit``
            Regression estimator. This is assumed to implement the scikit-learn estimator interface.

        x : list[str] or np.ndarray
            Explanatory variables. Should be list[str] if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        y : str or np.ndarray
            Objective variable. Should be str if ``data`` is pd.DataFrame. Should be np.ndarray if ``data`` is None

        data: pd.DataFrame
            Input data structure.

        x_colnames: list[str], optional
            Names of explanatory variables. Available only if ``data`` is NOT pd.DataFrame

        x_heat: list[str], optional
            X-axis and y-axis variables of heatmap. If None, use two variables in ``x`` from the front.

        scatter_hue : str, optional
            Grouping variable that will produce points with different colors. Available only if plot_scatter is set to ``hue``.

        pair_sigmarange: float, optional
            Set the range of subplots. The lower limit is mean({x3, x4}) - ``pair_sigmarange`` * std({x3, x4}). The higher limit is mean({x3, x4}) + ``pair_sigmarange`` * std({x3, x4}). Available only if len(x) is bigger than 2.

        pair_sigmainterval: float, optional
            Set the interval of subplots. For example, if ``pair_sigmainterval`` is set to 0.5 and ``pair_sigmarange`` is set to 1.0, The ranges of subplots are lower than μ-1σ, μ-1σ to μ-0.5σ, μ-0.5σ to μ, μ to μ+0.5σ, μ+0.5σ to μ+1σ, and higher than μ+1σ. Available only if len(x) is bigger than 2.

        heat_extendsigma: float, optional
            Set the axis view limits of the heatmap. The lower limit is min({x1, x2}) - std({x1, x2}) * ``heat_extendsigma``. The higher limit is max({x1, x2}) + std({x1, x2}) * ``heat_extendsigma``

        heat_division: int, optional
            Resolution of the heatmap.

        color_extendsigma: float, optional
            Set the colormap limits of the heatmap. The lower limit is min(y_ture) - std(y_ture) * ``color_extendsigma``. The higher limit is max(y_ture) - std(y_ture) * ``color_extendsigma``.

        plot_scatter: {'error', 'true', 'hue'}, optional
            Color decision of scatter plot. If 'error', to be mapped to colors using error value. If 'true', to be mapped to colors using y_ture value. If 'hue', to be mapped to colors using scatter_hue variable. If None, no scatter.

        rounddigit_rank: int, optional
            Round a number of error that are in the top posiotions for regression error to a given precision in decimal digits.

        rounddigit_x1: int, optional
            Round a number of x-axis valiable of the heatmap to a given precision in decimal digits.

        rounddigit_x2: int, optional
            Round a number of y-axis valiable of the heatmap to a given precision in decimal digits.

        rounddigit_x3: int, optional
            Round a number of y-axis valiable of subplots to a given precision in decimal digits.

        rank_number: int, optional
            Number of emphasized data that are in the top posiotions for regression error.

        rank_col: str, optional
            Variables that are displayed with emphasized data that are in the top posiotions for regression error.

        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. If None, no cross-validation is used and the training data is displayed. If int, to specify the number of folds in a KFold.

        cv_seed : int, optional
            Seed for random number generator of cross validation.

        cv_group: str, optional
            Group variable for the samples used while splitting the dataset into train/test set. This argument is passed to ``groups`` argument of cv.split().

        display_cv_indices : int or list, optional
            Cross validation index or indices to display.

        estimator_params : dict, optional
            Parameters passed to the regression estimator. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        fit_params : dict, optional
            Parameters passed to the fit() method of the regression estimator, e.g. ``early_stopping_round`` and ``eval_set`` of XGBRegressor. If the estimator is pipeline, each parameter name must be prefixed such that parameter p for step s has key s__p.

        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        subplot_kws: dict, optional
            Additional parameters passed to matplotlib.pyplot.subplots(), e.g. ``figsize``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

        heat_kws: dict, optional
            Additional parameters passed to sns.heatmap(), e.g. ``cmap``. See https://seaborn.pydata.org/generated/seaborn.heatmap.html

        scatter_kws: dict, optional
            Additional parameters passed to matplotlib.pyplot.scatter(), e.g. ``alpha``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

        legend_kws : dict
            Additional parameters passed to ax.legend(), e.g. ``loc``. See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        """

        # 入力データの形式統一
        X, y_true, data, x_colnames, y_colname, cv_group_colname = cls._reshape_input_data(x, y, data,
                                                                                           x_colnames,
                                                                                           cv_group)
        # 説明変数xの次元が2～4以外ならエラーを出す
        if len(x_colnames) < 2 or len(x_colnames) > 4:
            raise Exception('Dimension of x must be 2 to 4')
        
        # display_cv_indicesをList化
        if isinstance(display_cv_indices, int):
            display_cv_indices = [display_cv_indices]
        elif not isinstance(x_colnames, list):
            raise Exception('the "cv_display_indices" argument must be int or List[int]')
        # 学習器パラメータがあれば適用
        if estimator_params is not None:
            estimator.set_params(**estimator_params)
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
        # legend_kwsがNoneなら空のdictを入力
        if legend_kws is None:
            legend_kws = {}
        
        # ヒートマップ表示用の列を抽出
        if x_heat is None:  # 列名指定していないとき、前から2列を抽出
            x_heat = x_colnames[:2]
            x_heat_indices = [0, 1]
        else:  # 列名指定しているとき、該当列のXにおけるインデックス(0～3)を保持
            if len(x_heat) != 2:
                raise Exception('length of x_heat must be 2')
            x_heat_indices = []
            for colname in x_heat:
                x_heat_indices.append(x_colnames.index(colname))
        # ヒートマップ表示以外の列
        x_not_heat = [colname for colname in x_colnames if colname not in x_heat]        
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
            scatter_hue_dict = dict(zip(hue_list, cls._HEAT_SCATTER_HUECOLORS[0:len(hue_list)]))
        else:
            scatter_hue_dict = None
        
        # クロスバリデーション有無で場合分け
        # クロスバリデーション未実施時(学習データからプロット＆指標算出)
        if cv is None:
            # 学習と推論
            estimator.fit(X, y_true, **fit_params)
            y_pred = estimator.predict(X)
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
            cls._reg_heat_plot(estimator, X, y_pred, y_true, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                               pair_sigmarange = pair_sigmarange, pair_sigmainterval=pair_sigmainterval, heat_extendsigma=heat_extendsigma, heat_division=heat_division,
                               vmin=vmin, vmax=vmax, plot_scatter=plot_scatter, maxerror=maxerror,
                               rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_data, scatter_hue_dict=scatter_hue_dict,
                               rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                               cv_index=None, subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws, legend_kws=legend_kws)
            
        # クロスバリデーション実施時(分割ごとに別々にプロット＆指標算出)
        if cv is not None:
            # 分割法未指定時、cv_numとseedに基づきKFoldでランダムに分割
            if isinstance(cv, numbers.Integral):
                cv = KFold(n_splits=cv, shuffle=True, random_state=cv_seed)
            # LeaveOneOutのときエラーを出す
            if isinstance(cv, LeaveOneOut):
                raise Exception('"regression_heat_plot" method does not support "LeaveOneOut" cross validation')
            # cv_groupをグルーピング対象に指定(GroupKFold、LeaveOneGroupOut等)
            split_kws={}
            if cv_group_colname is not None:
                split_kws['groups'] = data[cv_group_colname].values
            elif isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need ``cv_group`` argument')
            # LeaveOneGroupOutのとき、クロスバリデーション分割数をcv_groupの数に指定
            if isinstance(cv, LeaveOneGroupOut):
                cv_num = len(set(data[cv_group_colname].values))
            else:
                cv_num = cv.n_splits

            # fit_paramsにeval_metricが入力されており、eval_setが入力されていないときの処理(eval_setにテストデータを使用)
            if validation_fraction is None:
                validation_fraction = 'cv'
            # 最終学習器以外の前処理変換器作成
            transformer = _make_transformer(validation_fraction, estimator)

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

                # random_state取得
                if isinstance(estimator, Pipeline):
                    random_state = estimator.steps[-1][1].random_state if 'random_state' in estimator.steps[-1][1].__dict__.keys() else None
                else:
                    random_state = estimator.random_state if 'random_state' in estimator.__dict__.keys() else None

                # eval_setの中から学習データ or テストデータのみを抽出
                fit_params_modified, train_divided = _eval_set_selection(
                    validation_fraction, 
                    transformer, 
                    X,
                    y,
                    fit_params, 
                    train, 
                    test,
                    random_state
                    )

                # 学習と推論
                estimator.fit(X[train_divided], y_true[train_divided], **fit_params_modified)
                y_pred = estimator.predict(X_test)
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
                cls._reg_heat_plot(estimator, X_test, y_pred, y_test, x_heat, x_not_heat, x_heat_indices, hue_data, hue_name,
                                   pair_sigmarange = pair_sigmarange, pair_sigmainterval = pair_sigmainterval, heat_extendsigma=heat_extendsigma, heat_division=heat_division,
                                   vmin=vmin, vmax=vmax, plot_scatter = plot_scatter, maxerror=maxerror,
                                   rank_number=rank_number, rank_col=rank_col, rank_col_data=rank_col_test, scatter_hue_dict=scatter_hue_dict,
                                   rounddigit_rank=rounddigit_rank, rounddigit_x1=rounddigit_x1, rounddigit_x2=rounddigit_x2, rounddigit_x3=rounddigit_x3,
                                   cv_index=i, subplot_kws=subplot_kws, heat_kws=heat_kws, scatter_kws=scatter_kws, legend_kws=legend_kws)

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class CustomPairPlot():
    #初期化
    def __init__(self):
        self.df = None
        self.hue = None
        self.hue_names = None
        self.corr_mat = None
    
    #hueごとに相関係数計算
    def _corrfunc(self, x, y, **kws):
        if self.hue_names is None:
            labelnum=0
            hue_num = 0
        else:
            labelnum=self.hue_names.index(kws["label"])
            hue_num = len(self.hue_names)
        #xまたはyがNaNの行を除外
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        x, y = x[mask], y[mask]
        #相関係数算出＆0.4ごとにフォントサイズ拡大
        r, _ = stats.pearsonr(x, y)
        fsize = min(9, 45/hue_num) + min(4.5, 22.5/hue_num) * np.ceil(abs(r)/0.4)
        fsize = min(9, 45/hue_num) if np.isnan(fsize) else fsize
        #該当マスのaxを取得
        if 'ax' in kws.keys():  # seaborn 0.11.1以降
            ax = kws['ax']
        else:  # seaborn 0.11.0以前
            ax = plt.gca()
        #既に表示したhueの分だけ下にさげて相関係数表示
        ax.annotate("r={:.2f}".format(r), xy=(.1, .65-min(.15,.75/hue_num)*labelnum), xycoords=ax.transAxes, size=fsize, color=kws["color"])
    
    #hueを分けない相関係数計算して上半分に表示
    def _corrall_upper(self, g):
        #右上を1マスずつ走査
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            #該当マスのaxesを取得
            ax = g.axes[i, j]
            plt.sca(ax)
            #フィールド名を取得
            x_var = g.x_vars[j]
            y_var = g.y_vars[i]
            #相関係数
            r = self.corr_mat[x_var][y_var]
            #相関係数0.2ごとにフォントサイズ拡大
            fsize = 10 + 5 * np.ceil(abs(r)/0.2)
            fsize = 10 if np.isnan(fsize) else fsize
            #一番上に表示
            ax.annotate("r={:.2f}".format(r), xy=(.1, .85), xycoords=ax.transAxes, size=fsize, color="black")

    #重複数に応じたバブルチャート
    def _duplicate_bubblescatter(self, data, x, y, hue=None, hue_names=None, hue_slide="horizontal", palette=None):
        #hueの要素数および値を取得
        if hue is not None:
            #hue_nameを入力していないとき、hueの要素から自動生成
            if hue_names is None:
                hue_names = data[hue].dropna().unique()
            hue_num = len(hue_names)
            hue_vals = data[hue]
        #hueを入力していないときも、groupby用にhue関係変数生成
        else:
            hue_names = ["_nolegend_"]
            hue_num = 0
            hue_vals = pd.Series(["_nolegend_"] * len(data),
                                      index=data.index)
        #hueで区切らない全データ数（NaNは除外）をカウント
        ndata = len(data[[x,y]].dropna(how="any"))

        ######hueごとにGroupByして表示処理######
        hue_grouped = data.groupby(hue_vals)
        for k, label_k in enumerate(hue_names):
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=data.columns,
                                      dtype=np.float)
            #X,Yごとに要素数をカウント
            df_xy = data_k[[x,y]].copy()
            df_xy["xyrec"] = 1
            df_xycount = df_xy.dropna(how="any").groupby([x,y], as_index=False).count()
            #hueが2個以上存在するとき、表示位置ずらし量（対象軸のユニーク値差分最小値÷4に収まるよう設定）を計算
            if hue_num >=2:
                if hue_slide == "horizontal":
                    x_distinct_sort = sorted(data[x].dropna().unique())
                    x_diff_min = min(np.diff(x_distinct_sort))
                    x_offset = k * (x_diff_min/4)/(hue_num - 1) - x_diff_min/8
                    y_offset = 0
                else:
                    y_distinct_sort = sorted(data[y].dropna().unique())
                    y_diff_min = min(np.diff(y_distinct_sort))
                    x_offset = 0
                    y_offset = k * (y_diff_min/4)/(hue_num - 1) - y_diff_min/8
            else:
                x_offset = 0
                y_offset = 0
            #散布図表示（要素数をプロットサイズで表現）
            ax = plt.gca()
            ax.scatter(df_xycount[x] + x_offset, df_xycount[y] + y_offset, s=df_xycount["xyrec"]*1000/ndata, color=palette[k])

    #plotter=scatterかつ要素数が2以下なら箱ひげ図、それ以外ならscatterplotを使用
    def _boxscatter_lower(self, g, **kwargs):
        #kw_color = kwargs.pop("color", None)
        kw_color = g.palette
        #左下を走査
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            ax = g.axes[i, j]
            plt.sca(ax)
            #軸表示対象のフィールド名を取得
            x_var = g.x_vars[j]
            y_var = g.y_vars[i]
            #XY軸データ抽出
            x_data = self.df[x_var]
            y_data = self.df[y_var]
            #XY軸のユニーク値
            x_distinct = x_data.dropna().unique()
            y_distinct = y_data.dropna().unique()
            
            #箱ひげ図(x方向)
            if len(x_distinct) ==2 and len(y_distinct) >= 5:
                sns.boxplot(data=self.df, x=x_var, y=y_var, orient="v",
                     hue=self.hue, palette=g.palette, **kwargs)
            #重複数に応じたバブルチャート(x方向)
            elif len(x_distinct) ==2 and len(y_distinct) < 5:
                self._duplicate_bubblescatter(data=self.df, x=x_var, y=y_var, hue=self.hue, hue_names=g.hue_names, hue_slide="horizontal", palette=g.palette)
            #箱ひげ図(y方向)
            elif len(y_distinct) ==2 and len(x_distinct) >= 5:
                sns.boxplot(data=self.df, x=x_var, y=y_var, orient="h",
                     hue=self.hue, palette=g.palette, **kwargs)
            #重複数に応じたバブルチャート(y方向)
            elif len(y_distinct) ==2 and len(x_distinct) < 5:
                self._duplicate_bubblescatter(data=self.df, x=x_var, y=y_var, hue=self.hue, hue_names=g.hue_names, hue_slide="vertical", palette=g.palette)
            #散布図
            else:
                if len(g.hue_kws) > 0 and "marker" in g.hue_kws.keys():#マーカー指定あるとき
                    markers = dict(zip(g.hue_names, g.hue_kws["marker"]))
                else:#マーカー指定ないとき
                    markers = True
                sns.scatterplot(data=self.df, x=x_var, y=y_var, hue=self.hue,
                     palette=g.palette, style=self.hue, markers=markers)
            #凡例を追加
            g._update_legend_data(ax)
            ax.legend_ = None

        if kw_color is not None:
            kwargs["color"] = kw_color
        #軸ラベルを追加
        g._add_axis_labels()

    #メイン関数
    def pairanalyzer(self, df, hue=None, palette=None, vars=None,
             lowerkind="boxscatter", diag_kind="kde", markers=None,
             height=2.5, aspect=1, dropna=True,
             lower_kws={}, diag_kws={}, grid_kws={}):
        """
        Plotting pair plot including scatter plot and correlation coefficient matrix simultaneously.
        This method mainly uses seaborn.PairGrid class.

        Parameters
        ----------
        df : pd.DataFrame
            Input data structure. Int, float, and bool columns are displayed in the output graph.
        hue : str
            Variable in data to map plot aspects to different colors.
        palette : str or dict[str]
            Set of colors for mapping the hue variable. If a dict, keys should be values in the hue variable.
        vars : list[str]
            Variables within data to use, otherwise use every column with a numeric datatype.
        lowerkind : {'boxscatter', 'scatter', or 'reg'}
            Kind of plot for the lower triangular subplots.
        diag_kind : {'kde' or 'hist'}
            Kind of plot for the diagonal subplots.
        markers : str or list[str]
            Marker to use for all scatterplot points or a list of markers. See https://matplotlib.org/stable/api/markers_api.html
        height : float
            Height (in inches) of each facet.
        aspect : float
            Aspect * height gives the width (in inches) of each facet.
        dropna : bool
            Drop missing values from the data before plotting.
        lower_kws : dict
            Additional parameters passed to seaborn.PairGrid.map_lower(). If ``lowerkind`` is 'scatter', the arguments are applied to seaborn.scatterplot method of the lower subplots.
        diag_kws : dict
            Additional parameters passed to seaborn.PairGrid.map_diag(). If ``lowerkind`` is 'kde', the arguments are applied to seaborn.kdeplot method of the diagonal subplots.
        grid_kws : dict
            Additional parameters passed to seaborn.PairGrid.__init__() other than the above arguments. See https://seaborn.pydata.org/generated/seaborn.PairGrid.html
        """
        # Check whether hue column is exist in df
        if hue not in df.columns:
            raise AttributeError(f"'{hue}' doesn't exist in the columns of `df`")
        # Select columns whose type is in [int, float, bool]
        self.df = df.select_dtypes(include=[int, float, bool])
        # bool型の列をintに変換
        bool_cols = self.df.select_dtypes(include=bool).columns
        for col in bool_cols:
            self.df[col] = self.df[col] * 1
        #メンバ変数入力
        self.hue = hue
        self.corr_mat = self.df.corr(method="pearson")
        #文字サイズ調整
        sns.set_context("notebook")

        # Add hue column if the column was deleted because of the type filter
        if hue not in self.df.columns:
            self.df[hue] = df[hue]

        #PairGridインスタンス作成
        plt.figure()
        diag_sharey = diag_kind == "hist"
        g = sns.PairGrid(self.df, hue=self.hue,
                 palette=palette, vars=vars, diag_sharey=diag_sharey,
                 height=height, aspect=aspect, dropna=dropna, **grid_kws)
        self.hue_names = g.hue_names

        #マーカーを設定
        if markers is not None:
            if g.hue_names is None:
                n_markers = 1
            else:
                n_markers = len(g.hue_names)
            if not isinstance(markers, list):
                markers = [markers] * n_markers
            if len(markers) != n_markers:
                raise ValueError(("markers must be a singleton or a list of "
                                "markers for each level of the hue variable"))
            g.hue_kws = {"marker": markers}

        #対角にヒストグラム or KDEをプロット
        if diag_kind == "hist":
            g.map_diag(sns.histplot, **diag_kws)
        elif diag_kind == "kde":
            diag_kws.setdefault("fill", True)
            diag_kws["legend"] = False
            g.map_diag(sns.kdeplot, **diag_kws)

        #各変数のユニーク数を計算
        nuniques = []
        for col_name in g.x_vars:
            col_data = self.df[col_name]
            nuniques.append(len(col_data.dropna().unique()))

        #左下に散布図etc.をプロット
        if lowerkind == "boxscatter":
            if min(nuniques) <= 2: #ユニーク数が2の変数が存在するときのみ、箱ひげ表示
                self._boxscatter_lower(g, **lower_kws)
            else: #ユニーク数が2の変数が存在しないとき、散布図(_boxscatter_lowerを実行すると凡例マーカーが消えてしまう)
                g.map_lower(sns.scatterplot, **lower_kws)
        elif lowerkind == "scatter":
            g.map_lower(sns.scatterplot, **lower_kws)
        else:
            g.map_lower(sns.regplot, **lower_kws)

        #色分け（hue）有無で場合分けしてプロット＆相関係数表示実行
        #hueなし
        if self.hue is None:
            #右上に相関係数表示
            self._corrall_upper(g)
        #hueあり
        else:
            #右上に相関係数表示(hueごとに色分け＆全体の相関係数を黒表示)
            g.map_upper(self._corrfunc)
            self._corrall_upper(g)
            g.add_legend()
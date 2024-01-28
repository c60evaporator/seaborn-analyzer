import copy
import numpy as np
import time
import numbers
from itertools import product
from collections import defaultdict
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import check_cv, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection._validation import _fit_and_score, _insert_error_scores, _aggregate_score_dicts, _normalize_score_results, _translate_train_sizes, _incremental_fit_estimator
from sklearn.utils.validation import indexable, check_random_state
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.base import is_classifier
from sklearn.utils.parallel import delayed, Parallel

def _transform_except_last_estimator(transformer, X_src, X_train):
    """パイプラインのとき、最終学習器以外のtransformを適用"""
    if transformer is not None:
        transformer.fit(X_train)
        X_dst = transformer.transform(X_src)
        return X_dst
    else:
        return X_src

def _eval_set_selection(validation_fraction, 
                        transformer, 
                        X, 
                        y,
                        fit_params, 
                        train, 
                        test, 
                        random_state
                        ):
    """eval_setの中から学習データ or テストデータのみを抽出"""
    fit_params_modified = copy.deepcopy(fit_params)
    # eval_setが存在しない or Noneなら、そのままfit_paramsを返す
    eval_sets = [v for v in fit_params.keys() if 'eval_set' in v]
    if len(eval_sets) == 0 or fit_params[eval_sets[0]] is None:
        return fit_params_modified, train
    # eval_setの列名(pipelineでは列名が変わるため)
    eval_set_name = eval_sets[0]
    # 元のeval_setからX, yを取得
    X_fit = fit_params[eval_set_name][0][0]
    y_fit = fit_params[eval_set_name][0][1]
    # Training dataの一部をeval_setとして使用する場合
    if isinstance(validation_fraction, float):
        # Select training data from source training data
        rng = np.random.default_rng(seed=random_state)
        size = int(train.shape[0]*(1.0 - validation_fraction))
        train_divided = rng.choice(train, size=size, replace=False)
        train_divided.sort()
        # Select validation data from difference set of the selected training data and the source training data
        val_divided = np.setdiff1d(train, train_divided)
        fit_params_modified[eval_set_name] = [(_transform_except_last_estimator(transformer, X[val_divided], X[train_divided])\
                                              , y[val_divided])]
    # Cross validationのテストデータをeval_setとして使用する場合
    elif validation_fraction == 'cv':
        fit_params_modified[eval_set_name] = [(_transform_except_last_estimator(transformer, X[test], X[train])\
                                              , y[test])]
        train_divided = train
    # eval_setをそのまま使用、またはPipelineのみ適用して使用する場合
    else:
        fit_params_modified[eval_set_name] = [(_transform_except_last_estimator(transformer, X_fit, X[train])\
                                              , y_fit)]
        train_divided = train

    return fit_params_modified, train_divided

def _fit_and_score_eval_set(
    validation_fraction,
    transformer,
    estimator,
    X, 
    y, 
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    score_params=None,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
    ):
    """Fit estimator and compute scores for a given dataset split."""
    # random_state取得
    if isinstance(estimator, Pipeline):
        random_state = estimator.steps[-1][1].random_state if 'random_state' in estimator.steps[-1][1].__dict__.keys() else None
    else:
        random_state = estimator.random_state if 'random_state' in estimator.__dict__.keys() else None
    # fit_params内のデータをvalidation_fractionに合わせて整形 (validation_fraction='')
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

    # 学習してスコア計算
    result = _fit_and_score(estimator, X, y, 
                            scorer=scorer, train=train_divided, 
                            test=test, verbose=verbose, parameters=parameters,
                            fit_params=fit_params_modified,
                            score_params=score_params,
                            return_train_score=return_train_score,
                            return_parameters=return_parameters, return_n_test_samples=return_n_test_samples,
                            return_times=return_times, return_estimator=return_estimator,
                            split_progress=split_progress, candidate_progress=candidate_progress,
                            error_score=error_score)
    return result

def _make_transformer(validation_fraction, estimator):
    """estimatorがパイプラインのとき、最終学習器以外の変換器(前処理クラスのリスト)を作成"""
    if isinstance(estimator, Pipeline) and validation_fraction is not None:
        transformer = Pipeline([step for i, step in enumerate(estimator.steps) if i < len(estimator) - 1])
        return transformer
    else:
        return None

def cross_validate_eval_set(estimator,
    X,
    y=None,
    validation_fraction='cv',
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    """
    Evaluate a scores by cross-validation with `eval_set` argument in `fit_params`

    This method is suitable for calculating cross validation scores with `early_stopping_round` in XGBoost or LightGBM.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of
        supervised learning.

    validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
        Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

        If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
        
        If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

        If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

        If None, use raw `eval_set`.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
    """

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # 最終学習器以外の前処理変換器作成
    transformer = _make_transformer(validation_fraction, estimator)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score_eval_set)(
            validation_fraction, transformer,
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)
        for train, test in cv.split(X, y, groups))

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret['fit_time'] = results["fit_time"]
    ret['score_time'] = results["score_time"]

    if return_estimator:
        ret['estimator'] = results["estimator"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret['test_%s' % name] = test_scores_dict[name]
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = train_scores_dict[name]

    return ret

def cross_val_score_eval_set(
    estimator,
    X,
    y=None,
    validation_fraction='cv',
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
):
    """
    Evaluate a score by cross-validation with `eval_set` argument in `fit_params`

    This method is suitable for calculating cross validation score with `early_stopping_round` in XGBoost or LightGBM.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    
    validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
        Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

        If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
        
        If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

        If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

        If None, use raw `eval_set`.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.

        Similar to :func:`cross_validate`
        but only a single metric is permitted.

        If `None`, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate_eval_set(estimator=estimator, X=X, y=y, validation_fraction=validation_fraction,
                                         groups=groups,
                                         scoring={'score': scorer}, cv=cv,
                                         n_jobs=n_jobs, verbose=verbose,
                                         fit_params=fit_params,
                                         pre_dispatch=pre_dispatch,
                                         error_score=error_score)
    return cv_results['test_score']

def validation_curve_eval_set(
    estimator,
    X, 
    y,
    *,
    param_name, 
    param_range, 
    validation_fraction='cv',
    groups=None,
    cv=None, 
    scoring=None, 
    n_jobs=None, 
    pre_dispatch="all",
    verbose=0, 
    error_score=np.nan, 
    fit_params=None
):
    """Validation curve.

    Determine training and test scores for varying parameter values with `eval_set` argument in `fit_params`

    Parameters
    ----------    
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : str
        Name of the parameter that will be varied.

    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.

    validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
        Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

        If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
        
        If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

        If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

        If None, use raw `eval_set`.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the combinations of each parameter
        value and each cross-validation split.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int or str, default='all'
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The str can
        be an expression like '2*n_jobs'.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. versionadded:: 0.24

    Returns
    -------
    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)

    # 最終学習器以外の前処理変換器作成
    transformer = _make_transformer(validation_fraction, estimator)

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    results = parallel(
        delayed(_fit_and_score_eval_set)(
            validation_fraction,
            transformer,
            clone(estimator), 
            X, 
            y, 
            scorer, 
            train, 
            test, 
            verbose,
            parameters={param_name: v}, 
            fit_params=fit_params,
            return_train_score=True, 
            error_score=error_score,
        )
        # NOTE do not change order of iteration to allow one time cv splitters
        for train, test in cv.split(X, y, groups) 
        for v in param_range
    )
    n_params = len(param_range)

    results = _aggregate_score_dicts(results)
    train_scores = results["train_scores"].reshape(-1, n_params).T
    test_scores = results["test_scores"].reshape(-1, n_params).T

    return train_scores, test_scores

def learning_curve_eval_set(
    estimator,
    X, 
    y,
    *,
    validation_fraction='cv',
    groups=None,
    train_sizes=np.linspace(0.1, 1.0, 5), 
    cv=None,
    scoring=None, 
    exploit_incremental_learning=False,
    n_jobs=None, 
    pre_dispatch="all",
    verbose=0, 
    shuffle=False,
    random_state=None, 
    error_score=np.nan,
    return_times=False,
    fit_params=None,
):
    """Learning curve.

    Determines cross-validated training and test scores for different training set sizes with `eval_set` argument in `fit_params`

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.

    validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
        Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

        If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
        
        If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

        If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

        If None, use raw `eval_set`.

    groups : array-like of  shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    train_sizes : array-like of shape (n_ticks,), \
            default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    exploit_incremental_learning : bool, default=False
        If the estimator supports incremental learning, this will be
        used to speed up fitting for different training set sizes.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the different training and test sets.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int or str, default='all'
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The str can
        be an expression like '2*n_jobs'.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    shuffle : bool, default=False
        Whether to shuffle training data before taking prefixes of it
        based on``train_sizes``.

    random_state : int, RandomState instance or None, default=None
        Used when ``shuffle`` is True. Pass an int for reproducible
        output across multiple function calls.
        See :term:`Glossary <random_state>`.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    return_times : bool, default=False
        Whether to return the fit and score times.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. versionadded:: 0.24

    Returns
    -------
    train_sizes_abs : array of shape (n_unique_ticks,)
        Numbers of training examples that has been used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.

    fit_times : array of shape (n_ticks, n_cv_folds)
        Times spent for fitting in seconds. Only present if ``return_times``
        is True.

    score_times : array of shape (n_ticks, n_cv_folds)
        Times spent for scoring in seconds. Only present if ``return_times``
        is True.
    """
    if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError("An estimator must support the partial_fit interface "
                         "to exploit incremental learning")
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes, n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    # 最終学習器以外の前処理変換器作成
    transformer = _make_transformer(validation_fraction, estimator)

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)

    if shuffle:
        rng = check_random_state(random_state)
        cv_iter = ((rng.permutation(train), test) for train, test in cv_iter)

    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel(
            delayed(_incremental_fit_estimator)(
                clone(estimator), 
                X, 
                y, 
                classes, 
                train, 
                test, 
                train_sizes_abs,
                scorer, 
                verbose, 
                return_times, 
                error_score=error_score,
                fit_params=fit_params
            )
            for train, test in cv_iter
        )
        out = np.asarray(out).transpose((2, 1, 0))
    else:
        train_test_proportions = []
        for train, test in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))

        results = parallel(
            delayed(_fit_and_score_eval_set)(
                validation_fraction, 
                transformer,
                clone(estimator), 
                X, 
                y, 
                scorer, 
                train, 
                test, 
                verbose,
                parameters=None, 
                fit_params=fit_params, 
                return_train_score=True,
                error_score=error_score, 
                return_times=return_times
            )
            for train, test in train_test_proportions
        )
        results = _aggregate_score_dicts(results)
        train_scores = results["train_scores"].reshape(-1, n_unique_ticks).T
        test_scores = results["test_scores"].reshape(-1, n_unique_ticks).T
        out = [train_scores, test_scores]

        if return_times:
            fit_times = results["fit_time"].reshape(-1, n_unique_ticks).T
            score_times = results["score_time"].reshape(-1, n_unique_ticks).T
            out.extend([fit_times, score_times])

    ret = train_sizes_abs, out[0], out[1]

    if return_times:
        ret = ret + (out[2], out[3])

    return ret

class GridSearchCVEvalSet(GridSearchCV):
    """
    Exhaustive search over specified parameter values for an estimator with `eval_set` argument in `fit_params`.
    """
    def fit(self,
            X, 
            y=None, 
            groups=None,
            validation_fraction='cv',
            **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        #fit_params = _check_method_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        # 最終学習器以外の前処理変換器作成
        transformer = _make_transformer(validation_fraction, estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score_eval_set)(
                        validation_fraction, 
                        transformer,
                        clone(base_estimator),
                        X, 
                        y,
                        train=train, 
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

class RandomizedSearchCVEvalSet(RandomizedSearchCV):
    """
    Randomized search on hyper parameters with `eval_set` argument in `fit_params`.
    """
    def fit(self,
            X, 
            y=None, 
            groups=None,
            validation_fraction='cv',
            **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        validation_fraction : {float, 'cv', 'transformed', or None}, default='cv'
            Select data passed to `eval_set` in `fit_params`. Available only if "estimator" is LGBMRegressor, LGBMClassifier, XGBRegressor, or XGBClassifier.

            If float, devide source training data into training data and eval_set according to the specified ratio like sklearn.ensemble.GradientBoostingRegressor.
            
            If "cv", select test data from `X` and `y` using cv.split() like lightgbm.cv.

            If "transformed", use `eval_set` transformed by `fit_transform()` of the pipeline if the `estimater` is sklearn.pipeline.Pipeline object.

            If None, use raw `eval_set`.

        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        # fit_params = _check_method_params(X, fit_params)  TODO: Replace _checkfit_params() to _check_method_params() to comply with scikit-learn >= 1.4

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        # 最終学習器以外の前処理変換器作成
        transformer = _make_transformer(validation_fraction, estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score_eval_set)(
                        validation_fraction, 
                        transformer,
                        clone(base_estimator),
                        X, 
                        y,
                        train=train, 
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
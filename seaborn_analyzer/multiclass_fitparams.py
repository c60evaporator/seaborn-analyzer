import warnings
import numpy as np
from sklearn import clone
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.parallel import delayed, Parallel

def _fit_binary(estimator, X, y, classes=None, **kwargs):
    """Fit a single binary estimator with kwargs."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." % str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y, **kwargs)
    return estimator

class OneVsRestClassifierPatched(OneVsRestClassifier):
    """One-vs-the-rest (OvR) multiclass strategy with ``fit_params``."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, fit_params_list):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]], **fit_params_list[i])
            for i, column in enumerate(columns))
        return self
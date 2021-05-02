# %%
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from custom_scatter_plot import classplot
iris = sns.load_dataset("iris")
features = ['petal_width', 'petal_length', 'sepal_width']
model = SVC()
classplot.class_separator_plot(model, features, 'species', iris)
# %%
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from custom_scatter_plot import classplot
iris = sns.load_dataset("iris")
features = ['petal_width', 'petal_length']
model = SVC(probability=True)
classplot.class_proba_plot(model, features, 'species', iris, proba_class='virginica', cv=2, display_cv_indices = [0, 1])
# %%

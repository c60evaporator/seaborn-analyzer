================
seaborn-analyzer
================

|python| |pypi| |license|

.. |python| image:: https://img.shields.io/pypi/pyversions/seaborn-analyzer
   :target: https://www.python.org/

.. |pypi| image:: https://img.shields.io/pypi/v/seaborn-analyzer?color=blue
   :target: https://pypi.org/project/seaborn-analyzer/

.. |license| image:: https://img.shields.io/pypi/l/seaborn-analyzer?color=blue
   :target: https://github.com/c60evaporator/seaborn-analyzer/blob/master/LICENSE
   
**A data analysis and visualization tool using Seaborn library.**

.. image:: https://user-images.githubusercontent.com/59557625/126887193-ceba9bdd-3653-4d58-a916-21dcfe9c38a0.png

=====
Usage
=====
An example of using CustomPairPlot class

.. code-block:: python

    from seaborn_analyzer import CustomPairPlot
    import seaborn as sns
 
    titanic = sns.load_dataset("titanic")
    cp = CustomPairPlot()
    cp.pairanalyzer(titanic, hue='survived')
   
If you want to know usage of other classes, see `API Reference
<https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#api-reference>`__ and `Examples
<https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#examples>`__

============
Requirements
============
seaborn-analyzer 0.1.2 requires

* Python >=3.6
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Scipy >=1.6.3
* Scikit-learn >=0.24.2

===========================
Installing seaborn-analyzer
===========================
Use pip to install the binary wheels on `PyPI <https://pypi.org/project/seaborn-analyzer/>`__

.. code-block:: console

    $ pip install seaborn-analyzer

=======
Support
=======
Bugs may be reported at https://github.com/c60evaporator/seaborn-analyzer/issues

=============
API Reference
=============
The following classes and methods are included in seaborn-analyzer

CustomPairPlot class
====================

.. csv-table::
    :header: "Method name", "Summary", "API Documentation", "Example"
    :widths: 30, 50, 15, 15

    "**pairanalyzer**", Plotting pair plot including scatter plot and correlation coefficient matrix simultaneously, `CustomPairPlot.pairanalyzer <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#custompairplotpairanalyzer>`__


hist class
==========

.. csv-table::
    :header: "Method name", "Summary", "API Documentation", "Example"
    :widths: 30, 50, 15, 15

    "**plot_normality**", Plotting normality test and QQ plot, `hist.plot_normality <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#histplot_normality>`__
    "**fit_dist**", Fitting distributions and calculating fitting scores, `hist.fit_dist <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#histfit_dist>`__


classplot class
===============

.. csv-table::
    :header: "Method name", "Summary", "API Documentation", "Example"
    :widths: 30, 50, 15, 15

    "**class_separator_plot**", Plotting class separation lines of any scikit-learn classification models, `hist.class_separator_plot <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#classplotclass_separator_plot>`__
    "**class_proba_plot**", Plotting class prediction probability of any scikit-learn classification models, `hist.class_proba_plot <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#classplotclass_proba_plot>`__


regplot class
=============

.. csv-table::
    :header: "Method name", "Summary", "API Documentation", "Example"
    :widths: 30, 50, 15, 15

    "**linear_plot**", Plotting linear regression line and calculating Pearson correlation coefficient, `regplot.linear_plot <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#regplotlinear_plot>`__
    "**regression_pred_true**", Plotting prediction value vs. true value scatter plots, `regplot.regression_pred_true <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#regplotregression_pred_true>`__
    "**regression_plot_1d**", Plotting 1d regression lines of any scikit-learn regression models, `regplot.regression_plot_1d <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#regplotregression_plot_1d>`__
    "**regression_heat_plot**", Plotting 2 to 4d regression heat maps of any scikit-learn regression models, `regplot.regression_heat_plot <https://pypi.org/project/seaborn-analyzer/>`__, `example <https://github.com/c60evaporator/seaborn-analyzer/blob/master/README.rst#regplotregression_heat_plot>`__


========
Examples
========

CustomPairPlot.pairanalyzer
===========================
.. code-block:: python

    from seaborn_analyzer import CustomPairPlot
    import seaborn as sns
    titanic = sns.load_dataset("titanic")
    cp = CustomPairPlot()
    cp.pairanalyzer(titanic, hue='survived')
.. image:: https://user-images.githubusercontent.com/59557625/115889860-4e8bde80-a48f-11eb-826a-cd3c79556a42.png

hist.plot_normality
===================
.. code-block:: python

    from seaborn_analyzer import hist
    from sklearn.datasets import load_boston
    import pandas as pd
    df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
    hist.plot_normality(df, x='LSTAT', norm_hist=False, rounddigit=5)
.. image:: https://user-images.githubusercontent.com/59557625/117275256-cfd46f80-ae98-11eb-9da7-6f6e133846fa.png

hist.fit_dist
=============
.. code-block:: python

    from seaborn_analyzer import hist
    from sklearn.datasets import load_boston
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    df = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
    all_params, all_scores = hist.fit_dist(df, x='LSTAT', dist=['norm', 'gamma', 'lognorm', 'uniform'])
    df_scores = pd.DataFrame(all_scores).T
    df_scores
.. image:: https://user-images.githubusercontent.com/59557625/115890066-81ce6d80-a48f-11eb-8390-f985d9e2b8b1.png
.. image:: https://user-images.githubusercontent.com/59557625/115890108-8d219900-a48f-11eb-9896-38f7dedbb6e4.png

classplot.class_separator_plot
==============================
.. code-block:: python

    import seaborn as sns
    from sklearn.svm import SVC
    from seaborn_analyzer import classplot
    iris = sns.load_dataset("iris")
    model = SVC()
    classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris)
.. image:: https://user-images.githubusercontent.com/59557625/117274234-d7474900-ae97-11eb-9de2-c8a74dc179a5.png

classplot.class_proba_plot
==========================
.. code-block:: python

    import seaborn as sns
    from sklearn.svm import SVC
    from seaborn_analyzer import classplot
    iris = sns.load_dataset("iris")
    model = SVC()
    classplot.class_proba_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                               proba_type='imshow')
.. image:: https://user-images.githubusercontent.com/59557625/117276085-a1a35f80-ae99-11eb-8368-cdd1cfa78346.png

regplot.linear_plot
===================
.. code-block:: python

    from seaborn_analyzer import regplot
    import seaborn as sns
    iris = sns.load_dataset("iris")
    regplot.linear_plot(x='petal_length', y='sepal_length', data=iris)
.. image:: https://user-images.githubusercontent.com/59557625/117276994-65243380-ae9a-11eb-8ec8-fa1fb5d60a55.png

regplot.regression_pred_true
============================
.. code-block:: python

    import pandas as pd
    from seaborn_analyzer import regplot
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
    regplot.regression_pred_true(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
.. image:: https://user-images.githubusercontent.com/59557625/117277036-6fdec880-ae9a-11eb-887a-5f8b2a93b0f9.png

regplot.regression_plot_1d
==========================
.. code-block:: python

    from seaborn_analyzer import regplot
    import seaborn as sns
    from sklearn.svm import SVR
    iris = sns.load_dataset("iris")
    regplot.regression_plot_1d(SVR(), x='petal_length', y='sepal_length', data=iris)
.. image:: https://user-images.githubusercontent.com/59557625/117277075-78cf9a00-ae9a-11eb-835c-01f635754f7b.png

regplot.regression_heat_plot
============================
.. code-block:: python

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from seaborn_analyzer import regplot
    df_temp = pd.read_csv(f'./sample_data/temp_pressure.csv')
    regplot.regression_heat_plot(LinearRegression(), x=['altitude', 'latitude'], y='temperature', data=df_temp)
.. image:: https://user-images.githubusercontent.com/59557625/115955837-1b4f5b00-a534-11eb-91b0-b913019d26ff.png

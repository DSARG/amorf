.. framework documentation master file, created by
   sphinx-quickstart on Sun Oct 20 18:19:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to amorf's documentation!
===================================== 

amorf is **a** **m**\ ulti-**o**\utput **r**\egression **f**\ramework in Python. 
It helps you to get started with multi-output regression fast
and allows you to increase your performance by using customized models that suit your needs. 

Installation
------------- 
Use the package manager pip_ to install amorf. 

.. code-block:: bash 

    pip install amorf


.. _pip: https://pip.pypa.io/en/stable/ 


Getting Started 
---------------
.. code-block:: python

    import amorf.neuralNetRegression as nnr 
    from amorf.metrics import average_relative_root_mean_squared_error as arrmse

    # for data generation
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=10000, n_features=12, n_targets=3, noise=0.1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    regressor = nnr.NeuralNetRegressor(patience=5, training_limit=None) #initialize neural net regressor
    regressor.fit(X_train, y_train) #fit regressor to training data
    prediction = regressor.predict(X_test) #predict test data 
    print(arrmse(prediction, y_test)) #print error

.. toctree::
   :maxdepth: 1
   :caption: Contents: 

   modules.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

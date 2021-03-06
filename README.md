[![Documentation Status](https://readthedocs.org/projects/amorf/badge/?version=latest)](https://amorf.readthedocs.io/en/latest/?badge=latest) 
[![PyPI version shields.io](https://img.shields.io/pypi/v/amorf.svg)](https://pypi.org/project/amorf/) 
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/amorf.svg)](https://pypi.python.org/pypi/ansicolortags/)

# amorf - A Multi-Output Regression Framework

**This Project is still a work in Progress** 

amorf is a Python library for multi-output regression. It combines several different approaches to help you get started with multi-output regression analysis. 

This project was created as part of a masters thesis by David Hildner

## Motivation 
Multi-output (or multi-target) regression models are models with multiple continuous target variables. 

This framework was largely inspired by 
 
> Borchani, Hanen & Varando, Gherardo & Bielza, Concha & Larranaga, Pedro. (2015). A survey on multi-output regression. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery. 5. 10.1002/widm.1157.  


This framework/library aims to collect and combine several different approaches for multi-output regression in one place. This allows you to get started real quick and then extend and tweak the provided models to suit your needs.
## Getting Started 

[//]: # (### Prerequisites )

### Installation 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install amorf.

```bash
pip install amorf
```
### Usage 
```python
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
```

## Documentation 

The [documentation](https://amorf.readthedocs.io/en/latest/) is hosted via [ReadTheDocs](https://readthedocs.org)

## Running The Tests 
Clone repository

```bash 
git clone https://github.com/DSAAR/amorf/
```
Change directory 
```bash 
cd amorf/
``` 
Discover and run tests 

```bash 
python -m unittest discover -s tests -p 'test_*.py'
```
## Authors  
* David Hildner - Student at [Eberhard-Karls Universität Tübingen](https://uni-tuebingen.de/)

## License 

MIT License



# amorf - A Multi-Output Regression Framework

**This Project is still a work in Progress** 

amorf is a Python library for multi-output regression. It combines several different approaches to help you get started with multi-output regression analyisis. 

This project was created as part of a masters thesis by David Hildner

## Motivation 

## Getting Started 

[//]: # (### Prerequisites )

### Installation 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install amorf.

```bash
pip install amorf
```
### Usage 
```python
import amorf.NeuralNetRegression as nnr 
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

Here be Link to [ReadTheDocs](https://readthedocs.org/)

## Running The Tests 
Clone repository

```bash 
git clone https:\\xxx
```
Change directory 
```bash 
cd amorf/
``` 
Disocver and run tests 

```bash 
python -m unittest discover -s tests -p 'test_*.py'
```
## Authors  
* David Hildner - Student at [Eberhard-Karls Universität Tübingen](https://uni-tuebingen.de/)

## License 

To be anounced



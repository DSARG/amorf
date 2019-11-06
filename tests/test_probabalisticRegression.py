import unittest
import framework.datasets as ds
from sklearn.model_selection import train_test_split
import numpy 

class TestBayesianRegression(unittest.TestCase): 

    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1) 

    def test_fit(): 

    def test_predict(): 
    
    def test_fit_with_gpu(): 

    def test_predict_with_gpu(): 

    
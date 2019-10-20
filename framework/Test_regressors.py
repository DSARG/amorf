import unittest 
import multiOutputRegressors as mor 
import datasets as ds 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import numpy 

class TestSingleTargetMethod(unittest.TestCase): 
    # TODO: Maybe add dictionary to check correct type of regressor
    def setUp(self): 
        X,y = ds.load_EDM()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.1)
        self.selectors = ['linear', 'kneighbors','adaboost','gradientboost','mlp','svr',]

    def test_correct_assignment(self): 
        for selector in self.selectors:
            regressor = mor.SingleTargetMethod(selector)
            self.assertEqual(regressor.MORegressor._estimator_type, 'regressor') 
        self.assertRaises(ValueError, mor.SingleTargetMethod,'nonexistent_selector') 
        self.assertEqual(mor.SingleTargetMethod(custom_regressor = RidgeCV()).MORegressor._estimator_type, 'regressor')
    
    def test_fit(self): 
        for selector in self.selectors:
            regressor = mor.SingleTargetMethod(selector) 
            self.assertEqual(regressor.fit(self.X_train,self.y_train)._estimator_type, 'regressor')  
    
    def test_predict(self):
        for selector in self.selectors:
            result = mor.SingleTargetMethod(selector).fit(self.X_train,self.y_train).predict(self.X_test) 
            self.assertEqual(result.shape, (len(self.X_test),len(self.y_test[0,:]))) 
            self.assertTrue(type(result) is numpy.ndarray) 
            self.assertTrue(result.dtype is numpy.dtype('float32') or result.dtype is numpy.dtype('float64')) 
    
    def test_custom_regressor(self): 
        valid_estimator = RidgeCV() 
        invalid_estimator = object() 
        self.assertFalse(mor.SingleTargetMethod.implements_SciKitLearn_API(None,object = invalid_estimator))
        self.assertTrue(mor.SingleTargetMethod.implements_SciKitLearn_API(None,object = valid_estimator))
class TestMultiLayerPerceptron(unittest.TestCase): 
    def setUp(self): 
        X,y = ds.load_EDM()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.1)

    def test_fit(self): 
        regressor = mor.MultiLayerPerceptron()
        self.assertEqual(regressor.fit(self.X_train,self.y_train)._estimator_type, 'regressor')
    
    def test_predict(self):
        result = mor.MultiLayerPerceptron().fit(self.X_train,self.y_train).predict(self.X_test) 
        self.assertEqual(result.shape, (len(self.X_test),len(self.y_test[0,:]))) 
        self.assertTrue(type(result) is numpy.ndarray) 
        self.assertTrue(result.dtype is numpy.dtype('float32') or result.dtype is numpy.dtype('float64')) 

class TestRegressionTree(unittest.TestCase):
    def setUp(self): 
        X,y = ds.load_EDM()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.1)

    def test_fit(self): 
        regressor = mor.MultiOutputRegressionTree()
        self.assertEqual(regressor.fit(self.X_train,self.y_train)._estimator_type, 'regressor')
    
    def test_predict(self):
        result = mor.MultiOutputRegressionTree().fit(self.X_train,self.y_train).predict(self.X_test) 
        self.assertEqual(result.shape, (len(self.X_test),len(self.y_test[0,:]))) 
        self.assertTrue(type(result) is numpy.ndarray) 
        self.assertTrue(result.dtype is numpy.dtype('float32') or result.dtype is numpy.dtype('float64')) 

class TestNeuronalNet(unittest.TestCase): 
    def setUp(self): 
        X,y = ds.load_EDM()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.1)

    
    def test_predict(self): 
        input_dim = len(self.X_train[0,:])
        target_dim =  len(self.y_train[0,:])

        model = mor.NeuronalNet(input_dim,target_dim,'mean')
        result = mor.NeuronalNetRegressor(model=model,patience=1).fit(self.X_train,self.y_train).predict(self.X_test) 
        self.assertEqual(result.shape, (len(self.X_test),len(self.y_test[0,:]))) 
        self.assertTrue(type(result) is numpy.ndarray) 
        self.assertTrue(result.dtype is numpy.dtype('float32') or result.dtype is numpy.dtype('float64')) 
    '''    result = mor.NeuronalNetRegressor(patience=1,selector='cnn').fit(self.X_train,self.y_train).predict(self.X_test) 
        self.assertEqual(result.shape, (len(self.X_test),len(self.y_test[0,:]))) 
        self.assertTrue(type(result) is numpy.ndarray) 
        self.assertTrue(result.dtype is numpy.dtype('float32') or result.dtype is numpy.dtype('float64'))
    ''' 
    #def test_batch_functionality(self): 

    #def test_save_load(self):

if __name__ == '__main__':
    unittest.main()
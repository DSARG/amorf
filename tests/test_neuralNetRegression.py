import unittest
import framework.neuralNetRegression as nnRegressor
import framework.datasets as ds
from sklearn.model_selection import train_test_split
import numpy


class TestLinearNeuralNet(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)

    def test_predict_without_GPU(self):
        input_dim = len(self.X_train[0, :])
        target_dim = len(self.y_train[0, :])

        model = nnRegressor.Linear_NN_Model(input_dim, target_dim, 'mean')
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1).fit(
            self.X_train, self.y_train)
        
        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64')) 
    
    def test_predict_with_GPU(self):
        input_dim = len(self.X_train[0, :])
        target_dim = len(self.y_train[0, :])

        model = nnRegressor.Linear_NN_Model(input_dim, target_dim, 'mean')
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1,use_gpu=True).fit(
            self.X_train, self.y_train) 
        
        
        result = fittedReg.predict(self.X_test)
        
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, True) 
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    # def test_batch_functionality(self):

    # def test_save_load(self):

class TestConvolutionalNeuralNet(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)

    def test_predict_without_GPU(self):
        input_dim = len(self.X_train[0, :])
        target_dim = len(self.y_train[0, :])

        model = nnRegressor.Convolutional_NN_Model(input_dim, target_dim)
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1).fit(
            self.X_train, self.y_train)
        
        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64')) 
    
    def test_predict_with_GPU(self):
        input_dim = len(self.X_train[0, :])
        target_dim = len(self.y_train[0, :])

        model = nnRegressor.Convolutional_NN_Model(input_dim, target_dim)
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1,use_gpu=True).fit(
            self.X_train, self.y_train) 
        
        
        result = fittedReg.predict(self.X_test)
        
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, True) 
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))
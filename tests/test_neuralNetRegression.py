import unittest
import framework.neuralNetRegression as nnRegressor
import framework.datasets as ds
from sklearn.model_selection import train_test_split
import numpy
import torch
import os


class TestLinearNeuralNet(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)
        self.selectors = ['mean', 'max', 'doubleInput']
        self.input_dim = len(self.X_train[0, :])
        self.target_dim = len(self.y_train[0, :])

    def test_predict_without_GPU(self):

        model = nnRegressor.Linear_NN_Model(
            self.input_dim, self.target_dim, 'mean')
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1).fit(
            self.X_train, self.y_train)

        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    def test_predict_without_GPU_training_limit(self):

        model = nnRegressor.Linear_NN_Model(
            self.input_dim, self.target_dim, 'mean')
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=100,training_limit=1).fit(
            self.X_train, self.y_train) 
        
        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    def test_predict_without_GPU_default_model(self):
        fittedReg = nnRegressor.NeuralNetRegressor(patience=1).fit(
            self.X_train, self.y_train)

        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    def test_predict_with_GPU(self):

        model = nnRegressor.Linear_NN_Model(
            self.input_dim, self.target_dim, 'mean')
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1, use_gpu=True).fit(
            self.X_train, self.y_train)

        result = fittedReg.predict(self.X_test)

        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, True)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    def test_save_load(self):
        model = nnRegressor.Linear_NN_Model(
            self.input_dim, self.target_dim, 'mean')
        reg = nnRegressor.NeuralNetRegressor(
            model=model, patience=1, use_gpu=True)
        reg.save('test')
        self.assertTrue(os.path.exists('test.ckpt'))

        newReg = nnRegressor.NeuralNetRegressor() 
        newReg.load('test.ckpt')
        self.assertEquals(newReg.model.fc1.in_features, self.input_dim)
        self.assertEquals(newReg.model.fc3.out_features, self.target_dim)

    # TODO: add test for scoring mehtod
    def test_score(self): 
        raise NotImplementedError

class TestConvolutionalNeuralNet(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)
        self.input_dim = len(self.X_train[0, :])
        self.target_dim = len(self.y_train[0, :])

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

    def test_predict_without_GPU_default_model(self):
        fittedReg = nnRegressor.NeuralNetRegressor(patience=1).fit(
            self.X_train, self.y_train)

        result = fittedReg.predict(self.X_test)
        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, False)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32') or
                        result.dtype is numpy.dtype('float64'))

    def test_predict_with_GPU(self):
        input_dim = len(self.X_train[0, :])
        target_dim = len(self.y_train[0, :])

        model = nnRegressor.Convolutional_NN_Model(input_dim, target_dim)
        fittedReg = nnRegressor.NeuralNetRegressor(model=model, patience=1, use_gpu=True).fit(
            self.X_train, self.y_train)

        result = fittedReg.predict(self.X_test)

        self.assertEqual(next(fittedReg.model.parameters()).is_cuda, True)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype('float32')
                        or result.dtype is numpy.dtype('float64'))

    def test_save_load(self):
        model = nnRegressor.Convolutional_NN_Model(
            self.input_dim, self.target_dim)
        reg = nnRegressor.NeuralNetRegressor(
            model=model, patience=1, use_gpu=True)
        reg.save('testCNN')
        self.assertTrue(os.path.exists('testCNN.ckpt'))

        newReg = nnRegressor.NeuralNetRegressor() 
        newReg.load('testCNN.ckpt')
        self.assertEquals(newReg.model.input_dim, self.input_dim)
        self.assertEquals(newReg.model.output_dim, self.target_dim)


class TestFullScenarios(unittest.TestCase):
    pass
    # TODO: def Scenario With GPU and With Batch Mechanism - Linear

    # TODO: def Scenaro With GPU and With Batch Mechanis - Convolutional

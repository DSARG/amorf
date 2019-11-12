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

    def test_batch_functionality(self):
        model = nnRegressor.Linear_NN_Model(
            self.input_dim, self.target_dim, 'mean')
        reg = nnRegressor.NeuralNetRegressor(
            model=model, patience=1, use_gpu=True)

        X_train_t, y_train_t = reg.model.convert_train_set_to_tensor(
            self.X_train, self.y_train, 'cpu')
        self.assertEqual(X_train_t.dtype, torch.float)
        self.assertEqual(y_train_t.dtype, torch.float)
        batch_x, batch_y = reg._NeuralNetRegressor__split_training_set_to_batches(
            X_train_t, y_train_t, 10)
        self.assertEqual(len(batch_x), 14)
        self.assertEqual(len(batch_y), 14)
        self.assertEqual(len(batch_x[0]), 10)
        self.assertEqual(len(batch_y[0]), 10)
        batch_x, batch_y = reg._NeuralNetRegressor__split_training_set_to_batches(
            X_train_t, y_train_t, None)
        self.assertEqual(len(batch_x), 1)
        self.assertEqual(len(batch_y), 1)
        self.assertEqual(batch_x[0].shape[0], 138)
        self.assertEqual(batch_x[0].shape[1], 16)
        self.assertEqual(batch_y[0].shape[0], 138)
        self.assertEqual(batch_y[0].shape[1], 2)

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

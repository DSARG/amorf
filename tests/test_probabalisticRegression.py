import os
import numpy
import unittest
from amorf.probabalisticRegression import BayesianNeuralNetworkRegression
import amorf.datasets as ds
from sklearn.model_selection import train_test_split 
import torch


class TestBayesianRegression(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)

    def test_fit_and_predict(self):
        model = BayesianNeuralNetworkRegression(
            patience=5, use_gpu=False, training_limit=1, learning_rate=1e-5, batch_size=2000)

        fitted = model.fit(self.X_train, self.y_train)
        self.assertEqual(fitted.patience, 5)
        self.assertEqual(fitted.batch_size, 2000)
        self.assertEqual(fitted.training_limit, 1)
        self.assertEqual(fitted.learning_rate, 1e-5)
        stds, means = model.predict(self.X_test)
        self.assertEqual(stds.shape, (16, 2))
        self.assertEqual(means.shape, (16, 2))

    def test_fit_and_predict_with_GPU(self):
        if torch.cuda.is_available():
            model = BayesianNeuralNetworkRegression(
                patience=5, use_gpu=True, training_limit=1, learning_rate=1e-5, batch_size=2000)

            fitted = model.fit(self.X_train, self.y_train)
            self.assertEqual(fitted.patience, 5)
            self.assertEqual(fitted.Device, 'cuda:0')
            self.assertEqual(fitted.batch_size, 2000)
            self.assertEqual(fitted.training_limit, 1)
            self.assertEqual(fitted.learning_rate, 1e-5)
            self.assertEqual(next(fitted.net.parameters()).is_cuda, True)

            stds, means = model.predict(self.X_test)
            self.assertEqual(stds.shape, (16, 2))
            self.assertEqual(means.shape, (16, 2))

    def test_load_and_save(self):
        model = BayesianNeuralNetworkRegression(
            patience=5, use_gpu=True, training_limit=1, learning_rate=1e-5, batch_size=2000)

        fitted = model.fit(self.X_train, self.y_train)
        fitted.save('test_bnn')
        self.assertTrue(os.path.exists('test_bnn_model'))
        self.assertTrue(os.path.exists('test_bnn_opt'))
        self.assertTrue(os.path.exists('test_bnn_params'))

        newReg = BayesianNeuralNetworkRegression(
            patience=5, use_gpu=True, training_limit=1, learning_rate=1e-5, batch_size=2000)
        newReg.load('test_bnn')
        stds, means = newReg.predict(self.X_test)
        self.assertEqual(stds.shape, (16, 2))
        self.assertEqual(means.shape, (16, 2))

    def test_score(self):
        model = BayesianNeuralNetworkRegression(
            patience=None, use_gpu=False, training_limit=1, learning_rate=1e-5, batch_size=2000)

        fitted = model.fit(self.X_train, self.y_train)
        score = fitted.score(self.X_test, self.y_test)

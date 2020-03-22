import unittest
from amorf.problemTransformation import AutoEncoderRegression, SingleTargetMethod, _implements_SciKitLearn_API
import amorf.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import numpy


class TestSingleTargetMethod(unittest.TestCase):

    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)
        self.selectors = ['linear', 'kneighbors',
                          'adaboost', 'gradientboost', 'mlp', 'svr', 'xgb']

    def test_correct_assignment(self):
        for selector in self.selectors:
            regressor = SingleTargetMethod(selector)
            self.assertEqual(
                regressor.MORegressor._estimator_type, 'regressor')
        self.assertRaises(ValueError, SingleTargetMethod,
                          'nonexistent_selector')
        self.assertEqual(SingleTargetMethod(
            custom_regressor=RidgeCV()).MORegressor._estimator_type, 'regressor')

    def test_false_assignment(self):
        valid_estimator = RidgeCV()
        invalid_estimator = object()

        with self.assertRaises(Warning):
            SingleTargetMethod(custom_regressor=invalid_estimator)
        with self.assertRaises(ValueError):
            SingleTargetMethod("selector", custom_regressor=invalid_estimator)
        with self.assertRaises(ValueError):
            SingleTargetMethod(valid_estimator)
        with self.assertRaises(ValueError):
            SingleTargetMethod(invalid_estimator)

    def test_fit(self):
        for selector in self.selectors:
            regressor = SingleTargetMethod(selector)
            self.assertEqual(regressor.fit(
                self.X_train, self.y_train)._estimator_type, 'regressor')

    def test_predict(self):
        for selector in self.selectors:
            result = SingleTargetMethod(selector).fit(
                self.X_train, self.y_train).predict(self.X_test)
            self.assertEqual(
                result.shape, (len(self.X_test), len(self.y_test[0, :])))
            self.assertTrue(type(result) is numpy.ndarray)
            self.assertTrue(result.dtype is numpy.dtype(
                'float32') or result.dtype is numpy.dtype('float64'))

    def test_custom_regressor(self):
        valid_estimator = RidgeCV()
        invalid_estimator = object()
        stm = SingleTargetMethod(custom_regressor=valid_estimator)
        self.assertFalse(_implements_SciKitLearn_API(
            invalid_estimator))
        self.assertTrue(_implements_SciKitLearn_API(
            valid_estimator))
        result = stm.fit(
            self.X_train, self.y_train).predict(self.X_test)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype(
            'float32') or result.dtype is numpy.dtype('float64'))

    def test_score(self):
        for selector in self.selectors:
            result = SingleTargetMethod(selector).fit(
                self.X_train, self.y_train)
            score = result.score(self.X_test, self.y_test)


class TestAutoEncoderRegression(unittest.TestCase):

    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)
        self.selectors = ['linear', 'kneighbors',
                          'adaboost', 'gradientboost', 'mlp', 'svr', 'xgb']

    def test_correct_assignment(self):
        for selector in self.selectors:
            regressor = AutoEncoderRegression(selector)
            self.assertEqual(
                regressor.regressor._estimator_type, 'regressor')
        self.assertRaises(ValueError, SingleTargetMethod,
                          'nonexistent_selector')
        self.assertEqual(AutoEncoderRegression(
            custom_regressor=RidgeCV()).regressor._estimator_type, 'regressor')

    def test_false_assignment(self):
        valid_estimator = RidgeCV()
        invalid_estimator = object()

        with self.assertRaises(Warning):
            AutoEncoderRegression(custom_regressor=invalid_estimator)
        with self.assertRaises(ValueError):
            AutoEncoderRegression(
                "selector", custom_regressor=invalid_estimator)
        with self.assertRaises(ValueError):
            AutoEncoderRegression(valid_estimator)
        with self.assertRaises(ValueError):
            AutoEncoderRegression(invalid_estimator)

    def test_fit(self):
        for selector in self.selectors:
            regressor = AutoEncoderRegression(selector)
            self.assertEqual(regressor.fit(
                self.X_train, self.y_train).regressor._estimator_type, 'regressor')

    def test_predict(self):
        for selector in self.selectors:
            result = AutoEncoderRegression(regressor=selector, patience=1, batch_size=10).fit(
                self.X_train, self.y_train).predict(self.X_test)
            self.assertEqual(
                result.shape, (len(self.X_test), len(self.y_test[0, :])))
            self.assertTrue(type(result) is numpy.ndarray)
            self.assertTrue(result.dtype is numpy.dtype(
                'float32') or result.dtype is numpy.dtype('float64'))

    def test_custom_regressor(self):
        valid_estimator = RidgeCV()
        invalid_estimator = object()
        reg = AutoEncoderRegression(custom_regressor=valid_estimator)
        self.assertFalse(_implements_SciKitLearn_API(
            invalid_estimator))
        self.assertTrue(_implements_SciKitLearn_API(
            valid_estimator))
        result = reg.fit(
            self.X_train, self.y_train).predict(self.X_test)
        self.assertEqual(
            result.shape, (len(self.X_test), len(self.y_test[0, :])))
        self.assertTrue(type(result) is numpy.ndarray)
        self.assertTrue(result.dtype is numpy.dtype(
            'float32') or result.dtype is numpy.dtype('float64'))

    def test_score(self):
        for selector in self.selectors:
            result = AutoEncoderRegression(selector).fit(
                self.X_train, self.y_train)
            score = result.score(self.X_test, self.y_test)

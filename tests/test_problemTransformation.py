import unittest
from framework.problemTransformation import SingleTargetMethod
import framework.datasets as ds
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
        stm = SingleTargetMethod()
        self.assertFalse(stm._SingleTargetMethod__implements_SciKitLearn_API(
            object=invalid_estimator))
        self.assertTrue(stm._SingleTargetMethod__implements_SciKitLearn_API(
            object=valid_estimator))

import unittest
import numpy as np
import amorf.multiOutputSVR as moSVR
import amorf.datasets as ds
from sklearn.model_selection import train_test_split


class TestMLSSVR(unittest.TestCase):
    def setUp(self):
        X, y = ds.EDM().get_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1)
        self.input_dim = len(self.X_train[0, :])
        self.target_dim = len(self.y_train[0, :])

    def test_initialize_and_fit(self):
        model = moSVR.MLSSVR(1, 1, 'poly')
        fitted_Model = model.fit(self.X_train, self.y_train, 0.1, 0.1)
        self.assertEqual(len(model.alpha), len(self.X_train))
        self.assertEqual(len(model.alpha[0]), self.target_dim)
        self.assertEqual(len(self.y_test[0, :]), np.size(model.b))

    def test_predict(self):
        model = moSVR.MLSSVR(1, 1, 'poly')
        y_pred = model.fit(self.X_train, self.y_train, 0.1, 0.1)\
            .predict(self.X_test, self.X_train)
        self.assertEqual(y_pred.shape, self.y_test.shape)
        self.assertEqual(type(y_pred), type(self.y_test))

    def test_selectors(self):
        selectors = ['linear', 'poly', 'rbf', 'erbf', 'sigmoid']
        for selector in selectors:
            model = moSVR.MLSSVR(1, 1, selector)
            self.assertEqual(model.kernel_selector, selector)

        with self.assertRaises(ValueError):
            model = moSVR.MLSSVR(1, 1, 'nonexistent_selector')

    def test_score(self):
        model = moSVR.MLSSVR(1, 1, 'poly')
        fitted_Model = model.fit(self.X_train, self.y_train, 0.1, 0.1)
        self.assertEqual(len(model.alpha), len(self.X_train))
        self.assertEqual(len(model.alpha[0]), self.target_dim)
        self.assertEqual(len(self.y_test[0, :]), np.size(model.b)) 
        result = fitted_Model.score(self.X_test, self.X_train, self.y_test)
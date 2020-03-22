# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import numpy

import amorf.datasets as datasets


class TestEDMDataset(unittest.TestCase):

    def test_getNumpy_dimensions(self):
        X, y = datasets.EDM().get_numpy()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X[0, :]), 16)
        self.assertEqual(len(y[0, :]), 2)

    def test_getNumpy_type(self):
        X, y = datasets.EDM().get_numpy()
        self.assertTrue(type(X) is numpy.ndarray)
        self.assertTrue(X.dtype is numpy.dtype('float32'))
        self.assertTrue(type(y) is numpy.ndarray)
        self.assertTrue(y.dtype is numpy.dtype('float32'))


class TestRF1Dataset(unittest.TestCase):

    def test_getNumpy_dimensions(self):
        X, y = datasets.RiverFlow1().get_numpy()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X[0, :]), 64)
        self.assertEqual(len(y[0, :]), 8)

    def test_getNumpy_type(self):
        X, y = datasets.RiverFlow1().get_numpy()
        self.assertTrue(type(X) is numpy.ndarray)
        self.assertTrue(X.dtype is numpy.dtype('float32'))
        self.assertTrue(type(y) is numpy.ndarray)
        self.assertTrue(y.dtype is numpy.dtype('float32'))


class TestWQDataset(unittest.TestCase):

    def test_getNumpy_dimensions(self):
        X, y = datasets.WaterQuality().get_numpy()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X[0, :]), 14)
        self.assertEqual(len(y[0, :]), 16)

    def test_getNumpy_type(self):
        X, y = datasets.WaterQuality().get_numpy()
        self.assertTrue(type(X) is numpy.ndarray)
        self.assertTrue(X.dtype is numpy.dtype('float32'))
        self.assertTrue(type(y) is numpy.ndarray)
        self.assertTrue(y.dtype is numpy.dtype('float32'))


class TestTransparentConductors(unittest.TestCase):

    def test_getNumpy_dimensions(self):
        X, y = datasets.TransparentConductors().get_numpy()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X[0, :]), 12)
        self.assertEqual(len(y[0, :]), 2)

    def test_getNumpy_type(self):
        X, y = datasets.TransparentConductors().get_numpy()
        self.assertTrue(type(X) is numpy.ndarray)
        self.assertTrue(X.dtype is numpy.dtype('float32'))
        self.assertTrue(type(y) is numpy.ndarray)
        self.assertTrue(y.dtype is numpy.dtype('float32'))


if __name__ == '__main__':
    unittest.main()

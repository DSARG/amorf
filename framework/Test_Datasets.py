import unittest 
import framework.datasets as datasets  
import numpy

class TestWQDataset(unittest.TestCase):

    def test_dimensions(self): 
        X,y = datasets.load_WQ()
        self.assertEqual(len(X),len(y)) 
        self.assertEqual(len(X[0,:]),14)
        self.assertEqual(len(y[0,:]),16)  
    
    def test_type(self): 
        X,y = datasets.load_WQ() 
        self.assertTrue (type(X) is numpy.ndarray) 
        self.assertTrue(X.dtype is numpy.dtype('float64')) 
        self.assertTrue(type(y) is numpy.ndarray) 
        self.assertTrue(y.dtype is numpy.dtype('float64')) 

class TestRF1Dataset(unittest.TestCase):

    def test_dimensions(self): 
        X,y = datasets.load_RF1()
        self.assertEqual(len(X),len(y)) 
        self.assertEqual(len(X[0,:]),64)
        self.assertEqual(len(y[0,:]),8) 

    def test_type(self): 
        X,y = datasets.load_RF1() 
        self.assertTrue (type(X) is numpy.ndarray) 
        self.assertTrue(X.dtype is numpy.dtype('float32')) 
        self.assertTrue(type(y) is numpy.ndarray) 
        self.assertTrue(y.dtype is numpy.dtype('float32')) 

class TestEDMDataset(unittest.TestCase):

    def test_dimensions(self): 
        X,y = datasets.load_EDM()
        self.assertEqual(len(X),len(y)) 
        self.assertEqual(len(X[0,:]),16)
        self.assertEqual(len(y[0,:]),2)  
    
    def test_type(self): 
        X,y = datasets.load_EDM() 
        self.assertTrue (type(X) is numpy.ndarray) 
        self.assertTrue(X.dtype is numpy.dtype('float32')) 
        self.assertTrue(type(y) is numpy.ndarray) 
        self.assertTrue(y.dtype is numpy.dtype('float32'))
        
if __name__ == '__main__':
    unittest.main()
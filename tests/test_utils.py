import unittest
import amorf.utils as utils
from amorf.neuralNetRegression import Linear_NN_Model


class TestEarlyStopping(unittest.TestCase):

    def setUp(self):
        self.model = Linear_NN_Model(5, 5)

    def test_constantIncrease(self):
        patience = 3
        stopping = utils.EarlyStopping(patience)
        a = [1, 2, 3, 4, 5, 6]
        counter = 1
        trueCounter = 0
        falseCounter = 0
        for value in a:
            if(counter >= patience):
                self.assertTrue(stopping.stop(value, self.model))
                trueCounter += 1
            else:
                self.assertFalse(stopping.stop(value, self.model))
                falseCounter += 1
            counter += 1
        self.assertTrue(falseCounter is 2)
        self.assertTrue(trueCounter is 4)

    def test_constantDecreas(self):
        patience = 3
        stopping = utils.EarlyStopping(patience)
        a = [6, 5, 4, 3, 2, 1]
        counter = 0
        falseCounter = 0
        for value in a:
            self.assertFalse(stopping.stop(value, self.model))
            falseCounter += 1
            counter += 1
        self.assertTrue(counter is 6)
        self.assertTrue(falseCounter is 6)

    def test_constantValues(self):
        patience = 3
        stopping = utils.EarlyStopping(patience)
        a = [3, 3, 3, 3, 3, 3]
        counter = 0
        falseCounter = 0
        for value in a:
            self.assertFalse(stopping.stop(value, self.model))
            falseCounter += 1
            counter += 1
        self.assertTrue(counter is 6)
        self.assertTrue(falseCounter is 6)

    def test_alternatingIncrease(self):
        patience = 3
        stopping = utils.EarlyStopping(patience)
        a = [3, 4, 2, 3, 3, 3]
        counter = 0
        falseCounter = 0
        for value in a:
            self.assertFalse(stopping.stop(value, self.model))
            falseCounter += 1
            counter += 1
        self.assertTrue(counter is 6)
        self.assertTrue(falseCounter is 6)


if __name__ == '__main__':
    unittest.main()

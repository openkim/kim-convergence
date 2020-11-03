"""Test mser_m module."""
import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestMSERModule(unittest.TestCase):
    """Test mser_m module components."""

    def test_mser_m(self):
        """Test mser_m function."""
        n = 100
        x = np.ones(n)

        # x is not one dimensional
        self.assertRaises(CVGError, cvg.mser_m,
                          x.reshape(5, 20))

        # constant data sets
        x = np.ones(n)
        truncated, truncated_i = cvg.mser_m(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i == 0)

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CVGError, cvg.mser_m, x)

        # inf in the input
        if n > 10:
            x[10] = np.inf

        self.assertRaises(CVGError, cvg.mser_m, x)

        if n > 10:
            x[10] = -np.inf

        self.assertRaises(CVGError, cvg.mser_m, x)

        # one value array
        n = 1
        x = np.ones(n) * 10
        truncated, truncated_i = cvg.mser_m(x)
        self.assertFalse(truncated)

        # const value array less than batch_size
        n = 4
        x = np.ones(n)
        truncated, truncated_i = cvg.mser_m(x)
        self.assertFalse(truncated)

        n = 100
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        # invalid int ignore_end_batch
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=0)
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=-1)

        # invalid float ignore_end_batch
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=0.0)
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=1.0)
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=-0.1)
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=1.1)

        # invalid ignore_end_batch
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch="None")
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch="1")

        # invalid ignore_end_batch
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=120)
        self.assertRaises(CVGError, cvg.mser_m,
                          x, ignore_end_batch=100)

        # Create synthetic data
        n = 1000
        x = np.arange(10.)
        _x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)
        x = np.concatenate((x, _x))

        truncated, truncated_i = cvg.mser_m(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)

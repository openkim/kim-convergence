"""Test UCL mser_m module."""
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestUCLMSERModule(unittest.TestCase):
    """Test UCL mser_m module components."""

    def test_mser_m(self):
        """Test mser_m function."""
        n = 100
        x = np.ones(n)

        # x is not one dimensional
        self.assertRaises(CVGError, cr.mser_m,
                          x.reshape(5, 20))

        # constant data sets
        x = np.ones(n)
        truncated, truncated_i = cr.mser_m(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i == 0)

        # nan in the input
        if n > 10:
            x[10] = np.nan

        self.assertRaises(CVGError, cr.mser_m, x)

        # inf in the input
        if n > 10:
            x[10] = np.inf

        self.assertRaises(CVGError, cr.mser_m, x)

        if n > 10:
            x[10] = -np.inf

        self.assertRaises(CVGError, cr.mser_m, x)

        # const value array less than batch_size
        # one value array
        n = 1
        x = np.ones(n) * 10
        truncated, truncated_i = cr.mser_m(x)
        self.assertFalse(truncated)
        self.assertEqual(truncated_i, 0)

        n = 4
        x = np.ones(n)
        truncated, truncated_i = cr.mser_m(x)
        self.assertFalse(truncated)
        self.assertEqual(truncated_i, 0)

        rng = np.random.RandomState(12345)

        n = 100
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        # invalid int ignore_end
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=0)
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=-1)

        # invalid float ignore_end
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=0.0)
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=1.0)
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=-0.1)
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=1.1)

        # invalid ignore_end
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end="None")
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end="1")

        # invalid ignore_end
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=120)
        self.assertRaises(CVGError, cr.mser_m,
                          x, ignore_end=100)

        # Create synthetic data
        rng = np.random.RandomState(12345)

        n = 1000
        x = np.arange(10.)
        y = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        x = np.concatenate((x, y))

        truncated, truncated_i = cr.mser_m(x)
        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)

        x = np.arange(20, 10, -1)
        x = np.concatenate((x, y))
        truncated, truncated_i = cr.mser_m(x)

        self.assertTrue(truncated)
        self.assertTrue(truncated_i >= 10)


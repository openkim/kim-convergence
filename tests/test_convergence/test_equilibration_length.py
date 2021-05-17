"""Test equilibration_length module."""
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestEquilibrationLengthModule(unittest.TestCase):
    """Test equilibration_length module components."""

    def test_estimate_equilibration_length(self):
        """Test estimate_equilibration_length function."""
        x = np.arange(100.)
        # si is not an str nor a valid si methods
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si=cr.statistical_inefficiency)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si=1.0)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si=1)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si=True)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='si')

        # x is not one dimensional
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x.reshape(5, 20))

        # constant data sets
        x = np.ones(100)
        n, si = cr.estimate_equilibration_length(x)
        self.assertTrue(n == 0)
        self.assertTrue(si == x.size)

        rng = np.random.RandomState(12345)

        x = np.ones(100) * 10 + (rng.random_sample(100) - 0.5)

        # invalid int ignore_end
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=0)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=-1)

        # invalid float ignore_end
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=0.0)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=1.0)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=-0.1)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=1.1)

        # invalid ignore_end
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end="None")
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end="1")

        # invalid ignore_end
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=120)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, ignore_end=100)

        # insufficient data points
        n = 1
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cr.estimate_equilibration_length, x)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_r_statistical_inefficiency')
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_split_r_statistical_inefficiency')
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_split_statistical_inefficiency')

        n = 3
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_r_statistical_inefficiency')

        n = 7
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_split_r_statistical_inefficiency')

        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, si='geyer_split_statistical_inefficiency')

        # invalid nskip
        n = 100
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, nskip=1.0)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, nskip=10.)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, nskip=-10.)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, nskip=0)
        self.assertRaises(CVGError, cr.estimate_equilibration_length,
                          x, nskip=-1)

        rng = np.random.RandomState(12345)
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)
        y = np.concatenate(
            (np.arange(n // 10)[::-1] / float(n // 10),
             np.zeros(n - n // 10)))

        x += y

        n1, si1 = cr.estimate_equilibration_length(x, fft=True)
        n2, si2 = cr.estimate_equilibration_length(x, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(x, nskip=2, fft=True)
        n2, si2 = cr.estimate_equilibration_length(x, nskip=2, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, fft=True, minimum_correlation_time=2)
        n2, si2 = cr.estimate_equilibration_length(
            x, fft=False, minimum_correlation_time=2)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(x, fft=True, ignore_end=25)
        n2, si2 = cr.estimate_equilibration_length(
            x, fft=False, ignore_end=25)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si='geyer_r_statistical_inefficiency', fft=True)
        n2, si2 = cr.estimate_equilibration_length(
            x, si='geyer_r_statistical_inefficiency', fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si='geyer_split_r_statistical_inefficiency', fft=True)
        n2, si2 = cr.estimate_equilibration_length(
            x, si='geyer_split_r_statistical_inefficiency', fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cr.estimate_equilibration_length(
            x, si='geyer_split_statistical_inefficiency', fft=True)
        n2, si2 = cr.estimate_equilibration_length(
            x, si='geyer_split_statistical_inefficiency', fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        # there is at least one value in the input array
        # which is non-finite or not-number
        n = 1000
        x = np.ones(n) * 10 + (rng.random_sample(n) - 0.5)

        x[2] = np.inf

        self.assertRaises(CVGError,
                          cr.estimate_equilibration_length, x)

        x[2] = np.nan

        self.assertRaises(CVGError,
                          cr.estimate_equilibration_length, x)

        x[2] = np.NINF

        self.assertRaises(CVGError,
                          cr.estimate_equilibration_length, x)

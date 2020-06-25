import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class Equilibration_LengthModule:
    """Test equilibration_length module components."""

    def test_estimate_equilibration_length(self):
        """Test estimate_equilibration_length function."""
        x = np.arange(100.)
        # si is not an str nor a valid si methods
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si=cvg.statistical_inefficiency)

        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si=None)

        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='si')

        # x is not one dimensional
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x.reshape(5, 20))

        # constant data sets
        x = np.ones(100)
        n, si = cvg.estimate_equilibration_length(x)
        self.assertTrue(n == 0)
        self.assertTrue(si == 1.0)

        x = np.ones(100) * 10 + (np.random.random_sample(100) - 0.5)

        # invalid int ignore_end
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=0)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=-1)

        # invalid float ignore_end
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=0.0)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=1.0)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=-0.1)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=1.1)

        # invalid ignore_end
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end="None")
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end="1")

        # invalid ignore_end
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=120)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, ignore_end=100)

        # insufficient data points
        n = 1
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cvg.estimate_equilibration_length, x)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='r_statistical_inefficiency')
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='split_r_statistical_inefficiency')
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='split_statistical_inefficiency')

        n = 3
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='r_statistical_inefficiency')

        n = 7
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='split_r_statistical_inefficiency')

        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, si='split_statistical_inefficiency')

        # invalid nskip
        n = 100
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, nskip=1.0)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, nskip=10.)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, nskip=-10.)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, nskip=0)
        self.assertRaises(CVGError, cvg.estimate_equilibration_length,
                          x, nskip=-1)

        n = 1000
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        n1, si1 = cvg.estimate_equilibration_length(x, fft=True)
        n2, si2 = cvg.estimate_equilibration_length(x, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cvg.estimate_equilibration_length(x, nskip=2, fft=True)
        n2, si2 = cvg.estimate_equilibration_length(x, nskip=2, fft=False)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cvg.estimate_equilibration_length(x, fft=True, mct=2)
        n2, si2 = cvg.estimate_equilibration_length(x, fft=False, mct=2)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n1, si1 = cvg.estimate_equilibration_length(x, fft=True, ignore_end=25)
        n2, si2 = cvg.estimate_equilibration_length(
            x, fft=False, ignore_end=25)

        self.assertTrue(n1 == n2)
        self.assertAlmostEqual(si1, si2, places=12)

        n = 1000
        x = np.ones(n) * 10 + (np.random.random_sample(n) - 0.5)

        x[2] = np.inf

        self.assertRaises(CVGError,
                          cvg.estimate_equilibration_length, x)

        x[2] = np.nan

        self.assertRaises(CVGError,
                          cvg.estimate_equilibration_length, x)

        x[2] = np.NINF

        self.assertRaises(CVGError,
                          cvg.estimate_equilibration_length, x)


class TestEquilibration_LengthModule(Equilibration_LengthModule, unittest.TestCase):
    pass

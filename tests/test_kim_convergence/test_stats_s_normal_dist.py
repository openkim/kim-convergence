"""Test stats s_normal_inv_cdf module."""

import unittest
from math import inf, nan
import numpy as np

try:
    import kim_convergence as cr
except Exception as e:  # noqa: BLE001  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module") from e


class TestStatsSNormalDistModule(unittest.TestCase):
    """Test stats s_normal_inv_cdf module components."""

    def test_s_normal_inv_cdf(self):
        """Test s_normal_inv_cdf function."""
        prob = np.array(
            [
                0.5,
                0.95,
                0.995,
                0.9995,
                0.99995,
                0.999995,
                0.9999995,
                0.99999995,
                0.999999995,
                0.9999999995,
            ],
            dtype=np.float64,
        )

        _ppf = np.array(
            [0.000, 1.645, 2.576, 3.291, 3.891, 4.417, 4.892, 5.327, 5.731, 6.109],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array(
            [
                0.75,
                0.95,
                0.9975,
                0.99975,
                0.999975,
                0.9999975,
                0.99999975,
                0.999999975,
                0.9999999975,
                0.99999999975,
            ],
            dtype=np.float64,
        )

        _ppf = np.array(
            [
                0.67448975,
                1.64485363,
                2.80703377,
                3.4807564,
                4.05562698,
                4.56478773,
                5.02631284,
                5.45131044,
                5.84717215,
                6.21910456,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array(
            [
                0.9,
                0.99,
                0.999,
                0.9999,
                0.99999,
                0.999999,
                0.9999999,
                0.99999999,
                0.999999999,
                0.9999999999,
            ],
            dtype=np.float64,
        )

        _ppf = np.array(
            [1.282, 2.326, 3.090, 3.719, 4.265, 4.753, 5.199, 5.612, 5.998, 6.361],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array(
            [
                0.5,
                0.05,
                0.005,
                0.0005,
                5e-05,
                4.9999999999999996e-06,
                5e-07,
                5e-08,
                5e-09,
                5e-10,
            ],
            dtype=np.float64,
        )

        _ppf = np.array(
            [0.000, 1.645, 2.576, 3.291, 3.891, 4.417, 4.892, 5.327, 5.731, 6.109],
            dtype=np.float64,
        )

        ppf = -np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array(
            [
                0.25,
                0.025,
                0.0025,
                0.00025,
                2.5e-05,
                2.4999999999999998e-06,
                2.5e-07,
                2.5e-08,
                2.5e-09,
                2.5e-10,
            ],
            dtype=np.float64,
        )

        _ppf = np.array(
            [0.674, 1.960, 2.807, 3.481, 4.056, 4.565, 5.026, 5.451, 5.847, 6.219],
            dtype=np.float64,
        )

        ppf = -np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array(
            [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10],
            dtype=np.float64,
        )

        _ppf = np.array(
            [1.282, 2.326, 3.090, 3.719, 4.265, 4.753, 5.199, 5.612, 5.998, 6.361],
            dtype=np.float64,
        )

        ppf = -np.array(list(map(cr.s_normal_inv_cdf, prob)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        self.assertTrue(cr.s_normal_inv_cdf(0.0) == -inf)
        self.assertTrue(cr.s_normal_inv_cdf(0) == -inf)

        self.assertTrue(cr.s_normal_inv_cdf(1.0) == inf)
        self.assertTrue(cr.s_normal_inv_cdf(1) == inf)

        self.assertTrue(cr.s_normal_inv_cdf(-0.1) is nan)
        self.assertTrue(cr.s_normal_inv_cdf(-1.0) is nan)
        self.assertTrue(cr.s_normal_inv_cdf(-10) is nan)
        self.assertTrue(cr.s_normal_inv_cdf(1.1) is nan)
        self.assertTrue(cr.s_normal_inv_cdf(10) is nan)

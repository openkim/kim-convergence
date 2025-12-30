"""Test stats t_dist module."""

import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception:  # noqa: BLE001  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module")


class TestStatsTDistModule(unittest.TestCase):
    """Test stats t_dist module components."""

    def test_t_cdf(self):
        """Test t_cdf function."""
        df = 2.74335149908

        v = cr.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.001, places=3)

        v = cr.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.5, places=3)

        v = cr.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.999, places=3)

        df = 5

        v = cr.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.001, places=3)

        v = cr.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.5, places=3)

        v = cr.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.999, places=3)

        df = 17.5

        v = cr.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.001, places=3)

        v = cr.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.5, places=3)

        v = cr.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.999, places=3)

        df = 25

        v = cr.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.001, places=3)

        v = cr.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.5, places=3)

        v = cr.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cr.t_cdf(v, df), 0.999, places=3)

    def test_t_inv_cdf(self):
        """Test t_inv_cdf function."""
        prob = np.array(
            [
                75.0e-2,
                80.0e-2,
                85.0e-2,
                90.0e-2,
                95.0e-2,
                97.5e-2,
                99.0e-2,
                99.5e-2,
                99.75e-2,
                99.9e-2,
                99.95e-2,
            ],
            dtype=np.float64,
        )

        n = prob.size

        _ppf = np.array(
            [
                1.000,
                1.376,
                1.963,
                3.078,
                6.314,
                12.71,
                31.82,
                63.66,
                127.3,
                318.3,
                636.6,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [1] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=1)

        _ppf = np.array(
            [
                0.711,
                0.896,
                1.119,
                1.415,
                1.895,
                2.365,
                2.998,
                3.499,
                4.029,
                4.785,
                5.408,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [7] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array(
            [
                0.687,
                0.860,
                1.064,
                1.325,
                1.725,
                2.086,
                2.528,
                2.845,
                3.153,
                3.552,
                3.850,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [20] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array(
            [
                0.683,
                0.854,
                1.055,
                1.311,
                1.699,
                2.045,
                2.462,
                2.756,
                3.038,
                3.396,
                3.659,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [29] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array(
            [
                0.679,
                0.848,
                1.045,
                1.296,
                1.671,
                2.000,
                2.390,
                2.660,
                2.915,
                3.232,
                3.460,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [60] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array(
            [
                0.677,
                0.845,
                1.042,
                1.290,
                1.660,
                1.984,
                2.364,
                2.626,
                2.871,
                3.174,
                3.390,
            ],
            dtype=np.float64,
        )

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [100] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array([90e-2, 95e-2, 97.5e-2, 99.5e-2], dtype=np.float64)

        n = prob.size

        _ppf = np.array([1.88562, 2.91999, 4.30265, 9.92484], dtype=np.float64)

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [2] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.53321, 2.13185, 2.77645, 4.60409], dtype=np.float64)

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [4] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.47588, 2.01505, 2.57058, 4.03214], dtype=np.float64)

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [5] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.37218, 1.81246, 2.22814, 3.16927], dtype=np.float64)

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [10] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.31042, 1.69726, 2.04227, 2.75000], dtype=np.float64)

        ppf = np.array(list(map(cr.t_inv_cdf, prob, [30] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

    def test_t_interval(self):
        """Test t_interval function."""

        # scipy reference value
        ci_ref = (-12.706204736, 12.706204736)
        ci_l, ci_u = cr.t_interval(95.0e-2, 1)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-1.983971518, 1.983971518)
        ci_l, ci_u = cr.t_interval(95.0e-2, 100)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-1.960201239, 1.960201239)
        ci_l, ci_u = cr.t_interval(95.0e-2, 10000)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-25.45169957, 25.451699579)
        ci_l, ci_u = cr.t_interval(97.5e-2, 1)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-2.633766915, 2.633766915)
        ci_l, ci_u = cr.t_interval(97.5e-2, 10)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-2.241740325, 2.241740325)
        ci_l, ci_u = cr.t_interval(97.5e-2, 10000)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-63.656741162, 63.656741162)
        ci_l, ci_u = cr.t_interval(99e-2, 1)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-3.169272667, 3.169272667)
        ci_l, ci_u = cr.t_interval(99e-2, 10)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

        ci_ref = (-2.576321046, 2.576321046)
        ci_l, ci_u = cr.t_interval(99e-2, 10000)

        self.assertAlmostEqual(ci_l, ci_ref[0], places=7)
        self.assertAlmostEqual(ci_u, ci_ref[1], places=7)

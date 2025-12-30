"""Test scale module."""

from math import isclose
import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception:  # noqa: BLE001  # intentional catch-all
    raise RuntimeError("Failed to import `kim-convergence` utility module")

from kim_convergence import CRError


class TestScaleModule(unittest.TestCase):
    """Test scale module components."""

    # Make some data
    rng = np.random.RandomState(0)
    n_samples = 1000
    offset = rng.uniform(-1, 1)
    scale = rng.uniform(1, 10)
    x = rng.randn(n_samples) * scale + offset

    def test_minmax_scale(self):
        """Test minmax_scale function."""
        x = np.array([0, 1, 2, 3, 4, 5, 10], dtype=np.float64)
        scaled_x = cr.minmax_scale(x)
        self.assertTrue(np.allclose(scaled_x, x / 10.0))

        mms = cr.MinMaxScale()
        scaled_x = mms.scale(x)
        self.assertTrue(np.allclose(scaled_x, x / 10.0))
        inverse_scaled_x = mms.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        scaled_x = mms.scale(self.x)
        inverse_scaled_x = mms.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, self.x))

        x = [-1.0, 3.0]
        scaled_x = mms.scale(x)
        inverse_scaled_x = mms.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        x = [-1.0, 3.0, 100.0]
        scaled_x = mms.scale(x)
        inverse_scaled_x = mms.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        x = [-1.0, 3.0, 100.0, np.nan]
        self.assertRaises(CRError, mms.scale, x)

        x = [-1.0, np.inf, 3.0, 100.0]
        self.assertRaises(CRError, mms.scale, x)

        self.assertRaises(CRError, mms.scale, self.x.reshape((2, -1)))

    def test_translate_scale(self):
        """Test translate_scale function."""
        x = [1.0, 2.0, 2.0, 2.0, 3.0]
        scaled_x = cr.translate_scale(x)
        scaled_x_ = [0.0, 1.0, 1.0, 1.0, 2.0]
        self.assertTrue(np.allclose(scaled_x, scaled_x_))

        tsc = cr.TranslateScale()
        scaled_x = tsc.scale(x)
        self.assertTrue(np.allclose(scaled_x, scaled_x_))
        inverse_scaled_x = tsc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        scaled_x = tsc.scale(self.x)
        inverse_scaled_x = tsc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, self.x))

        self.assertRaises(CRError, tsc.scale, self.x.reshape((2, -1)))

    def test_standard_scale(self):
        """Test standard_scale function."""
        ssc = cr.StandardScale()

        scaled_x = ssc.scale(self.x)

        mean_ = np.mean(scaled_x)
        std_ = np.std(scaled_x)

        self.assertTrue(isclose(mean_, 0, abs_tol=1e-14))
        self.assertTrue(isclose(std_, 1, rel_tol=1e-14))

        inverse_scaled_x = ssc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, self.x))

        x = [4.0, 1.0, -2.0]
        scaled_x = ssc.scale(x)
        inverse_scaled_x = ssc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        self.assertRaises(CRError, ssc.scale, self.x.reshape((2, -1)))

    def test_maxabs_scale(self):
        """Test maxabs_scale function."""
        msc = cr.MaxAbsScale()

        scaled_x = msc.scale(self.x)
        inverse_scaled_x = msc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, self.x))

        x = [4.0, 1.0, -9.0]
        scaled_x = msc.scale(x)
        inverse_scaled_x = msc.inverse(scaled_x)
        self.assertTrue(np.allclose(inverse_scaled_x, x))

        self.assertRaises(CRError, msc.scale, self.x.reshape((2, -1)))

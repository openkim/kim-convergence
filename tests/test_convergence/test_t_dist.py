import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class T_DistModule:
    """Test t_dist module components."""

    def test_t_cdf(self):
        """Test t_cdf function."""
        df = 2.74335149908

        v = cvg.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.001, places=3)

        v = cvg.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.5, places=3)

        v = cvg.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.999, places=3)

        df = 5

        v = cvg.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.001, places=3)

        v = cvg.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.5, places=3)

        v = cvg.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.999, places=3)

        df = 17.5

        v = cvg.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.001, places=3)

        v = cvg.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.5, places=3)

        v = cvg.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.999, places=3)

        df = 25

        v = cvg.t_inv_cdf(0.001, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.001, places=3)

        v = cvg.t_inv_cdf(0.5, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.5, places=3)

        v = cvg.t_inv_cdf(0.999, df)
        self.assertAlmostEqual(cvg.t_dist.t_cdf(v, df), 0.999, places=3)

    def test_t_inv_cdf(self):
        """Test t_inv_cdf function."""
        prob = np.array([75e-2, 80e-2, 85e-2, 90e-2, 95e-2, 97.5e-2,
                         99e-2, 99.5e-2, 99.75e-2, 99.9e-2, 99.95e-2],
                        dtype=np.float64)

        n = prob.size

        _ppf = np.array([1.000, 1.376, 1.963, 3.078, 6.314,
                         12.71, 31.82, 63.66, 127.3, 318.3, 636.6],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [1] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=1)

        _ppf = np.array([0.711, 0.896, 1.119, 1.415, 1.895, 2.365,
                         2.998, 3.499, 4.029, 4.785, 5.408],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [7] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array([0.687, 0.860, 1.064, 1.325, 1.725, 2.086,
                         2.528, 2.845, 3.153, 3.552, 3.850],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [20] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array([0.683, 0.854, 1.055, 1.311, 1.699, 2.045,
                         2.462, 2.756, 3.038, 3.396, 3.659],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [29] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array([0.679, 0.848, 1.045, 1.296, 1.671,
                         2.000, 2.390, 2.660, 2.915, 3.232, 3.460],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [60] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        _ppf = np.array([0.677, 0.845, 1.042, 1.290, 1.660, 1.984,
                         2.364, 2.626, 2.871, 3.174, 3.390],
                        dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [100] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=3)

        prob = np.array([90e-2, 95e-2, 97.5e-2, 99.5e-2], dtype=np.float64)

        n = prob.size

        _ppf = np.array([1.88562, 2.91999, 4.30265, 9.92484], dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [2] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.53321, 2.13185, 2.77645, 4.60409], dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [4] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.47588, 2.01505, 2.57058, 4.03214], dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [5] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.37218, 1.81246, 2.22814, 3.16927], dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [10] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)

        _ppf = np.array([1.31042, 1.69726, 2.04227, 2.75000], dtype=np.float64)

        ppf = np.array(
            list(map(cvg.t_inv_cdf, prob, [30] * n)), dtype=np.float64)

        for p, _p in zip(ppf, _ppf):
            self.assertAlmostEqual(p, _p, places=5)


class TestT_DistModule(T_DistModule, unittest.TestCase):
    pass

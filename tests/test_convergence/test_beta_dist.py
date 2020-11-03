"""Test beta_dist module."""
import unittest
# from math import gamma

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')


class TestBeta_DistModule(unittest.TestCase):
    """Test beta_dist module components."""

    def test_beta(self):
        """Test beta function."""
        self.assertEqual(cvg.beta_dist.beta(1, 1), 1.0)

        # self.assertAlmostEqual(
        #     cvg.beta_dist.beta(-100.3, 1.e-200), gamma(1.e-200), 3)

        self.assertAlmostEqual(cvg.beta_dist.beta(0.0342, 171),
                               24.070498359873497, places=10)

    def test_betacf(self):
        """Test betacf function."""
        self.assertAlmostEqual(cvg.beta_dist.betacf(
            0.1, 0.1, 0.1), 1.019300784881282, places=10)

    def test_betai(self):
        """Test betai function."""
        self.assertEqual(cvg.beta_dist.betai(1, 1, 1), 1.0)

        self.assertAlmostEqual(cvg.beta_dist.betai(0.0342, 171, 1e-10),
                               0.55269916901806648, places=10)

    def test_betai_cdf_ccdf(self):
        """Test betai_cdf_ccdf function."""
        df = 2.74335149908

        cdf, ccdf = cvg.beta_dist.betai_cdf_ccdf(0.5 * df, 0.5, 0.001)
        self.assertAlmostEqual(cdf, 3.381367395485997e-05, places=10)
        self.assertAlmostEqual(ccdf, 0.9999661863260452, places=10)

        cdf, ccdf = cvg.beta_dist.betai_cdf_ccdf(0.5 * df, 0.5, 0.5)
        self.assertAlmostEqual(cdf, 0.20465790631037784, places=10)
        self.assertAlmostEqual(ccdf, 0.7953420936896222, places=10)

        cdf, ccdf = cvg.beta_dist.betai_cdf_ccdf(0.5 * df, 0.5, 0.999)
        self.assertAlmostEqual(cdf, 0.961786078269304, places=10)
        self.assertAlmostEqual(ccdf, 0.03821392173069604, places=10)

    def test_betai_cdf(self):
        """Test betai_cdf function."""
        df = 2.74335149908

        self.assertAlmostEqual(cvg.beta_dist.betai_cdf(
            0.5 * df, 0.5, 0.001), 3.381367395485997e-05, places=10)

        self.assertAlmostEqual(cvg.beta_dist.betai_cdf(
            0.5 * df, 0.5, 0.5), 0.20465790631037784, places=10)

        self.assertAlmostEqual(cvg.beta_dist.betai_cdf(
            0.5 * df, 0.5, 0.999), 0.961786078269304, places=10)

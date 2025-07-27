"""Test stats non-normal test module."""
import unittest
import numpy as np

try:
    import kim_convergence as cr
except Exception:  # noqa: BLE001  # intentional catch-all
    raise RuntimeError('Failed to import `kim-convergence` utility module')

from kim_convergence import CRError


class TestStatsNonNormalTestModule(unittest.TestCase):
    """Test stats nonnormal_tset module components."""

    def test_check_population_cdf_args(self):
        """Test check_population_cdf_args function."""
        population_cdf = 'unknown'
        population_args = ()
        self.assertRaises(CRError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_cdf = 'alpha'
        population_args = ()
        self.assertRaises(CRError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_cdf = 'alpha'
        population_args = (2, 2)
        self.assertRaises(CRError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_args = ()
        self.assertRaises(CRError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        for cdf in cr.ContinuousDistributions.keys():
            args = np.arange(
                cr.ContinuousDistributionsNumberOfRequiredArguments[cdf] + 1)
            self.assertRaises(CRError, cr.check_population_cdf_args,
                              population_cdf=cdf,
                              population_args=args)

    def test_get_distribution_stats(self):
        """Test get_distribution_stats function."""
        population_cdf = 'unknown'
        population_args = ()
        population_loc = None
        population_scale = None
        self.assertRaises(CRError, cr.get_distribution_stats,
                          population_cdf=population_cdf,
                          population_args=population_args,
                          population_loc=population_loc,
                          population_scale=population_scale)

        population_cdf = 'alpha'
        population_args = (2,)
        median, mean, var, std = cr.get_distribution_stats(
            population_cdf=population_cdf,
            population_args=population_args,
            population_loc=population_loc,
            population_scale=population_scale)

        self.assertAlmostEqual(median, 0.49297099121602045)
        self.assertTrue(mean == np.inf)
        self.assertTrue(var == np.inf)
        self.assertTrue(std == np.inf)

        population_cdf = 'beta'
        population_args = (2.31, 0.627)
        median, mean, var, std = cr.get_distribution_stats(
            population_cdf=population_cdf,
            population_args=population_args,
            population_loc=population_loc,
            population_scale=population_scale)

        self.assertAlmostEqual(median, 0.852865976189898)
        self.assertAlmostEqual(mean, 0.7865168539325842)
        self.assertAlmostEqual(var, 0.04264874077027537)
        self.assertAlmostEqual(std, 0.20651571555277667)

    def test_levene_test(self):
        """Test levene_test function."""
        rng = np.random.RandomState(12345)
        n_tries = 100

        shape, scale = 2., 2.
        results = [
            cr.levene_test(
                rng.gamma(shape, scale, size=5000),
                population_cdf='gamma',
                population_args=(shape,),
                population_loc=None,
                population_scale=scale,
                significance_level=0.05
            )
            for _ in range(n_tries)
        ]
        # expect ≈ 5 % rejections; allow up to 10 %
        rejection_rate = 1.0 - np.mean(results)
        self.assertLessEqual(rejection_rate, 0.10)

        shape = 1.99
        results = [
            cr.levene_test(
                rng.gamma(shape, 1.0, size=5000),
                population_cdf='gamma',
                population_args=(shape,),
                population_loc=None,
                population_scale=None,
                significance_level=0.05
            )
            for _ in range(n_tries)
        ]
        # expect ≈ 5 % rejections; allow up to 10 %
        rejection_rate = 1.0 - np.mean(results)
        self.assertLessEqual(rejection_rate, 0.10)

        self.assertFalse(cr.levene_test(rng.beta(2, 2, size=1000),
                                        population_cdf='gamma',
                                        population_args=(shape,),
                                        population_loc=None,
                                        population_scale=None,
                                        significance_level=0.05))

        results = [
            cr.levene_test(
                rng.beta(2, 2, size=5000),
                population_cdf='beta',
                population_args=(2, 2),
                population_loc=None,
                population_scale=None,
                significance_level=0.05
            )
            for _ in range(n_tries)
        ]
        # expect ≈ 5 % rejections; allow up to 10 %
        rejection_rate = 1.0 - np.mean(results)
        self.assertLessEqual(rejection_rate, 0.10)

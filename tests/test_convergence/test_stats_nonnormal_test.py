"""Test stats non-normal test module."""
import unittest
import numpy as np

try:
    import convergence as cr
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TestStatsNonNormalTestModule(unittest.TestCase):
    """Test stats nonnormal_tset module components."""

    def test_check_population_cdf_args(self):
        """Test check_population_cdf_args function."""
        population_cdf = 'unknown'
        population_args = ()
        self.assertRaises(CVGError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_cdf = 'alpha'
        population_args = ()
        self.assertRaises(CVGError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_cdf = 'alpha'
        population_args = (2, 2)
        self.assertRaises(CVGError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        population_args = ()
        self.assertRaises(CVGError, cr.check_population_cdf_args,
                          population_cdf=population_cdf,
                          population_args=population_args)

        for cdf in cr.ContinuousDistributions.keys():
            args = np.arange(
                cr.ContinuousDistributionsNumberOfRequiredArguments[cdf] + 1)
            self.assertRaises(CVGError, cr.check_population_cdf_args,
                              population_cdf=cdf,
                              population_args=args)

    def test_get_distribution_stats(self):
        """Test get_distribution_stats function."""
        population_cdf = 'unknown'
        population_args = ()
        population_loc = None
        population_scale = None
        self.assertRaises(CVGError, cr.get_distribution_stats,
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
        shape, scale = 2., 2.
        rvs = rng.gamma(shape, scale, size=1000)
        self.assertTrue(cr.levene_test(rvs,
                                       population_cdf='gamma',
                                       population_args=(shape,),
                                       population_loc=None,
                                       population_scale=scale,
                                       significance_level=0.05))

        shape = 1.99
        rvs = rng.gamma(shape, 1.0, size=1000)
        self.assertTrue(cr.levene_test(rvs,
                                       population_cdf='gamma',
                                       population_args=(shape,),
                                       population_loc=None,
                                       population_scale=None,
                                       significance_level=0.05))

        rvs = rng.beta(2, 2, size=1000)
        self.assertFalse(cr.levene_test(rvs,
                                        population_cdf='gamma',
                                        population_args=(shape,),
                                        population_loc=None,
                                        population_scale=None,
                                        significance_level=0.05))

        self.assertTrue(cr.levene_test(rvs,
                                       population_cdf='beta',
                                       population_args=(2, 2),
                                       population_loc=None,
                                       population_scale=None,
                                       significance_level=0.05))

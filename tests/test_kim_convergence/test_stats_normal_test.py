"""Test stats normal test module."""
import unittest
import numpy as np

try:
    import kim_convergence as cr
except:
    raise Exception('Failed to import `kim-convergence` utility module')

from kim_convergence import CRError


class TestStatsNormalTestModule(unittest.TestCase):
    """Test stats normal_tset module components."""

    def test_t_test(self):
        """Test t_test function."""
        rng = np.random.RandomState(12345)

        rvs = rng.randn(1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 0.0

        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 0.1
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        population_mean = 1.0
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        self.assertRaises(CRError, cr.t_test,
                          sample_mean=sample_mean,
                          sample_std=sample_std,
                          sample_size=sample_size,
                          population_mean=population_mean,
                          significance_level=0.0)

        rvs = rng.random_sample(1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 0.5
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 0.1
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        population_mean = 0.4
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        rvs = rng.chisquare(df=2, size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 2.0
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 1.0
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        population_mean = 1.5
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        rvs = rng.exponential(size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 1.0
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 0.9
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        population_mean = 1.1
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        rvs = rng.f(dfnum=1, dfden=48, size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 1.0434782608695652
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 1.2
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        rvs = rng.gamma(shape=1.99, size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 1.99
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        population_mean = 1.5
        self.assertFalse(cr.t_test(sample_mean=sample_mean,
                                   sample_std=sample_std,
                                   sample_size=sample_size,
                                   population_mean=population_mean,
                                   significance_level=0.05))

        rvs = rng.laplace(size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 0.0
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        rvs = rng.beta(a=2.31, b=0.627, size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 0.7865168539325842
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

        rvs = rng.standard_t(df=10, size=1000)
        sample_mean = rvs.mean()
        sample_std = rvs.std()
        sample_size = rvs.size
        population_mean = 0.0
        self.assertTrue(cr.t_test(sample_mean=sample_mean,
                                  sample_std=sample_std,
                                  sample_size=sample_size,
                                  population_mean=population_mean,
                                  significance_level=0.05))

    def test_chi_square_test(self):
        """Test chi_square_test function."""
        rng = np.random.RandomState(12345)

        rvs = rng.randn(100)
        sample_var = rvs.var()
        sample_size = rvs.size
        population_var = 1.0

        self.assertTrue(cr.chi_square_test(sample_var=sample_var,
                                           sample_size=sample_size,
                                           population_var=population_var,
                                           significance_level=0.05))

        population_var = 0.1
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 0.8
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 1.5
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        self.assertRaises(CRError, cr.chi_square_test,
                          sample_var=sample_var,
                          sample_size=sample_size,
                          population_var=population_var,
                          significance_level=0.0)

        rvs = rng.random_sample(20)
        sample_var = rvs.var()
        sample_size = rvs.size
        population_var = 0.08333333333333333
        self.assertTrue(cr.chi_square_test(sample_var=sample_var,
                                           sample_size=sample_size,
                                           population_var=population_var,
                                           significance_level=0.05))

        population_var = 0.04
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 0.2
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 0.5
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        rvs = rng.chisquare(df=2, size=50)
        sample_var = rvs.var()
        sample_size = rvs.size
        population_var = 4.0
        self.assertTrue(cr.chi_square_test(sample_var=sample_var,
                                           sample_size=sample_size,
                                           population_var=population_var,
                                           significance_level=0.05))

        population_var = 3.0
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 10.0
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        rvs = rng.exponential(size=200)
        sample_var = rvs.var()
        sample_size = rvs.size
        population_var = 1.0
        self.assertTrue(cr.chi_square_test(sample_var=sample_var,
                                           sample_size=sample_size,
                                           population_var=population_var,
                                           significance_level=0.05))

        population_var = 1.5
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

        population_var = 0.5
        self.assertFalse(cr.chi_square_test(sample_var=sample_var,
                                            sample_size=sample_size,
                                            population_var=population_var,
                                            significance_level=0.05))

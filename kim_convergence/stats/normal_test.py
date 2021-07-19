"""Test module for normal distributed data.

Note:
    The tests in this module are modified and fixed for the kim-convergence
    package use.

"""
from math import sqrt, fabs
import numpy as np
from scipy.stats import chi2

from kim_convergence._default import _DEFAULT_CONFIDENCE_COEFFICIENT
from kim_convergence import cr_check
from .t_dist import t_cdf


__all__ = [
    't_test',
    'chi_square_test',
]


def t_test(
        sample_mean: float,
        sample_std: float,
        sample_size: int,
        population_mean: float,
        significance_level: float = 1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """T-test for the mean.

    Calculate the T-test for the mean. This is a two-sided test for the null
    hypothesis that the expected value (mean) of a sample of independent
    observations `x` is equal to the given population mean, `population_mean`.

    Args:
        sample_mean (float): Sample mean.
        sample_std (float): Sample standard deviation.
        sample_size (int): Number of samples.
        population_mean (float): Expected value in the null hypothesis.
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected. (default: 0.05)

    Returns:
        bool: True for the expected value (mean) of a sample of independent
            observations `x` is equal to the given population mean,
            `population_mean`.

    """
    cr_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    nomin = sample_mean - population_mean
    denom = sample_std / sqrt(sample_size)
    t = fabs(nomin / denom)
    df = sample_size - 1
    prob = 2 * (1.0 - t_cdf(t, df))
    return significance_level < prob


def chi_square_test(
        sample_var: float,
        sample_size: int,
        population_var: float,
        significance_level: float = 1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    r"""Chi-square test for the variance.

    Calculate the chi-square test for the variance. This is a two-sided test.
    Test Statistic is :math:`T=(N−1)\frac{\text{var}}{\text{var}_0}`, where
    where `N` is the sample size and `var` is the sample variance. The ratio
    `var/var0` compares the ratio of the sample variance to the target
    variance.  The more this ratio deviates from 1, the more likely we are to
    reject the null hypothesis.

    The null hypothesis is that the variance of a sample of independent
    observations `x` is equal to the given population variance,
    `population_var`.

    Args:
        sample_var (float): Sample variance.
        sample_size (int): Number of samples.
        population_var (float): population variance.
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected. (default: 0.05)

    Returns:
        bool: True for the variance of a sample of independent observations `x`
            is equal to the given population variance, `population_var`.

    """
    cr_check(significance_level,
              var_name='significance_level',
              var_lower_bound=np.finfo(np.float64).resolution)

    df = sample_size - 1
    t = df * sample_var / population_var
    q1 = chi2.ppf(significance_level / 2, df)
    q2 = chi2.ppf(1 - (significance_level / 2), df)
    return q1 < t < q2

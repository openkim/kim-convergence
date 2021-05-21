"""t test module."""
from math import sqrt, fabs

from convergence._default import _DEFAULT_CONFIDENCE_COEFFICIENT
from .t_dist import t_cdf


__all__ = [
    't_test',
]


def t_test(sample_mean: float,
           sample_std: float,
           sample_size: int,
           population_mean: float,
           significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
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
            below which the null hypothesis will be rejected.

    Returns:
        bool: True for the expected value (mean) of a sample of independent
            observations `x` is equal to the given population mean,
            `population_mean`.

    """
    nomin = sample_mean - population_mean
    denom = sample_std / sqrt(sample_size)
    t = fabs(nomin / denom)
    df = sample_size - 1
    prob = 2 * (1.0 - t_cdf(t, df))
    return significance_level < prob

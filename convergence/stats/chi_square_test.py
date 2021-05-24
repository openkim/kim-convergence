"""Chi-square test module."""
from scipy.stats import chi2

from convergence._default import _DEFAULT_CONFIDENCE_COEFFICIENT


__all__ = [
    'chi_square_test',
]


def chi_square_test(
        sample_var: float,
        sample_size: int,
        population_var: float,
        significance_level=1 - _DEFAULT_CONFIDENCE_COEFFICIENT) -> bool:
    """Chi-square test for the variance.

    Calculate the chi-square test for the variance. This is a two-sided test
    for the null hypothesis that the variance of a sample of independent
    observations `x` is equal to the given population variance,
    `population_var`.

    The null hypothesis is that the sample variance equals the population
    variance.

    Args:
        sample_var (float): Sample variance.
        sample_size (int): Number of samples.
        population_var (float): population variance.
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected.

    Returns:
        bool: True for the variance of a sample of independent observations `x`
            is equal to the given population variance, `population_var`.

    """
    df = sample_size - 1
    t = df * sample_var / population_var
    q1 = chi2.ppf(significance_level / 2, df)
    q2 = chi2.ppf(1 - (significance_level / 2), df)
    return q1 < t < q2

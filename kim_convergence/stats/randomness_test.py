"""Independence test module."""

from math import sqrt
import numpy as np
from typing import Union

from .normal_dist import normal_interval
from kim_convergence import CRError, CRSampleSizeError, cr_check

__all__ = [
    "randomness_test",
]


def randomness_test(
    x: Union[np.ndarray, list[float]], significance_level: float
) -> bool:
    r"""Testing for independence of observations.

    The von-Neumann ratio test of independence of variables is a test designed
    for checking the independence of subsequent observations.

    The null hypothesis is that the data are independent and normally
    distributed.

    Args:
        x (array_like, 1d): Time series data.
        significance_level (float): Significance level. A probability threshold
            below which the null hypothesis will be rejected.

    Returns:
        bool: True for the independence of observations

    Note:
        Given a series :math:`x` of :math:`n` data points, the Von-Neumann test
        [vonneumann1941]_ [vonneumann1941b]_ statistic is

        .. math::

            v = \frac{\sum_{i=2}^{n} (x_i - x_{i-1})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}

        Under the null hypothesis of independence, the mean :math:`\bar{v} = 2`
        and the variance :math:`\sigma^2_v = \frac{4 (n - 2)}{(n^2-1)}` (see
        [williams1941]_, and [madansky1988]_ for a simple derivation).

    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise CRError("x is not an array of one-dimension.")

    x_size = x.size

    cr_check(
        significance_level,
        var_name="significance_level",
        var_lower_bound=np.finfo(np.float64).resolution,  # type: ignore[arg-type]
    )

    if x_size < 3:
        raise CRSampleSizeError(
            f"{x_size} input data points are not sufficient to be used "
            "by randomness_test method."
        )

    x_diff_square = np.diff(x, n=1)
    x_diff_square *= x_diff_square
    x_diff_square_mean = x_diff_square.mean()

    von_neumann_mean = 2.0
    von_neumann_std = sqrt(4.0 * (x_size - 2.0) / (x_size * x_size - 1.0))

    lower_interval, upper_interval = normal_interval(
        1.0 - significance_level, loc=von_neumann_mean, scale=von_neumann_std
    )

    # denominator must be sample variance (n-1) to match the documented formula
    von_neumann_ratio = x_diff_square_mean / x.var(ddof=1)

    return lower_interval < von_neumann_ratio < upper_interval

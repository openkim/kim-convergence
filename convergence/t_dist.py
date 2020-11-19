"""T distribution module.

This module is specilized for the ``convergence`` code and is not a general
function to be used for other purposes.
"""

import numpy as np
from math import lgamma, fabs
from copy import deepcopy

from .err import CVGError
from .zero_rc_bounds import ZERO_RC_BOUNDS
from .s_normal_dist import s_normal_inv_cdf
from .beta_dist import betai_cdf_ccdf

__all__ = [
    't_cdf_ccdf',
    't_cdf',
    't_inv_cdf',
    't_interval'
]


def t_cdf_ccdf(t: float, df: float):
    r"""Compute the cumulative distribution of the t-distribution.

    The cumulative distribution of the t-distribution for t > 0, can be
    written in terms of the regularized incomplete beta function as,

    .. math::

        \int_{-\infty}^t f(u)\,du = 1 - \frac{1}{2} I_{x(t)}\left(\frac{\nu}{2}, \frac{1}{2}\right),

    where,

    .. math::

        x(t) = \frac{\nu}{{t^2+\nu}}.

    Other t values would be obtained by symmetry.

    Args:
        t (float): Upper limit of the integration.
        df (float): Degrees of freedom, must be a positive number.

    Returns:
        float, float: cdf, ccdf
            Cumulative t-distribution, compliment of the cumulative
            t-distribution.

    """
    if df < 1:
        msg = 'df = {} is wrong. Degrees of freedom, must be '.format(df)
        msg += 'positive and greater than 1.'
        raise CVGError(msg)

    tt = t * t
    denom = df + tt
    x = df / denom

    bcdf, bccdf = betai_cdf_ccdf(0.5 * df, 0.5, x)

    if t <= 0.0:
        cdf = 0.5 * bcdf
        ccdf = bccdf + cdf
        return cdf, ccdf

    ccdf = 0.5 * bcdf
    cdf = bccdf + ccdf
    return cdf, ccdf


def t_cdf(t: float, df: float):
    r"""Compute the cumulative distribution of the t-distribution.

    The cumulative distribution of the t-distribution for t > 0, can be
    written in terms of the regularized incomplete beta function as,

    .. math::

        \int_{-\infty}^t f(u)\,du = 1 - \frac{1}{2} I_{x(t)}\left(\frac{\nu}{2}, \frac{1}{2}\right),

    where,

    .. math::

        x(t) = \frac{\nu}{{t^2+\nu}}.

    Other t values would be obtained by symmetry.

    Args:
        t (float): Upper limit of the integration.
        df (float): Degrees of freedom, must be a positive number.

    Returns:
        float: Cumulative t-distribution.

    """
    cdf, _ = t_cdf_ccdf(t, df)
    return cdf


def t_inv_cdf(p: float,
              df: float,
              *,
              _tol=1.0e-8,
              _atol=1.0e-50,
              _rtinf=1.0e100):
    """Compute the t_distribution inverse cumulative distribution function.

    Compute the inverse cumulative distribution function (percent point
    function or quantile function) for t-distributions with df degrees of
    freedom. Inverse cumulative distribution function finds the value of the
    random variable such that the probability of the variable being less than
    or equal to that value equals the given probability.

    Args:
        p (float): Probability (must be between 0.0 and 1.0)
        df (float): Degrees of freedom, must be > 1.

    Returns:
        float: The inverse cumulative distribution function.
            The inverse cumulative distribution function.
            The value x of the random variable X such that the
            probability of the variable being less than or equal to that
            value equals the given probability p. :math:`x : P(X <= x) = p`.

    """
    if p <= 0.0 or p >= 1.0:
        msg = 'p = {} is not in the range (0.0 1.0).'.format(p)
        raise CVGError(msg)

    if df < 1:
        msg = 'df = {} is wrong. Degrees of freedom, must be '.format(df)
        msg += 'positive and greater than 1.'
        raise CVGError(msg)

    x = fabs(s_normal_inv_cdf(p))
    x_sq = x * x

    num = np.array(
        ((1.0e+0 * x_sq + 1.0e+0) * x,
         ((5.0e+0 * x_sq + 16.0e+0) * x_sq + 3.0e+0) * x,
         (((3.0e+0 * x_sq + 19.0e+0) * x_sq + 17.0) * x_sq - 15.0e+0) * x,
         ((((79.0e+0 * x_sq + 776.0e+0) * x_sq + 1482.0)
           * x_sq - 1920.0e+0) * x_sq - 945.0e+0) * x
         ), dtype=np.float64)

    _df = np.array((df, df, df, df), dtype=np.float64)
    denpow = np.multiply.accumulate(_df)

    den = np.array((4.0, 96.0, 384.0, 92160.0), dtype=np.float64)
    den *= denpow

    num /= den

    x += np.sum(num)

    if p < 0.5:
        x = -x

    q = 1. - p
    qporq = p <= q

    d = ZERO_RC_BOUNDS(-_rtinf, _rtinf, 0.5, 0.5,
                       5.0, abs_tol=_atol, rel_tol=_tol)

    status = 0
    fx = 0.0

    status, x = d.zero(status, x, fx)

    while True:
        if status == 0:
            return x

        cdf, ccdf = t_cdf_ccdf(x, df)

        if qporq:
            fx = cdf - p
        else:
            fx = ccdf - q

        status, x = d.zero(status, x, fx)


def t_interval(p: float, df: float):
    r"""Compute the t_distribution confidence interval.

    Compute the t_distribution confidence interval with equal areas around
    the median.

    Args:
        p (float): Probability (must be between 0.0 and 1.0)
        df (float): Degrees of freedom, must be > 0.

    Returns:
        float, float : lower bound, upper bound of the confidence interval
            end-points of range that contain :math:`100 \alpha \%` of the
            t_distribution possible values.

    """
    if p <= 0.0 or p >= 1.0:
        msg = 'p = {} is not in the range (0.0 1.0).'.format(p)
        raise CVGError(msg)

    lower = (1.0 - p) / 2
    upper = (1.0 + p) / 2

    lower_interval = t_inv_cdf(lower, df)
    upper_interval = t_inv_cdf(upper, df)

    return lower_interval, upper_interval

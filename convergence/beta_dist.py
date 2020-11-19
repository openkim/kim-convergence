"""Beta distribution module."""

import numpy as np
from math import lgamma
from copy import deepcopy

from .err import CVGError

__all__ = [
    'beta',
    'betacf',
    'betai',
    'betai_cdf_ccdf',
    'betai_cdf',
]


def beta(a, b):
    r"""Beta function.

    Beta function [7]_ is defined as,

    .. math::

        B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)},

    where :math:`\Gamma` is the gamma function.

    Args:
        a (float): First parameter of the beta distribution.
        b (float): Second parameter of the beta distribution.

    Returns:
        float: Beta function value.

    References:
        .. [7] Numerical Recipes: The Art of Scientific Computing, Third
            Edition (2007)

    """
    _beta = np.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
    return _beta


def betacf(a, b, x, *,
           eps=np.finfo(np.float64).resolution,
           max_iteration=200,
           _fpmin=1.0e-30):
    """Continued fraction for incomplete beta function by modified Lentz’s method.

    Evaluates continued fraction for incomplete beta function by modified
    Lentz’s method [7]_.

    Args:
        a (float): First parameter of the beta distribution.
        b (float): Second parameter of the beta distribution.
        x (float): Real-valued such that it must be between 0.0 and 1.0.
        eps (float, optional): Machine precision epsilon.
            (default: {np.finfo(np.float64).resolution})
        max_iteration (int, optional): Maximum number of iterations.
            (default: 200)
        _fpmin (float, optional): Minimum floating point precision.
            (default: 1.0e-30)

    Returns:
        float: Continued fraction for incomplete beta function.

    """
    _fpmax = 1. / _fpmin

    # These q’s will be used in factors
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    # First step of Lentz’s method.
    d = 1.0 - qab * x / qap
    c = 1.0

    if (abs(d) < _fpmin):
        d = deepcopy(_fpmax)
        h = deepcopy(_fpmax)
    else:
        d = 1.0 / d
        h = deepcopy(d)

    for m in range(1, max_iteration + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))

        # One step (the even one) of the recurrence
        d = 1.0 + aa * d
        c = 1.0 + aa / c

        _d = abs(d) < _fpmin
        _c = abs(c) < _fpmin

        if _d and _c:
            d = deepcopy(_fpmax)
            c = deepcopy(_fpmin)
            _del = 1.0
        elif _d:
            d = deepcopy(_fpmax)
            _del = _fpmax * c
        elif _c:
            c = deepcopy(_fpmin)
            _del = _fpmin / d
        else:
            d = 1.0 / d
            _del = d * c
        h *= _del

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))

        # Next step of the recurrence (the odd one)
        d = 1.0 + aa * d
        c = 1.0 + aa / c

        _d = abs(d) < _fpmin
        _c = abs(c) < _fpmin

        if _d and _c:
            d = deepcopy(_fpmax)
            c = deepcopy(_fpmin)
            _del = 1.0
        elif _d:
            d = deepcopy(_fpmax)
            _del = _fpmax * c
        elif _c:
            c = deepcopy(_fpmin)
            _del = _fpmin / d
        else:
            d = 1.0 / d
            _del = d * c
        h *= _del

        if (abs(_del - 1.0) < eps):
            return h

    msg = 'betacf failed with the current result = {}, '.format(h)
    msg = 'where a={} or b={} are too big, or '.format(a, b)
    msg += 'max_iteration={} is too small.'.format(max_iteration)
    raise CVGError(msg)


def betai(a, b, x):
    r"""Incomplete beta function.

    Incomplete beta function [7]_ is defined as,

    .. math::

        I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^x~t^{a-1}(1-t)^{b-1}~dt,

    Args:
        a (float): First parameter of the beta distribution.
        b (float): Second parameter of the beta distribution.
        x (float): Real-valued such that it must be between 0.0 and 1.0.

    Returns:
        float: Incomplete beta function value.

    """
    if x < 0.0 or x > 1.0:
        return np.nan
    
    if x == 0.0 or x == 1.0:
        return x

    _beta = lgamma(a + b) - lgamma(a) - lgamma(b) + \
        a * np.log(x) + b * np.log(1.0 - x)
    _beta = np.exp(_beta)

    frac = (a + 1.) / (a + b + 2.)

    if x < frac:
        _beta *= betacf(a, b, x)
        _beta /= a
        return _beta
    
    _beta *= betacf(b, a, 1.0 - x)
    _beta /= b
    return 1. - _beta


def betai_cdf_ccdf(a, b, x):
    r"""Calculate the cumulative distribution of the incomplete beta distribution.

    Calculate the cumulative distribution of the incomplete beta
    distribution with parameters a and b as,

    .. math::

        \int_0^x \frac{t^{a-1}~(1-t)^{b-1}}{Beta(a,b)}~dt,

    where, :math:`Beta(a,b)` is the beta function.

    Args:
        a (float): First parameter of the beta distribution.
        b (float): Second parameter of the beta distribution.
        x (float): Upper limit of integration

    Returns:
        float, float: Cumulative incomplete beta distribution, compliment of the cumulative incomplete beta distribution.

    """
    if x <= 0.0:
        return 0.0, 1.0
    
    if x >= 1.0:
        return 1.0, 0.0

    cdf = betai(a, b, x)
    ccdf = 1.0 - cdf
    return cdf, ccdf


def betai_cdf(a, b, x):
    r"""Calculate the cumulative distribution of the incomplete beta distribution.

    Calculate the cumulative distribution of the incomplete beta
    distribution with parameters a and b as,

    .. math::

        \int_0^x \frac{t^{a-1}~(1-t)^{b-1}}{Beta(a,b)}~dt,

    where, :math:`Beta(a,b)` is the beta function.

    Args:
        a (float): First parameter of the beta distribution.
        b (float): Second parameter of the beta distribution.
        x (float): Upper limit of integration

    Returns:
        float: Cumulative incomplete beta distribution.

    """
    cdf, _ = betai_cdf_ccdf(a, b, x)
    return cdf

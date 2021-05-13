"""normal distribution module.

`s_normal_inv_cdf` code is adapted from python statistics module [1]_ by
Yaser Afshar.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python
    Software Foundation;
    All Rights Reserved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

References:
    .. [1] Python statistics module.
           https://www.python.org/

"""

from copy import deepcopy
from math import log, fabs, sqrt, inf, nan

from convergence import CVGError

__all__ = [
    's_normal_inv_cdf',
    'normal_inv_cdf',
    'normal_interval'
]


def s_normal_inv_cdf(p: float):
    r"""Compute the standard normal distribution inverse cumulative distribution function.

    Compute the inverse cumulative distribution function (percent point
    function or quantile function) for standard normal distribution [5]_,
    [6]_.

    Ars:
        p {float} -- Probability (must be between 0.0 and 1.0)

    Returns:
        float -- the inverse cumulative distribution function.
            the value x of the random variable X such that the probability of
            the variable being less than or equal to that value equals the
            given probability p. :math:`x : P(X <= x) = p`.

    References:
        .. [5] Python statistics module. https://www.python.org/

        .. [6] Wichura, M.J. (1988). "Algorithm AS241: The Percentage Points
            of the Normal Distribution" Applied Statistics. Blackwell
            Publishing. 37(3), 477–484.

    """
    if p == 0.0:
        return -inf

    if p == 1.0:
        return inf

    if p < 0.0 or p > 1.0:
        return nan

    q = p - 0.5
    if fabs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.50908_09287_30122_6727e+3 * r +
                     3.34305_75583_58812_8105e+4) * r +
                    6.72657_70927_00870_0853e+4) * r +
                   4.59219_53931_54987_1457e+4) * r +
                  1.37316_93765_50946_1125e+4) * r +
                 1.97159_09503_06551_4427e+3) * r +
                1.33141_66789_17843_7745e+2) * r +
               3.38713_28727_96366_6080e+0) * q
        den = (((((((5.22649_52788_52854_5610e+3 * r +
                     2.87290_85735_72194_2674e+4) * r +
                    3.93078_95800_09271_0610e+4) * r +
                   2.12137_94301_58659_5867e+4) * r +
                  5.39419_60214_24751_1077e+3) * r +
                 6.87187_00749_20579_0830e+2) * r +
                4.23133_30701_60091_1252e+1) * r +
               1.0)
        x = num / den
        return x

    r = p if q <= 0.0 else 1.0 - p

    r = sqrt(-log(r))

    if r <= 5.0:
        r -= 1.6
        num = (((((((7.74545_01427_83414_07640e-4 * r +
                     2.27238_44989_26918_45833e-2) * r +
                    2.41780_72517_74506_11770e-1) * r +
                   1.27045_82524_52368_38258e+0) * r +
                  3.64784_83247_63204_60504e+0) * r +
                 5.76949_72214_60691_40550e+0) * r +
                4.63033_78461_56545_29590e+0) * r +
               1.42343_71107_49683_57734e+0)
        den = (((((((1.05075_00716_44416_84324e-9 * r +
                     5.47593_80849_95344_94600e-4) * r +
                    1.51986_66563_61645_71966e-2) * r +
                   1.48103_97642_74800_74590e-1) * r +
                  6.89767_33498_51000_04550e-1) * r +
                 1.67638_48301_83803_84940e+0) * r +
                2.05319_16266_37758_82187e+0) * r +
               1.0)
    else:
        r -= 5.0
        num = (((((((2.01033_43992_92288_13265e-7 * r +
                     2.71155_55687_43487_57815e-5) * r +
                    1.24266_09473_88078_43860e-3) * r +
                   2.65321_89526_57612_30930e-2) * r +
                  2.96560_57182_85048_91230e-1) * r +
                 1.78482_65399_17291_33580e+0) * r +
                5.46378_49111_64114_36990e+0) * r +
               6.65790_46435_01103_77720e+0)
        den = (((((((2.04426_31033_89939_78564e-15 * r +
                     1.42151_17583_16445_88870e-7) * r +
                    1.84631_83175_10054_68180e-5) * r +
                   7.86869_13114_56132_59100e-4) * r +
                  1.48753_61290_85061_48525e-2) * r +
                 1.36929_88092_27358_05310e-1) * r +
                5.99832_20655_58879_37690e-1) * r +
               1.0)
    x = num / den

    if q < 0.0:
        return -x
    return x


def normal_inv_cdf(p: float, *, loc=0.0, scale=1.0):
    r"""Compute the normal distribution inverse cumulative distribution function.

    Ars:
        p {float} -- Probability (must be between 0.0 and 1.0)
        loc (float, optional): location parameter (default: 0.0)
        scale (float, optional): scale parameter (default: 1.0)

    Returns:
        float -- the inverse cumulative distribution function.
            the value x of the random variable X such that the probability of
            the variable being less than or equal to that value equals the
            given probability p. :math:`x : P(X <= x) = p`.

    """
    return s_normal_inv_cdf(p) * scale + loc


def normal_interval(confidence_level: float, *, loc=0.0, scale=1.0):
    r"""Compute the normal distribution confidence interval.

    Compute the normal-distribution confidence interval with equal areas around
    the median.

    Args:
        confidence_level (float): (or confidence coefficient) must be between
            0.0 and 1.0
        loc (float, optional): location parameter (default: 0.0)
        scale (float, optional): scale parameter (default: 1.0)

    Returns:
        float, float : lower bound, upper bound of the confidence interval
            end-points of range that contain
            :math:`100 \text{confidence_level} \%` of the normal distribution
            possible values.

    Note:
        - Confidence interval is a range of values that is likely to contain an
          unknown population parameter.

        - Confidence level is the percentage of the confidence intervals which
          will hold the population parameter.

        - The significance level or alpha is the probability of rejecting the
          null hypothesis when it is true. To find alpha, just subtract the
          confidence interval from 100%. E.g., the significance level for a 90%
          confidence level is 100% – 90% = 10%.

    """
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        msg = 'confidence level = {} is not in '.format(confidence_level)
        msg += 'the range (0.0 1.0).'
        raise CVGError(msg)

    lower = (1.0 - confidence_level) / 2
    upper = (1.0 + confidence_level) / 2

    lower_interval = s_normal_inv_cdf(lower) * scale + loc
    upper_interval = s_normal_inv_cdf(upper) * scale + loc

    return lower_interval, upper_interval

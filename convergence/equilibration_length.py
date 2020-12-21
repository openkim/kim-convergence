"""Equilibration_length estimation module."""

from math import isclose
import numpy as np

from .err import CVGError
from .statistical_inefficiency import \
    statistical_inefficiency,\
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency, \
    split_statistical_inefficiency, \
    si_methods

__all__ = [
    'estimate_equilibration_length',
]


def estimate_equilibration_length(x, *,
                                  si='statistical_inefficiency',
                                  nskip=1,
                                  fft=False,
                                  minimum_correlation_time=None,
                                  ignore_end=None):
    """Estimate the equilibration point in a time series data.

    Estimate the equilibration point in a time series data using the
    statistical inefficiencies [11]_, [8]_, [9]_.

    Args:
        x (array_like, 1d): time series data.
        si (str, optional): statistical inefficiency method.
            (default: 'statistical_inefficiency')
        nskip (int, optional): the number of data points to skip.
            (default: 1)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: False)
        minimum_correlation_time (int, optional): the minimum amount of 
            correlation function to compute. The algorithm terminates after 
            computing the correlation time out to minimum_correlation_time when 
            the correlation function first goes negative. (default: None)
        ignore_end (int, or float, or None, optional): if ``int``, it is the
            last few points that should be ignored. if ``float``, should be in
            ``(0, 1)`` and it is the percent of number of points that should be
            ignored. If ``None`` it would be set to the one fourth of the total
            number of points. (default: None)

    Returns:
        int, float: equilibration index, statistical inefficiency estimates
            equilibration index, and statitical inefficiency estimates of a
            time series at the equilibration index estimate.

    References:
        .. [11] Chodera, J. D., (2016). "A Simple Method for Automated
               Equilibration Detection in Molecular Simulations". J. Chem.
               Theory and Comp., Simulation., 12(4), p. 1799--1805.

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # Get the length of the timeseries.
    x_size = x.size

    if isinstance(si, str):
        if si not in si_methods:
            msg = 'method {} not found. Valid statistical '.format(si)
            msg += 'inefficiency (si) methods are:\n\t- '
            msg += '{}'.format('\n\t- '.join(si_methods))
            raise CVGError(msg)
    else:
        msg = 'si is not a `str`.'
        raise CVGError(msg)

    si_func = si_methods[si]

    if not isinstance(nskip, int):
        if nskip is None:
            nskip = 1
        else:
            msg = 'nskip must be an `int`.'
            raise CVGError(msg)
    elif nskip < 1:
        msg = 'nskip must be a positive `int`.'
        raise CVGError(msg)

    if not isinstance(ignore_end, int):
        if ignore_end is None:
            ignore_end = max(1, x_size // 4)
        elif isinstance(ignore_end, float):
            if not 0.0 < ignore_end < 1.0:
                msg = 'invalid ignore_end = {}. If '.format(ignore_end)
                msg += 'ignore_end input is a `float`, it should be in a '
                msg += '`(0, 1)` range.'
                raise CVGError(msg)
            ignore_end *= x_size
            ignore_end = max(1, int(ignore_end))
        else:
            msg = 'invalid ignore_end = {}. '.format(ignore_end)
            msg += 'ignore_end is not an `int`, `float`, or `None`.'
            raise CVGError(msg)
    elif ignore_end < 1:
        msg = 'invalid ignore_end = {}. '.format(ignore_end)
        msg += 'ignore_end should be a positive `int`.'
        raise CVGError(msg)

    # Upper bound check
    if si == 'r_statistical_inefficiency':
        if x_size < 4:
            msg = '{} number of input data points is not '.format(x_size)
            msg += 'sufficient to be used by "{}" method.'.format(si)
            raise CVGError(msg)
        ignore_end = max(3, ignore_end)
    elif si == 'split_r_statistical_inefficiency':
        if x_size < 8:
            msg = '{} number of input data points is not '.format(x_size)
            msg += 'sufficient to be used by "{}" method.'.format(si)
            raise CVGError(msg)
        ignore_end = max(7, ignore_end)
    elif si == 'split_statistical_inefficiency':
        if x_size < 8:
            msg = '{} number of input data points is not '.format(x_size)
            msg += 'sufficient to be used by "{}" method.'.format(si)
            raise CVGError(msg)
        ignore_end = max(7, ignore_end)

    if x_size <= ignore_end:
        msg = 'invalid ignore_end = {}.\n'.format(ignore_end)
        msg = 'Wrong number of data points is requested to be ignored '
        msg += 'from the total {} points.'.format(x_size)
        raise CVGError(msg)

    # Special case if timeseries is constant.
    _std = np.std(x)

    if not np.isfinite(_std):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

    if isclose(_std, 0, abs_tol=1e-14):
        # index and si
        return 0, 1.0

    del _std

    # Upper bound check
    upper_bound = x_size - ignore_end

    nskip = min(nskip, upper_bound)

    # Estimate of statistical inefficiency
    statistical_inefficiency_estimate = 1.0

    # Effective samples size
    effective_samples_size = 0.0

    # Equilibration estimate index
    equilibration_index_estimate = 0

    for t in range(0, upper_bound, nskip):
        # Compute the statitical inefficiency of a time series
        try:
            si_value = si_func(
                x[t:], fft=fft,
                minimum_correlation_time=minimum_correlation_time)
        except CVGError:
            si_value = float(x_size - t)

        _effective_samples_size = float(x_size - t) / si_value

        # Find the maximum
        if _effective_samples_size > effective_samples_size:
            statistical_inefficiency_estimate = si_value
            effective_samples_size = _effective_samples_size
            equilibration_index_estimate = t

    return equilibration_index_estimate, statistical_inefficiency_estimate

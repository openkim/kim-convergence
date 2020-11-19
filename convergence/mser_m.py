"""MSER-M module."""

from math import isclose
import numpy as np

from .err import CVGError
from .batch import batch

__all__ = [
    'mser_m',
]


def mser_m(x,
           *,
           batch_size=5,
           scale='translate_scale',
           with_centering=False,
           with_scaling=False,
           ignore_end_batch=None):
    r"""Determine the truncation point using marginal standard error rules.

    Determine the truncation point using marginal standard error rules
    (MSER). The MSER [3]_ and MSER-5 [4]_ rules determine the truncation
    point as the value of :math:`d` that best balances the tradeoff between
    improved accuracy (elimination of bias) and decreased precision
    (reduction in the sample size) for the input series. They select a
    truncation point that minimizes the width of the marginal confidence
    interval about the truncated sample mean. The marginal confidence
    interval is a measure of the homogeneity of the truncated series.
    The optimal truncation point :math:`d(j)^*` selected by MSER can be
    expressed as:

    .. math::

        d(j)^* = \underset{n>d(j) \geq 0}{\text{argmin}}\left[\frac{1}{(n(j)-d(j))^2} \sum_{i=d}^{n}{\left(X_i(j)- \bar{X}_{n,d}(j) \right )^2}\right]

    MSER-m applies the equation to a series of batch averages instead of the
    raw series.

    Args:
        x (array_like, 1d): Time series data.
        batch_size (int, optional): batch size. (default: {5})
        scale (str, optional): A method to standardize a dataset.
            (default: {'translate_scale'})
        with_centering (bool, optional): If True, use x minus the scale metod
            centering approach. (default: {False})
        with_scaling (bool, optional): If True, scale the data to scale metod
            scaling approach. (default: {False})
        ignore_end_batch (int, or float, or None, optional): if `int`, it is
            the last few batch points that should be ignored. if `float`,
            should be in `(0, 1)` and it is the percent of last batch points
            that should be ignored. if `None` it would be set to the
            `batch_size`. (default: {None})

    Returns:
        bool, int: truncated, truncation point.
            Truncation point is the index to truncate.

    Note:
      MSER-m sometimes erroneously reports a truncation point at the end of
      the data series. This is because the method can be overly sensitive to
      observations at the end of the data series that are close in value.
      Here, we avoid this artifact, by not allowing the algorithm to consider
      the standard errors calculated from the last few data points.
      If the truncation point returned by MSER-m > n/2, it is considered an
      invalid value and `truncated` will return as `False`. It means the
      method has not been provided with enough data to produce a valid
      result, and more data is required.

    References:
        .. [3] White, K.P., Jr. (1997). "An effective truncation heuristic
            for bias reduction in simulation output.". Simulation., 69(6),
            p. 323--334.
        .. [4] Spratt, S. C. (1998). "Heuristics for the startup problem."
            M.S. Thesis, Department OS Systems Engineering, University of
            Virginia.

    """
    x = np.array(x, copy=False)

    # Check inputs
    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # Special case if timeseries is constant.
    _std = np.std(x)

    if not np.isfinite(_std):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)

    # assures that the two values are the same within about 14 decimal digits.
    if isclose(_std, 0, rel_tol=1e-14):
        if not isinstance(batch_size, int):
            msg = 'batch_size = {} is not an `int`.'.format(batch_size)
            raise CVGError(msg)

        if batch_size < 1:
            msg = 'batch_size = {} < 1 is not valid.'.format(batch_size)
            raise CVGError(msg)

        if x.size < batch_size:
            return False, 0

        return True, 0

    del _std

    # Initialize
    z = batch(x,
              batch_size=batch_size,
              scale=scale,
              with_centering=with_centering,
              with_scaling=with_scaling)

    # Number of batches
    n_batches = z.size

    if not isinstance(ignore_end_batch, int):
        if ignore_end_batch is None:
            ignore_end_batch = max(1, batch_size)
        elif isinstance(ignore_end_batch, float):
            if not 0.0 < ignore_end_batch < 1.0:
                msg = 'invalid ignore_end_batch = '
                msg += '{}. If ignore_end_batch '.format(ignore_end_batch)
                msg += 'input is a `float`, it should be in a `(0, 1)` '
                msg += 'range.'
                raise CVGError(msg)
            ignore_end_batch *= n_batches
            ignore_end_batch = max(1, int(ignore_end_batch))
        else:
            msg = 'invalid ignore_end_batch = {}. '.format(ignore_end_batch)
            msg += 'ignore_end_batch is not an `int`, `float`, or `None`.'
            raise CVGError(msg)
    elif ignore_end_batch < 1:
        msg = 'invalid ignore_end_batch = {}. '.format(ignore_end_batch)
        msg += 'ignore_end_batch should be a positive `int`.'
        raise CVGError(msg)

    if n_batches <= ignore_end_batch:
        msg = 'invalid ignore_end_batch = {}.\n'.format(ignore_end_batch)
        msg += 'Wrong number of batches is requested to be ignored '
        msg += 'from the total {} batches.'.format(n_batches)
        raise CVGError(msg)

    # Correct the size of data
    n = n_batches * batch_size

    # To find the optimal truncation point in MSER-m

    n_batches_minus_d_inv = 1. / np.arange(n_batches, 0, -1)

    sum_z = np.add.accumulate(z[::-1])[::-1]
    sum_z_sq = sum_z * sum_z
    sum_z_sq *= n_batches_minus_d_inv

    n_batches_minus_d_inv *= n_batches_minus_d_inv

    zsq = z * z
    sum_zsq = np.add.accumulate(zsq[::-1])[::-1]

    d = n_batches_minus_d_inv * (sum_zsq - sum_z_sq)

    # Convert truncation from batch to raw data
    truncate_index = np.nanargmin(d[:-ignore_end_batch]) * batch_size

    if truncate_index > n // 2:
        # Any truncation value > n/2 is considered 
        # an invalid value and rejected
        return False, truncate_index
    
    return True, truncate_index

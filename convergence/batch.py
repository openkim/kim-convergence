"""Batch module."""

import numpy as np

from .err import CVGError
from ._default import \
    _DEFAULT_BATCH_SIZE, \
    _DEFAULT_SCALE_METHOD, \
    _DEFAULT_WITH_CENTERING, \
    _DEFAULT_WITH_SCALING
from .scale import scale_methods


__all__ = [
    'batch',
]


def batch(time_series_data,
          *,
          batch_size=_DEFAULT_BATCH_SIZE,
          func=np.mean,
          scale=_DEFAULT_SCALE_METHOD,
          with_centering=_DEFAULT_WITH_CENTERING,
          with_scaling=_DEFAULT_WITH_SCALING):
    r"""Batch the time series data.

    Args:
        time_series_data (array_like, 1d): Time series data.
        batch_size (int, optional): batch size. (default: 5)
        func (callable, optional): Reduction function capable of receiving a
            single axis argument. It is called with `time_series_data` as first
            argument. (default: np.mean)
        scale (str, optional): A method to standardize a dataset.
            (default: 'translate_scale')
        with_centering (bool, optional): If True, use time_series_data minus
            the scale metod centering approach. (default: False)
        with_scaling (bool, optional): If True, scale the data to scale
            metod scaling approach. (default: False)

    Returns:
        1darray: Batched (, and rescaled) data.

    Notes:
        This function will terminate the end of the data points which are
        remainder of the division of data points by the batch_size.

        By default, this method is using ``np.mean`` and compute the arithmetic
        mean.

    """
    time_series_data = np.array(time_series_data, copy=False)

    # Check inputs
    if time_series_data.ndim != 1:
        msg = 'time_series_data is not an array of one-dimension.'
        raise CVGError(msg)

    if not isinstance(batch_size, int):
        msg = 'batch_size = {} is not an `int`.'.format(batch_size)
        raise CVGError(msg)

    if batch_size < 1:
        msg = 'batch_size = {} < 1 is not valid.'.format(batch_size)
        raise CVGError(msg)

    if not np.all(np.isfinite(time_series_data)):
        msg = 'there is at least one value in the input '
        msg += 'array which is non-finite or not-number.'
        raise CVGError(msg)

    # Initialize

    # Number of batches
    number_batches = time_series_data.size // batch_size

    if number_batches == 0:
        msg = 'invalid number of batches = {}.\n'.format(number_batches)
        msg += 'The number of input data points = '
        msg += '{} are '.format(time_series_data.size)
        msg += 'not enough to produce batches with the batch size of '
        msg += '{} data points.'.format(batch_size)
        raise CVGError(msg)

    # Correct the size of data
    processed_sample_size = number_batches * batch_size

    # Compute batch averages

    # The raw data is batched into non-overlapping batches of size batch_size
    try:
        batched_time_series_data = func(
            time_series_data[:processed_sample_size].reshape((-1, batch_size)),
            axis=1, dtype=np.float64)
    # Reduction function like median has no dtype args
    except TypeError:
        batched_time_series_data = func(
            time_series_data[:processed_sample_size].reshape((-1, batch_size)),
            axis=1)

    if with_centering or with_scaling:
        if not isinstance(scale, str):
            msg = 'scale is not a `str`.\nScale = {} is not '.format(scale)
            msg += 'a valid method to scale and standardize a dataset.'
            raise CVGError(msg)

        if scale not in scale_methods:
            msg = 'method "{}" not found. Valid methods '.format(scale)
            msg += 'to scale and standardize a dataset are:\n\t- '
            msg += '{}'.format('\n\t- '.join(scale_methods))
            raise CVGError(msg)

        scale_func = scale_methods[scale]

        batched_time_series_data = scale_func(batched_time_series_data,
                                              with_centering=with_centering,
                                              with_scaling=with_scaling)

    return batched_time_series_data

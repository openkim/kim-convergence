"""Batch module."""

import numpy as np

from .err import CVGError
from .scale import scale_methods

__all__ = [
    'batch',
]


def batch(time_series_data,
          *,
          batch_size=5,
          scale='translate_scale',
          with_centering=False,
          with_scaling=False):
    r"""Batch the time series data.

    Args:
        time_series_data (array_like, 1d): Time series data.
        batch_size (int, optional): batch size. (default: {5})
        scale (str, optional): A method to standardize a dataset.
            (default: {'translate_scale'})
        with_centering (bool, optional): If True, use time_series_data minus
            the scale metod centering approach. (default: {False})
        with_scaling (bool, optional): If True, scale the data to scale
            metod scaling approach. (default: {False})

    Returns:
        1darray: Batched (, and rescaled) data.

    Note:
        This function will terminate the end of the data points which are
        remainder of the division of data points by the batch_size.

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

    if isinstance(scale, str):
        if scale not in scale_methods:
            msg = 'method "{}" not found. Valid methods '.format(scale)
            msg += 'to scale and standardize a dataset are:\n\t- '
            msg += '{}'.format('\n\t- '.join(scale_methods))
            raise CVGError(msg)
        scale_func = scale_methods[scale]
    else:
        msg = 'scale is not a `str`.\nScale = {} is not '.format(scale)
        msg += 'a valid method to scale and standardize a dataset.'
        raise CVGError(msg)

    # Initialize

    # Number of batches
    n_batches = time_series_data.size // batch_size

    if n_batches == 0:
        msg = 'invalid number of batches = {}.\n'.format(n_batches)
        msg += 'The number of input data points = '
        msg += '{} are '.format(time_series_data.size)
        msg += 'not enough to produce batches with the batch size of '
        msg += '{} data points.'.format(batch_size)
        raise CVGError(msg)

    # Correct the size of data
    max_size = n_batches * batch_size

    # Compute batch averages

    # The raw data is batched into non-overlapping batches of size batch_size
    batched_time_series_data = np.matmul(
        time_series_data[:max_size].reshape((-1, batch_size)),
        np.ones([batch_size], dtype=np.float64))

    # Calculate the batch means
    batched_time_series_data /= np.float64(batch_size)

    if with_centering or with_scaling:
        batched_time_series_data = scale_func(batched_time_series_data,
                                              with_centering=with_centering,
                                              with_scaling=with_scaling)

    return batched_time_series_data

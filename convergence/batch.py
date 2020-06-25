"""Batch module."""

import numpy as np

from .err import CVGError
from .stats import \
    translate_scale, \
    standard_scale, \
    robust_scale

__all__ = [
    'batch',
]

scale_methods = {
    'translate_scale': translate_scale,
    'standard_scale': standard_scale,
    'robust_scale': robust_scale
}


def batch(x,
          *,
          batch_size=5,
          scale='translate_scale',
          with_centering=False,
          with_scaling=False):
    r"""Batch the time series data.

    Args:
        x (array_like, 1d): Time series data.
        batch_size (int, optional): batch size. (default: {5})
        scale (str, optional): A method to standardize a dataset.
            (default: {'translate_scale'})
        with_centering (bool, optional): If True, use x minus the scale metod
            centering approach. (default: {False})
        with_scaling (bool, optional): If True, scale the data to scale
            metod scaling approach. (default: {False})

    Returns:
        1darray: Batched (, and rescaled) data.

    Note:
        This function will terminate the end of the data points which are
        remainder of the division of data points by the batch_size.

    """
    x = np.array(x, copy=False)

    # Check inputs
    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    if not isinstance(batch_size, int):
        msg = 'batch_size = {} is not an `int`.'.format(batch_size)
        raise CVGError(msg)
    elif batch_size < 1:
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
    n_batches = x.size // batch_size

    if n_batches == 0:
        msg = 'invalid number of batches = {}.\n'.format(n_batches)
        msg += 'The number of input data points = {} are '.format(x.size)
        msg += 'not enough to produce batches with the batch size of '
        msg += '{} data points.'.format(batch_size)
        raise CVGError(msg)

    # Correct the size of data
    n = n_batches * batch_size

    # Compute batch averages

    # The raw data is batched into non-overlapping batches of size batch_size
    z = np.matmul(x[:n].reshape([-1, batch_size]),
                  np.ones([batch_size], dtype=np.float64))

    # Calculate the batch means
    z /= np.float64(batch_size)

    if with_centering or with_scaling:
        z = scale_func(z,
                       with_centering=with_centering,
                       with_scaling=with_scaling)

    return z

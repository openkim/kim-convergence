"""Error module."""

import inspect
import numpy as np

__all__ = [
    'CVGError',
    'cvg_warning',
]


class CVGError(Exception):
    """Raise an exception.

    It raises an exception when receives an error message.

    msg (str): The error message.

    """

    def __init__(self, msg):
        """Constuctor."""
        _msg = '\nERROR(@' + \
            inspect.currentframe().f_back.f_code.co_name + '): ' + msg
        Exception.__init__(self, _msg)
        self.msg = _msg

    def __reduce__(self):
        """Efficient pickling."""
        return self.__class__, (self.msg)

    def __str__(self):
        """Message string representation."""
        return self.msg


def cvg_warning(msg):
    """Print a warning message.

    Args:
        msg (str): The warning message.

    """
    _msg = '\nWARNING(@' + \
        inspect.currentframe().f_back.f_code.co_name + '): ' + msg
    print(_msg)


def _check_ndim(func):
    def wrapper(x, *args, **kwargs):
        if np.ndim(x) != 1:
            msg = 'input data is not an array of one-dimension.'
            raise CVGError(msg)
        return func(x, *args, **kwargs)
    return wrapper


def _check_isfinite(func):
    def wrapper(x, *args, **kwargs):
        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input '
            msg += 'array which is non-finite or not-number.'
            raise CVGError(msg)
        return func(x, *args, **kwargs)
    return wrapper

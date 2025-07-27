"""Error module."""

import inspect
import numpy as np

__all__ = [
    'CRError',
    'CRSampleSizeError',
    'cr_warning',
    'cr_check'
]


class CRError(Exception):
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


class CRSampleSizeError(CRError):
    """Raise an exception if there is not enough samples."""
    pass


def cr_warning(msg: str) -> None:
    """Print a warning message.

    Args:
        msg (str): The warning message.

    """
    _msg = '\nWARNING(@' + \
        inspect.currentframe().f_back.f_code.co_name + '): ' + msg
    print(_msg)


def _check_ndim(func: callable) -> callable:
    def wrapper(x, *args, **kwargs):
        if np.ndim(x) != 1:
            raise CRError('input data is not an array of one-dimension.')
        return func(x, *args, **kwargs)
    return wrapper


def _check_isfinite(func: callable) -> callable:
    def wrapper(x, *args, **kwargs):
        if not np.all(np.isfinite(x)):
            raise CRError(
                'there is at least one value in the input array which is '
                'non-finite or not-number.'
            )
        return func(x, *args, **kwargs)
    return wrapper


def cr_check(var,
             var_name: str,
             var_type=None,
             var_lower_bound=0,
             var_upper_bound=None):
    """Check the variable type and lower bound.

    Args:
        var (var_type): variable
        var_name (str): variable name
        var_type (type, optional): variable type. (default: None)
        var_lower_bound (var_type, optional): variable lower bound.
            (default: 0)
        var_upper_bound (var_type, optional): variable upper bound.
            (default: None)

    """
    if var_upper_bound is None:
        if var_type is None:
            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be greater than or equal '
                    f'{var_lower_bound}.'
                )
        else:
            if not isinstance(var, var_type):
                raise CRError(f'"{var_name}" must be a `{var_type}`.')

            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` greater than '
                    f'or equal {var_lower_bound}.'
                )
    else:
        if var_type is None:
            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be greater than or '
                    f'equal {var_lower_bound}.'
                )

            if var > var_upper_bound:
                raise CRError(
                    f'"{var_name}" must be smaller than or '
                    f'equal {var_upper_bound}.'
                )
        else:
            if not isinstance(var, var_type):
                raise CRError(f'"{var_name}" must be a `{var_type}`.')

            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` greater than '
                    f'or equal {var_lower_bound}.'
                )

            if var > var_upper_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` smaller than '
                    f'or equal {var_upper_bound}.'
                )

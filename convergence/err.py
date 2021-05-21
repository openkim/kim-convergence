"""Error module."""

import inspect
import numpy as np

__all__ = [
    'CVGError',
    'CVGSampleSizeError',
    'cvg_warning',
    'cvg_check'
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


class CVGSampleSizeError(CVGError):
    """Raise an exception if there is not enough samples."""
    pass


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


def cvg_check(var,
              var_name,
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
                msg = '"{}" must be '.format(var_name)
                msg += 'greater than or equal {}.'.format(var_lower_bound)
                raise CVGError(msg)
        else:
            if not isinstance(var, var_type):
                msg = '"{}" must be a `{}`.'.format(var_name, var_type)
                raise CVGError(msg)

            if var < var_lower_bound:
                msg = '"{}" must be a `{}` '.format(var_name, var_type)
                msg += 'greater than or equal {}.'.format(var_lower_bound)
                raise CVGError(msg)
    else:
        if var_type is None:
            if var < var_lower_bound:
                msg = '"{}" must be '.format(var_name)
                msg += 'greater than or equal {}.'.format(var_lower_bound)
                raise CVGError(msg)

            if var > var_upper_bound:
                msg = '"{}" must be '.format(var_name)
                msg += 'smaller than or equal {}.'.format(var_upper_bound)
                raise CVGError(msg)
        else:
            if not isinstance(var, var_type):
                msg = '"{}" must be a `{}`.'.format(var_name, var_type)
                raise CVGError(msg)

            if var < var_lower_bound:
                msg = '"{}" must be a `{}` '.format(var_name, var_type)
                msg += 'greater than or equal {}.'.format(var_lower_bound)
                raise CVGError(msg)

            if var > var_upper_bound:
                msg = '"{}" must be a `{}` '.format(var_name, var_type)
                msg += 'smaller than or equal {}.'.format(var_upper_bound)
                raise CVGError(msg)

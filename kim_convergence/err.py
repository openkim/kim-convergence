r"""Error module."""

import inspect
import numpy as np
import sys
from typing import Callable

__all__ = ["CRError", "CRSampleSizeError", "cr_warning", "cr_check"]


def _get_caller_name(offset: int = 1) -> str:
    r"""
    Return the name of the calling function, offset steps up the stack.

    Args:
        offset: How many frames to go back. Default 1 = immediate caller of this function.

    Returns:
        Function name, or "<unknown>" if cannot be determined.
    """
    frame = inspect.currentframe()
    if frame is None:
        return "<unknown>"

    try:
        # Go back 'offset + 1' because this function itself is one frame
        caller_frame = frame
        for _ in range(offset + 1):
            caller_frame = caller_frame.f_back
            if caller_frame is None:
                return "<unknown>"

        return caller_frame.f_code.co_name
    finally:
        del frame


class CRError(Exception):
    r"""Raise an exception.

    It raises an exception when receives an error message.

    msg (str): The error message.

    """

    def __init__(self, msg):
        r"""Constuctor."""
        caller_name = _get_caller_name()
        _msg = f"\nERROR(@{caller_name}): {msg}"
        Exception.__init__(self, _msg)
        self.msg = _msg

    def __reduce__(self):
        r"""Efficient pickling."""
        return self.__class__, (self.msg,)

    def __str__(self):
        r"""Message string representation."""
        return self.msg


class CRSampleSizeError(CRError):
    r"""Raise an exception if there is not enough samples."""

    pass


def cr_warning(msg: str) -> None:
    r"""Print a warning message.

    Args:
        msg (str): The warning message.

    """
    caller_name = _get_caller_name()
    _msg = f"\nWARNING(@{caller_name}): {msg}"
    print(_msg, file=sys.stderr, flush=True)


def _check_ndim(func: Callable) -> Callable:
    def wrapper(x, *args, **kwargs):
        if np.ndim(x) != 1:
            raise CRError("input data is not an array of one-dimension.")
        return func(x, *args, **kwargs)

    return wrapper


def _check_isfinite(func: Callable) -> Callable:
    def wrapper(x, *args, **kwargs):
        if not np.all(np.isfinite(x)):
            raise CRError(
                "there is at least one value in the input array which is "
                "non-finite or not-number."
            )
        return func(x, *args, **kwargs)

    return wrapper


def cr_check(
    var, var_name: str, var_type=None, var_lower_bound=0, var_upper_bound=None
):
    r"""Check the variable type and lower bound.

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
                    f'"{var_name}" must be greater than or equal ' f"{var_lower_bound}."
                )
        else:
            if not isinstance(var, var_type):
                raise CRError(f'"{var_name}" must be a `{var_type}`.')

            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` greater than '
                    f"or equal {var_lower_bound}."
                )
    else:
        if var_type is None:
            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be greater than or ' f"equal {var_lower_bound}."
                )

            if var > var_upper_bound:
                raise CRError(
                    f'"{var_name}" must be smaller than or ' f"equal {var_upper_bound}."
                )
        else:
            if not isinstance(var, var_type):
                raise CRError(f'"{var_name}" must be a `{var_type}`.')

            if var < var_lower_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` greater than '
                    f"or equal {var_lower_bound}."
                )

            if var > var_upper_bound:
                raise CRError(
                    f'"{var_name}" must be a `{var_type}` smaller than '
                    f"or equal {var_upper_bound}."
                )

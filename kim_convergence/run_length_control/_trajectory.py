r"""
Trajectory acquisition utilities for run-length control.

This private module provides helpers for safely retrieving and validating new
simulation trajectory segments via the user-supplied ``get_trajectory`` callback.

Key features:
  - Validates that the callback is callable and follows the expected signature
  - Acquires trajectory data of requested length, supporting both single-variable
    (1-D) and multi-variable (2-D) return formats
  - Ensures returned data is a finite NumPy array with correct dimensionality
    and shape (rows = number_of_variables, columns = steps for multi-variable case)
  - Raises clear errors on invalid returns or non-finite values

All symbols in this module are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

import numpy as np
from typing import Callable

from kim_convergence import CRError

__all__ = []  # private module


def _check_get_trajectory(get_trajectory: Callable) -> None:
    if not callable(get_trajectory):
        raise CRError(
            'the "get_trajectory" input is not a callback function.\nOne '
            'has to provide the "get_trajectory" function as an input. It '
            "expects to have a specific signature:\nget_trajectory(nstep: "
            "int) -> 1darray,\nwhere nstep is the number of steps and the "
            "function should return a time-series data with the requested "
            "length equals to the number of steps."
        )


def _get_trajectory(
    get_trajectory: Callable,
    run_length: int,
    ndim: int,
    number_of_variables: int = 1,
    get_trajectory_args: dict = {},
) -> np.ndarray:
    if run_length == 0:
        return np.array([], dtype=np.float64)

    if isinstance(get_trajectory_args, dict) and get_trajectory_args:
        try:
            tsd = get_trajectory(run_length, get_trajectory_args)
        except Exception:  # noqa: BLE001  # intentional catch-all
            raise CRError(
                "failed to get the time-series data or do the simulation "
                f"for {run_length} number of steps."
            )
    else:
        try:
            tsd = get_trajectory(run_length)
        except Exception:  # noqa: BLE001  # intentional catch-all
            raise CRError(
                "failed to get the time-series data or do the simulation "
                f"for {run_length} number of steps."
            )

    tsd = np.asarray(tsd, dtype=np.float64)

    # Extra check
    if not np.all(np.isfinite(tsd)):
        raise CRError(
            "there is/are value/s in the input which is/are non-finite "
            "or not number."
        )

    if np.ndim(tsd) != ndim:
        raise CRError(
            'the return from the "get_trajectory" function has a wrong '
            f"dimension of {tsd.ndim} != {ndim}."
        )

    if ndim == 2 and number_of_variables != np.shape(tsd)[0]:
        raise CRError(
            'the return of "get_trajectory" function has a wrong number of '
            f"variables = {np.shape(tsd)[0]} != {number_of_variables}.\n"
            'In a two-dimensional return array of "get_trajectory" function, '
            "each row corresponds to the time series data for one variable."
        )

    return tsd

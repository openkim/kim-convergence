r"""
Variable-list factory for run-length control.

This module provides a single internal helper that converts user input into a
clean Python list of exact length `number_of_variables`, while:
  - Preserving arbitrary objects (lists, dicts, custom classes, etc.)
  - Replacing only non-finite *numeric scalars* (int/float/np.number) with None
  - Never crashing on strings, nested lists, or other objects
  - Raising clear errors on shape mismatch

The produced list is ready for downstream run-length control logic.

All symbols are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

import numpy as np
from typing import Any, Optional, Sequence, Union

from kim_convergence import CRError, cr_warning

__all__ = []  # private module


def _is_numeric_scalar(obj: Any) -> bool:
    """Return True if obj is a real-valued numeric scalar (not complex, not array)."""
    return (
        isinstance(obj, (int, float, np.number))
        and (not isinstance(obj, (bool, complex)))
        and np.isscalar(obj)
    )


def _clean_numeric_scalar(value: Any) -> Optional[Any]:
    """Replace non-finite numeric scalar with None; return others unchanged."""
    if value is None:
        return None
    if not _is_numeric_scalar(value):
        return value
    try:
        if not np.isfinite(value):
            return None
    except (TypeError, ValueError):
        cr_warning(f"np.isfinite failed for {value!r} (type {type(value).__name__})")
    return value


def _make_variable_list(
    value: Optional[Union[Sequence[Any], Any, np.ndarray]],
    number_of_variables: int,
) -> list[Any]:
    r"""
    Convert arbitrary input into a list of length exactly ``number_of_variables``.

    Behavior:
        - ``None`` -> ``[None] * number_of_variables``
        - Scalar (int, float, str, list, etc.) ->
          repeated: ``[scalar] * number_of_variables``
            - Only *numeric* scalars have inf/nan -> None applied
        - Sequence (list, tuple, ndarray) -> must have length ==
          number_of_variables
            - Each element cleaned (only numeric scalars affected)
            - Nested lists, dicts, custom objects preserved exactly

    Returns:
        list[Any]
            Length-exact list with non-finite numerics replaced by ``None``.

    Raises:
        CRError: On unsupported type or length mismatch.
    """
    if value is None:
        return [None] * number_of_variables

    # Scalar case: repeat it (with cleaning only if numeric)
    if isinstance(value, (int, float, str, np.generic)) or not isinstance(
        value, (list, tuple, np.ndarray)
    ):
        cleaned = _clean_numeric_scalar(value)
        return [cleaned] * number_of_variables

    # Sequence case: convert to list, check length, clean only numeric entries
    if isinstance(value, (tuple, np.ndarray)):
        try:
            seq = np.asarray(value).tolist()  # handles ndarray, including 0d/1d
        except Exception as e:
            raise CRError(f"Cannot convert input to list: {e}") from e
    elif isinstance(value, list):
        seq = value
    else:
        raise CRError(
            f"Unsupported input type {type(value).__name__}. "
            "Expected None, scalar, list, tuple, or ndarray."
        )

    if len(seq) != number_of_variables:
        raise CRError(
            f"Length mismatch: expected {number_of_variables} variables, "
            f"got sequence of length {len(seq)}."
        )

    # Clean only numeric scalars; leave everything else (str, list, None, objects) untouched
    return [_clean_numeric_scalar(item) for item in seq]

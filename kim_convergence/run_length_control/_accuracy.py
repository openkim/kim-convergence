r"""
Accuracy parameter validation for run-length control.

This private module performs semantic validation of the per-variable accuracy
specifications **after** they have been normalized into lists of length
``number_of_variables`` by the caller (typically via ``_make_variable_list``).

Validation rules:
  - At least one of ``relative_accuracy`` or ``absolute_accuracy`` must be
    specified for each variable.
  - All provided values must be non-negative.
  - When ``relative_accuracy`` is ``None``, ``absolute_accuracy`` must be
    ≥ _DEFAULT_MIN_ABSOLUTE_ACCURACY.

All symbols are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

from typing import Optional

from kim_convergence import CRError, cr_check
from kim_convergence._default import _DEFAULT_MIN_ABSOLUTE_ACCURACY


__all__ = []  # private module


def _check_accuracy(
    number_of_variables: int,
    relative_accuracy: list[Optional[float]],
    absolute_accuracy: list[Optional[float]],
) -> None:
    r"""
    Validate per-variable accuracy specifications after normalization.

    Ensures that for each variable:
      - At least one of relative or absolute accuracy is provided
      - Values are non-negative
      - Absolute accuracy meets the minimum threshold when relative accuracy is absent

    Raises:
        CRError: If any per-variable accuracy rule is violated (both None,
            negative values, or absolute_accuracy below minimum when
            relative_accuracy is None).
    """
    if len(relative_accuracy) != number_of_variables:
        raise CRError("Internal error: relative_accuracy list length mismatch.")
    if len(absolute_accuracy) != number_of_variables:
        raise CRError("Internal error: absolute_accuracy list length mismatch.")

    for i in range(number_of_variables):
        rela = relative_accuracy[i]
        absa = absolute_accuracy[i]

        # Both None is not allowed
        if rela is None and absa is None:
            raise CRError(
                f"For variable {i}: at least one of 'relative_accuracy' or "
                "'absolute_accuracy' must be specified (not both None)."
            )

        # If relative_accuracy is None -> absolute_accuracy must be meaningful
        if rela is None:
            cr_check(
                absa,
                f"absolute_accuracy[{i}]",
                var_lower_bound=_DEFAULT_MIN_ABSOLUTE_ACCURACY,  # type: ignore[assignment]
            )
        else:
            # rela is not None -> must be ≥ 0.0
            if rela < 0.0:
                raise CRError(
                    f"relative_accuracy[{i}] = {rela} is negative. "
                    "Must be ≥ 0.0 or None."
                )

        # If absolute_accuracy is specified, it must be ≥ 0.0
        if absa is not None:
            if absa < 0.0:
                raise CRError(
                    f"absolute_accuracy[{i}] = {absa} is negative. "
                    "Must be ≥ 0.0 or None."
                )

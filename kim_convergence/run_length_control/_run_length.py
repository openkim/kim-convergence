r"""
Run-length adjustment utilities for run-length control.

This private module provides a helper function to dynamically compute the next
trajectory segment length during adaptive simulation extension.

Key features:
  - Applies a user-specified growth factor to the previous run length
  - Ensures the result is a positive integer (at least 1)
  - Caps the length to respect the remaining budget up to maximum_run_length
  - Handles edge cases: zero initial length, or when the maximum is already reached

All symbols in this module are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

__all__ = []  # private module


def _get_run_length(
    run_length: int,
    run_length_factor: float,
    total_run_length: int,
    maximum_run_length: int,
) -> int:
    if total_run_length >= maximum_run_length:
        return 0

    # apply growth factor
    candidate = int(run_length * run_length_factor)
    candidate = max(candidate, 1)

    # respect the remaining budget
    remaining = maximum_run_length - total_run_length
    return min(candidate, remaining)

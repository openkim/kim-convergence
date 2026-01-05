r"""
Population parameter validation utilities for run-length control.

This private helper validates population distribution specifications **after**
all inputs have been normalized by :func:`_make_variable_list`.

At entry:
  - All arguments are Python lists of length exactly ``number_of_variables``
  - Non-finite numerics have been replaced with ``None``
  - Length/shape errors have already been caught

This function performs only **semantic/logical** validation:
  - If population_cdf is given -> mean/std must be None, loc/scale allowed
  - If population_cdf is None -> assumes normal -> mean and std > 0 required
  - Validates each distribution name + its arguments via SciPy/Stats

All symbols are private to ``kim_convergence.run_length_control``.
"""

import numpy as np
from typing import cast, Optional, Sequence

from kim_convergence import (
    CRError,
    check_population_cdf_args,
    chi_square_test,
    levene_test,
    t_test,
)
from kim_convergence.ucl import UCLBase

__all__ = []  # private module


def _validate_population_params(
    number_of_variables: int,
    population_mean: Sequence[Optional[float]],
    population_standard_deviation: Sequence[Optional[float]],
    population_cdf: Sequence[Optional[str]],
    population_args: Sequence[Optional[tuple]],
    population_loc: Sequence[Optional[float]],
    population_scale: Sequence[Optional[float]],
) -> None:
    r"""
    Enforce mutual exclusivity and validity rules for population parameters.

    Validates that:
      - Either a custom distribution (via population_cdf) is specified with
        loc/scale/args, and mean/std are None
      - Or no cdf is given → normal distribution assumed, requiring finite mean
        and std > 0, with args/loc/scale forbidden

    Raises:
        CRError: If lengths mismatch or any semantic rule is violated.
    """
    if not (
        len(population_mean)
        == len(population_standard_deviation)
        == len(population_cdf)
        == len(population_args)
        == len(population_loc)
        == len(population_scale)
        == number_of_variables
    ):
        raise CRError(
            "Internal error: population parameter lists have inconsistent lengths."
        )

    for i in range(number_of_variables):
        mean = population_mean[i]
        std = population_standard_deviation[i]
        cdf_name = population_cdf[i]
        args = population_args[i]
        loc = population_loc[i]
        scale = population_scale[i]

        # Case A: User specified a known distribution via population_cdf
        if cdf_name is not None:
            args = () if args is None else args
            # Validate distribution name + its arguments
            try:
                check_population_cdf_args(population_cdf=cdf_name, population_args=args)
            except Exception as e:
                raise CRError(
                    f"Variable {i + 1}: invalid distribution specification: {e}"
                ) from e

            # For custom distributions: mean and std must NOT be provided
            if mean is not None:
                raise CRError(
                    f"Variable {i + 1}: population_mean must be None when "
                    f"population_cdf='{cdf_name}' is given. Use population_loc "
                    "and population_scale instead."
                )
            if std is not None:
                raise CRError(
                    f"Variable {i + 1}: population_standard_deviation must be "
                    f"None when population_cdf='{cdf_name}' is given. Use "
                    "population_scale to control spread."
                )

            # loc and scale are optional but must be valid if given
            if loc is not None and not np.isfinite(loc):
                raise CRError(
                    f"Variable {i + 1}: population_loc must be finite (got "
                    f"{loc})."
                )
            if scale is not None:
                if not np.isfinite(scale) or scale <= 0:
                    raise CRError(
                        f"Variable {i + 1}: population_scale must be > 0 (got "
                        f"{scale})."
                    )

        # Case B: No distribution specified -> assume normal distribution
        else:
            # args, loc and scale are not allowed for normal case
            if args is not None:
                raise CRError(
                    f"Variable {i + 1}: population_args cannot be used without "
                    "population_cdf."
                )
            if loc is not None:
                raise CRError(
                    f"Variable {i + 1}: population_loc cannot be used without "
                    "population_cdf."
                )
            if scale is not None:
                raise CRError(
                    f"Variable {i + 1}: population_scale cannot be used "
                    "without population_cdf."
                )

            # mean must be finite if provided
            if mean is not None and not np.isfinite(mean):
                raise CRError(
                    f"Variable {i + 1}: population_mean must be finite (got "
                    f"{mean})."
                )

            # std must be finite, and positive, if provided
            if std is not None and (not np.isfinite(std) or std <= 0):
                raise CRError(
                    f"Variable {i + 1}: population_standard_deviation must be "
                    f"finite and > 0 (got {std})."
                )


def _population_tests(
    ucl_obj: UCLBase,
    time_series_data: np.ndarray,
    confidence_coefficient: float,
    population_mean: Optional[float],
    population_std: Optional[float],
    population_cdf: Optional[str],
    population_args: Optional[tuple],
    population_loc: Optional[float],
    population_scale: Optional[float],
) -> bool:
    r"""
    Run optional statistical tests against population parameters for one variable.

    Executes t-test, chi-square test, and Levene test if the corresponding
    population parameters are provided. Short-circuits on the first failure.

    Returns:
        bool
            - True if all enabled tests pass (sample consistent with
              population).
            - False if any test fails (more data needed).
    """
    sig = 1.0 - confidence_coefficient
    ok = True

    # 1. t-test against known mean
    if population_mean is not None:
        ok = t_test(
            sample_mean=cast(float, ucl_obj.mean),
            sample_std=cast(float, ucl_obj.std),
            sample_size=cast(int, ucl_obj.sample_size),
            population_mean=population_mean,
            significance_level=sig,
        )

    # 2.  chi-square test against known variance
    if ok and population_std is not None:
        ok = chi_square_test(
            sample_var=cast(float, ucl_obj.std) ** 2,
            sample_size=cast(int, ucl_obj.sample_size),
            population_var=population_std**2,
            significance_level=sig,
        )

    # 3. Levene test against known distribution
    if ok and population_cdf is not None:
        args = () if population_args is None else population_args
        ok = levene_test(
            time_series_data=time_series_data[ucl_obj.indices],
            population_cdf=population_cdf,
            population_args=args,
            population_loc=population_loc,
            population_scale=population_scale,
            significance_level=sig,
        )

    return ok

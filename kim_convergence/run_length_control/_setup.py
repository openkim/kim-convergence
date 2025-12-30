r"""
Initial setup and validation for the run-length control algorithm.

This private module handles the early-stage preparation of the run-length control
algorithm before entering the equilibration and convergence phases. It is
responsible for:

  - Validating the core scalar input parameters (types, bounds, consistency).
  - Checking the validity of the ``get_trajectory`` callback function.
  - Applying default behavior and issuing warnings for ``maximum_equilibration_step``.
  - Enforcing the hard limit relationship between equilibration and total run length.
  - Validating optional minimum independent sample requirement.
  - Instantiating and configuring the upper confidence limit (UCL) estimator object
    based on the requested ``confidence_interval_approximation_method``.

The function returns the (possibly modified) ``maximum_equilibration_step`` and the
fully configured ``ucl_obj`` ready for use in later stages.

All symbols are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

from typing import Callable, Optional

from kim_convergence import CRError, cr_check, cr_warning, ucl_methods
from kim_convergence.ucl import UCLBase

from ._trajectory import _check_get_trajectory

__all__ = []  # private module


def _setup_algorithm(
    get_trajectory: Callable,
    number_of_variables: int,
    initial_run_length: int,
    run_length_factor: float,
    maximum_run_length: int,
    maximum_equilibration_step: Optional[int],
    minimum_number_of_independent_samples: Optional[int],
    confidence_interval_approximation_method: str,
    confidence_coefficient: float,
    heidel_welch_number_points: int,
    number_of_cores: int,
    minimum_correlation_time: Optional[int],
) -> tuple[int, UCLBase]:
    r"""
    Perform initial validation and setup of parameters and UCL estimator.

    This function validates fundamental inputs, applies defaults where needed,
    and initializes the UCL object that will be used throughout the convergence
    stage. It is executed once at the beginning of ``run_length_control``.

    Returns
    -------
    maximum_equilibration_step : int
        The (possibly defaulted) hard limit for equilibration detection.
    ucl_obj : UCLBase
        Instantiated and configured UCL estimator object.

    Raises
    ------
    CRError
        If any input validation or UCL instantiation fails.
    """

    _check_get_trajectory(get_trajectory)

    cr_check(number_of_variables, "number_of_variables", int, 1)
    cr_check(initial_run_length, "initial_run_length", int, 1)
    cr_check(run_length_factor, "run_length_factor", float, 0)
    cr_check(maximum_run_length, "maximum_run_length", int, 1)
    cr_check(confidence_coefficient, "confidence_coefficient", float, 0, 1)
    cr_check(heidel_welch_number_points, "heidel_welch_number_points", int, 25)
    cr_check(number_of_cores, "number_of_cores", int, 1)
    if minimum_correlation_time is not None:
        cr_check(minimum_correlation_time, "minimum_correlation_time", int, 1)

    if maximum_equilibration_step is None:
        maximum_equilibration_step = maximum_run_length // 2
        cr_warning(
            '"maximum_equilibration_step" is not given on input!\nThe '
            "maximum number of steps as an equilibration hard limit "
            f"is set to {maximum_equilibration_step}."
        )

    # Set the hard limit for the equilibration step
    cr_check(
        maximum_equilibration_step,
        "maximum_equilibration_step",
        int,
        1,
        maximum_run_length - 1,
    )

    if minimum_number_of_independent_samples is not None:
        cr_check(
            minimum_number_of_independent_samples,
            "minimum_number_of_independent_samples",
            int,
            1,
        )

    # UCL object
    if confidence_interval_approximation_method not in ucl_methods:
        raise CRError(
            f'method "{confidence_interval_approximation_method}" to '
            "aproximate confidence interval not found. Valid methods are:"
            "\n\t- " + "\n\t- ".join(ucl_methods)
        )

    try:
        ucl_obj = ucl_methods[confidence_interval_approximation_method]()
    except CRError:
        raise CRError("Failed to initialize the UCL object.")

    if ucl_obj.name == "heidel_welch":
        ucl_obj.set_heidel_welch_constants(
            confidence_coefficient=confidence_coefficient,
            heidel_welch_number_points=heidel_welch_number_points,
        )

    return maximum_equilibration_step, ucl_obj

r"""
Convergence stage, report generation, and output for run-length control.

This private module implements the **convergence phase** of the run-length control
algorithm, which follows equilibration detection. It is responsible for:

  - Running the sequential convergence loop that extends the simulation until
    all variables meet the specified statistical criteria (accuracy on the
    confidence interval half-width, minimum independent samples, and optional
    population hypothesis tests).
  - Computing per-variable upper confidence limits (UCL), effective sample sizes,
    and final statistics.
  - Handling unified behavior for both single-variable and multi-variable cases.
  - Building the final structured convergence report dictionary.
  - Serializing and outputting (or returning) the report in JSON or EDN format.

The module contains the main convergence logic, tightly coupled helpers,
the report builder, and output handling.

All symbols are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

import json
import kim_edn
from math import isclose, fabs
import numpy as np
import sys
from typing import Any, cast, Callable, Optional, Sequence, Union

from kim_convergence import CRError, CRSampleSizeError, cr_warning
from kim_convergence._default import _DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL
from kim_convergence.ucl import UCLBase

from ._population import _population_tests
from ._run_length import _get_run_length
from ._trajectory import _get_trajectory

__all__ = []  # private module


def _convergence_message(
    number_of_variables: int,
    converged: bool,
    total_run_length: int,
    maximum_equilibration_step: int,
    equilibration_detected: bool,
    equilibration_step: list[int],
    confidence_coefficient: float,
    relative_accuracy: list[Optional[float]],
    absolute_accuracy: list[Optional[float]],
    upper_confidence_limit: list[Optional[float]],
    upper_confidence_limit_method: str,
    relative_half_width_estimate: list[float],
    time_series_data_mean: list[Optional[float]],
    time_series_data_std: list[Optional[float]],
    effective_sample_size: list[float],
    minimum_number_of_independent_samples: Optional[int],
    relative_accuracy_undefined: list[bool],
) -> dict:
    r"""
    Build a structured dictionary summarizing convergence results.

    Producing a consistent report suitable for JSON/EDN serialization or printing.

    Returns
    -------
    dict
        Convergence report with keys such as 'converged', 'total_run_length',
        'equilibration_detected', per-variable statistics, confidence intervals,
        and accuracy details.

    Raises
    ------
    AssertionError
        If input types or shapes are inconsistent (defensive checks only).
    """
    eq_step = equilibration_step
    rel_acc = relative_accuracy
    abs_acc = absolute_accuracy
    rel_hw = relative_half_width_estimate
    ucl = upper_confidence_limit
    mean = time_series_data_mean
    std = time_series_data_std
    ess = effective_sample_size
    rau = relative_accuracy_undefined

    assert isinstance(eq_step, list) and len(eq_step) == number_of_variables
    assert isinstance(rel_acc, list) and len(rel_acc) == number_of_variables
    assert isinstance(abs_acc, list) and len(abs_acc) == number_of_variables
    assert isinstance(rel_hw, list) and len(rel_hw) == number_of_variables
    assert isinstance(mean, list) and len(mean) == number_of_variables
    assert isinstance(ucl, list) and len(ucl) == number_of_variables
    assert isinstance(std, list) and len(std) == number_of_variables
    assert isinstance(ess, list) and len(ess) == number_of_variables
    assert isinstance(rau, list) and len(rau) == number_of_variables

    confidence = f"{round(confidence_coefficient * 100, 3)}%"

    msg = {
        "converged": converged,
        "total_run_length": total_run_length,
        "maximum_equilibration_step": maximum_equilibration_step,
        "equilibration_detected": equilibration_detected,
    }

    rss = (
        minimum_number_of_independent_samples
        if minimum_number_of_independent_samples is not None
        else "None"
    )

    for i in range(number_of_variables):
        eq_step_i = int(eq_step[i])

        # absolute mode
        if rel_acc[i] is None:
            abs_acc_i = abs_acc[i]
            rel_acc_i = "None"
            rel_hw_i = "None"
        else:
            abs_acc_i = "None"
            rel_acc_i = float(cast(float, rel_acc[i]))
            rel_hw_i = float(rel_hw[i])

        mean_i = float(cast(float, mean[i]))
        std_i = float(cast(float, std[i]))
        ucl_i = float(cast(float, ucl[i]))
        ess_i = float(cast(float, ess[i]))
        ci_i = (
            f"{confidence} confidence interval for the mean is "
            f"({mean_i - ucl_i}, {mean_i + ucl_i})"
        )

        var_dict = {
            "equilibration_detected": equilibration_detected,
            "equilibration_step": eq_step_i,
            "confidence": confidence,
            "relative_accuracy": rel_acc_i,
            "absolute_accuracy": abs_acc_i,
            "relative_half_width": rel_hw_i,
            "upper_confidence_limit": ucl_i,
            "upper_confidence_limit_method": upper_confidence_limit_method,
            "mean": mean_i,
            "standard_deviation": std_i,
            "effective_sample_size": ess_i,
            "requested_sample_size": rss,
            "confidence_interval": ci_i,
        }
        if rau[i]:
            var_dict["relative_accuracy_undefined"] = True

        # For single variable: flatten into top-level dict
        # For multiple variables: nest under "0", "1", ...
        if number_of_variables == 1:
            msg.update(var_dict)
        else:
            msg[str(i)] = var_dict

    return msg


def _equilibrated_series(
    tsd: np.ndarray, ndim: int, equilibration_step: int, var_idx: int
) -> np.ndarray:
    r"""
    Return a view of the equilibrated portion of the time series for the given variable.

    The view shares memory with the original ``tsd`` array.
    For single-variable cases (ndim == 1), ``var_idx`` is ignored.
    """
    return tsd[var_idx, equilibration_step:] if ndim == 2 else tsd[equilibration_step:]


def _compute_ucl_and_check_accuracy(
    i: int,
    ucl_obj: UCLBase,
    time_series_data: np.ndarray,
    confidence_coefficient: float,
    heidel_welch_number_points: int,
    batch_size: int,
    fft: bool,
    scale: str,
    with_centering: bool,
    with_scaling: bool,
    test_size: Any,
    train_size: Any,
    population_standard_deviation: Optional[float],
    si: str,
    minimum_correlation_time: Optional[int],
    relative_accuracy: Optional[float],
    absolute_accuracy: Optional[float],
    final_pass: bool = False,
) -> tuple[Optional[float], float, bool]:
    r"""
    Compute the upper confidence limit (UCL) for a single variable and determine
    whether the required statistical accuracy has been achieved.

    This internal helper performs the core accuracy check during the convergence
    loop. It calculates the half-width of the confidence interval using the
    provided UCL estimator, handles cases of insufficient sample size, and
    evaluates either relative or absolute accuracy criteria. When accuracy is
    satisfied (and not in final_pass mode), it also initializes the statistical
    inefficiency indices needed for subsequent effective sample size estimation.

    In ``final_pass=True`` mode the function bypasses the accuracy check entirely
    and only computes the final UCL and (if required) the statistical inefficiency
    value for reporting the best available estimates when full convergence is not
    reached.

    Returns
    -------
    upper_confidence_limit : Optional[float]
        The computed half-width of the confidence interval.
        ``None`` if the sample size was insufficient (CRSampleSizeError).
    relative_half_width_estimate : float
        Estimated relative half-width when relative accuracy is requested;
        0.0 when absolute accuracy is used or in final_pass mode.
    accurate : bool
        ``True`` if the accuracy criterion is met (ignored in final_pass mode,
        where it is always ``True``).

    Raises
    ------
    CRError
        If UCL computation fails for reasons other than insufficient sample size,
        or if relative accuracy is requested but the estimated mean is too close
        to zero to allow a meaningful relative estimate.
    """
    try:
        upper_confidence_limit = ucl_obj.ucl(
            time_series_data,
            confidence_coefficient=confidence_coefficient,
            equilibration_length_estimate=0,
            heidel_welch_number_points=heidel_welch_number_points,
            batch_size=batch_size,
            fft=fft,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            test_size=test_size,
            train_size=train_size,
            population_standard_deviation=population_standard_deviation,
            si=si,
            minimum_correlation_time=minimum_correlation_time,
        )
    except CRSampleSizeError:
        # do not have enough data and need more
        return (None, 0.0, False)
    except CRError as e:
        raise CRError(
            f"Failed to compute the ucl for variable number = {i + 1}."
        ) from e

    if final_pass:
        if ucl_obj.name != "uncorrelated_sample":
            ucl_obj.set_si(
                time_series_data,
                si=si,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
            )
        return (upper_confidence_limit, 0.0, True)

    if relative_accuracy is None:
        accurate = upper_confidence_limit < cast(float, absolute_accuracy)
        relative_half_width_estimate = 0.0
    else:
        assert isinstance(ucl_obj.mean, float)  # keeps mypy happy
        # Estimate the relative half width
        if isclose(
            ucl_obj.mean, 0, abs_tol=_DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL
        ):
            raise CRError(
                "It is not possible to estimate the relative half width for "
                f"the close to zero mean = {ucl_obj.mean}, for the variable "
                f"number = {i + 1}. Consider using absolute_accuracy instead."
            )

        relative_half_width_estimate = upper_confidence_limit / fabs(ucl_obj.mean)

        accurate = relative_half_width_estimate < cast(float, relative_accuracy)

    if accurate and ucl_obj.name != "uncorrelated_sample":
        ucl_obj.set_indices(
            time_series_data,
            si=si,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
        )

    return (upper_confidence_limit, relative_half_width_estimate, accurate)


def _convergence_stage(
    get_trajectory: Callable,
    get_trajectory_args: dict,
    number_of_variables: int,
    tsd: np.ndarray,
    equilibration_step: list[int],
    run_length: int,
    total_run_length: int,
    run_length_factor: float,
    maximum_run_length: int,
    minimum_number_of_independent_samples: Optional[int],
    minimum_correlation_time: Optional[int],
    relative_accuracy_list: Sequence[Optional[float]],
    absolute_accuracy_list: Sequence[Optional[float]],
    population_mean_list: Sequence[Optional[float]],
    population_standard_deviation_list: Sequence[Optional[float]],
    population_cdf_list: Sequence[Optional[str]],
    population_args_list: Sequence[Optional[tuple]],
    population_loc_list: Sequence[Optional[float]],
    population_scale_list: Sequence[Optional[float]],
    ucl_obj: UCLBase,
    confidence_coefficient: float,
    heidel_welch_number_points: int,
    fft: bool,
    batch_size: int,
    scale: str,
    with_centering: bool,
    with_scaling: bool,
    test_size: Any,
    train_size: Any,
    si: str,
    dump_trajectory: bool,
    dump_trajectory_fp: str,
) -> tuple[
    bool,  # converged
    int,  # total_run_length
    list[Optional[float]],  # mean
    list[Optional[float]],  # std
    list[float],  # effective_sample_size
    list[Optional[float]],  # upper_confidence_limit
    list[float],  # relative_half_width_estimate
    list[bool],  # relative_accuracy_undefined
]:
    r"""
    Execute the accuracy/convergence stage after equilibration.

    This function runs the unified convergence loop that extends the simulation
    until all variables satisfy:
      - The specified relative or absolute accuracy criterion
      - The minimum number of independent samples (if requested)
      - Optional population distribution tests (t-test, chi-square, Levene)

    If any variable lacks sufficient data for UCL computation, the run is
    extended for all variables. Convergence requires *all* variables to pass
    their criteria.

    When convergence is not achieved (e.g., maximum_run_length reached),
    final statistics are computed for reporting the best available estimates.

    Returns
    -------
    converged : bool
        True if all variables fully converged.
    total_run_length: int
        Total number of steps in the trajectory.
    mean : list[Optional[float]]
        Estimated means (None only if UCL computation failed entirely).
    std : list[Optional[float]]
        Estimated standard deviations.
    effective_sample_size : list[float]
        Effective sample sizes (N / statistical inefficiency).
    upper_confidence_limit : list[Optional[float]]
        Final upper confidence limits (half-widths).
    relative_half_width_estimate : list[float]
        Relative half-width estimates (0.0 when absolute accuracy was used).
    relative_accuracy_undefined : list[bool]
        True for variables whose CI covers zero.
    """

    ndim: int = tsd.ndim

    done: list[bool] = [False] * number_of_variables
    mean: list[Optional[float]] = [None] * number_of_variables
    std: list[Optional[float]] = [None] * number_of_variables
    effective_sample_size: list[float] = [0.0] * number_of_variables
    upper_confidence_limit: list[Optional[float]] = [None] * number_of_variables
    relative_half_width_estimate: list[float] = [0.0] * number_of_variables
    relative_accuracy_undefined: list[bool] = [False] * number_of_variables

    converged = False
    while not converged:
        # Reset convergence flags for all variables at the start of each iteration
        for i in range(number_of_variables):
            done[i] = False

            # slice a numpy array, the memory is shared between the slice and the original
            time_series_data = _equilibrated_series(tsd, ndim, equilibration_step[i], i)
            time_series_data_size = time_series_data.size

            (
                upper_confidence_limit[i],
                relative_half_width_estimate[i],
                enough_accuracy,
            ) = _compute_ucl_and_check_accuracy(
                i,
                ucl_obj,
                time_series_data,
                confidence_coefficient,
                heidel_welch_number_points,
                batch_size,
                fft,
                scale,
                with_centering,
                with_scaling,
                test_size,
                train_size,
                population_standard_deviation_list[i],
                si,
                minimum_correlation_time,
                relative_accuracy_list[i],
                absolute_accuracy_list[i],
            )

            if upper_confidence_limit[i] is None:
                break

            if enough_accuracy:
                assert isinstance(ucl_obj.si, float)  # keeps mypy happy
                effective_sample_size[i] = time_series_data_size / ucl_obj.si

                if (
                    minimum_number_of_independent_samples is None
                    or effective_sample_size[i] >= minimum_number_of_independent_samples
                ):
                    if _population_tests(
                        ucl_obj,
                        time_series_data,
                        confidence_coefficient,
                        population_mean_list[i],
                        population_standard_deviation_list[i],
                        population_cdf_list[i],
                        population_args_list[i],
                        population_loc_list[i],
                        population_scale_list[i],
                    ):
                        mean[i] = ucl_obj.mean
                        std[i] = ucl_obj.std
                        done[i] = True

        # Python for-else: the else clause executes only if the loop completes
        # without breaking. Here it means all variables computed UCL successfully.
        else:
            converged = all(done)
            if converged:
                break

        # get the run length
        run_length = _get_run_length(
            run_length, run_length_factor, total_run_length, maximum_run_length
        )

        # We have reached the maximum limit
        if run_length == 0:
            break

        total_run_length += run_length
        ext_tsd = _get_trajectory(
            get_trajectory,
            run_length=run_length,
            ndim=ndim,
            number_of_variables=number_of_variables,
            get_trajectory_args=get_trajectory_args,
        )
        tsd = np.concatenate((tsd, ext_tsd), axis=ndim - 1)

    failed = [str(i + 1) for i, u in enumerate(upper_confidence_limit) if u is None]
    if failed:
        raise CRError(
            f'For variable number(s) {", ".join(failed)}. Failed to '
            "compute the UCL."
        )

    if dump_trajectory:
        kim_edn.dump(tsd.tolist(), dump_trajectory_fp)

    if not converged:
        for i in range(number_of_variables):
            if not done[i]:
                # slice a numpy array, the memory is shared
                # between the slice and the original
                time_series_data = _equilibrated_series(
                    tsd, ndim, equilibration_step[i], i
                )
                time_series_data_size = time_series_data.size

                upper_confidence_limit[i], _, _ = _compute_ucl_and_check_accuracy(
                    i,
                    ucl_obj,
                    time_series_data,
                    confidence_coefficient,
                    heidel_welch_number_points,
                    batch_size,
                    fft,
                    scale,
                    with_centering,
                    with_scaling,
                    test_size,
                    train_size,
                    population_standard_deviation_list[i],
                    si,
                    minimum_correlation_time,
                    None,
                    None,
                    True,
                )

                mean[i] = ucl_obj.mean
                std[i] = ucl_obj.std
                assert isinstance(ucl_obj.si, float)  # keeps mypy happy
                effective_sample_size[i] = time_series_data_size / ucl_obj.si

                if relative_accuracy_list[i] is not None and abs(
                    cast(float, mean[i])
                ) <= cast(float, upper_confidence_limit[i]):
                    relative_accuracy_undefined[i] = True
                    cr_warning(
                        f"Variable {i + 1}: confidence interval includes zero, meaning relative "
                        "accuracy is ill-defined. Consider using absolute_accuracy instead."
                    )

    return (
        converged,
        total_run_length,
        mean,
        std,
        effective_sample_size,
        upper_confidence_limit,
        relative_half_width_estimate,
        relative_accuracy_undefined,
    )


def _output_convergence_report(
    cmsg: dict, converged: bool, fp: Any, fp_format: str
) -> Union[str, bool]:
    r"""
    Serialize and output (or return) the final convergence report.

    Handles writing the convergence message to a file-like object or returning
    it as a string, according to ``fp`` and ``fp_format``.

    Returns
    -------
    str
        Serialized report if ``fp == "return"``.
    bool
        ``converged`` if output was written to ``fp`` or stdout.

    Raises
    ------
    CRError
        If ``fp`` or ``fp_format`` is invalid.
    """
    if fp is None:
        fp = sys.stdout
    elif isinstance(fp, str):
        if fp != "return":
            raise CRError(
                'Keyword argument `fp` is a `str` and not equal to "return".'
            )
        fp = None
    elif not hasattr(fp, "write"):
        raise CRError(
            "Keyword argument `fp` must be either a `str` and equal to "
            '"return", or None, or an object with write(string) method.'
        )

    if fp_format not in ("txt", "json", "edn"):
        raise CRError(
            "fp format is unknown. Valid formats are:\n\t- "
            + "\n\t- ".join(("txt", "json", "edn"))
        )

    # It should return the string
    if fp is None:
        if fp_format == "json":
            return json.dumps(cmsg, indent=4)

        return kim_edn.dumps(cmsg, indent=4)

    # Otherwise it uses fp to print the message
    if fp_format == "json":
        json.dump(cmsg, fp, indent=4)
        return converged

    kim_edn.dump(cmsg, fp, indent=4)
    return converged

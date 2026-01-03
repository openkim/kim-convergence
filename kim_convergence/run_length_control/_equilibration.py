r"""
Equilibration detection and validation for run-length control.

This private module implements the equilibration (warm-up) detection stage of the
run-length control algorithm. Its main responsibility is to determine, for each
observed variable, how many initial steps must be discarded before the time series
reaches stationarity.

Key features:
  - Incremental trajectory acquisition with exponential run-length growth (capped by maximum_run_length)
  - Primary detection via Marginal Standard Error Rule (MSER)
  - Optional consistency check against known population parameters (3-sigma rule)
  - Refinement of truncation point using integrated autocorrelation time
  - Hard limit enforcement on equilibration length with clear error messages
  - Optional trajectory dump on detection failure for debugging
  - Full support for single- and multi-variable time series

All symbols in this module are private implementation details of the
``kim_convergence.run_length_control`` package.
"""

import kim_edn
import numpy as np
from typing import Callable, Optional, Sequence, Union

from kim_convergence import (
    CRError,
    get_distribution_stats,
    mser_m,
    estimate_equilibration_length,
)

from ._run_length import _get_run_length
from ._trajectory import _get_trajectory

__all__ = []  # private module


def _check_equilibration_step(
    equilibration_step: Union[int, list[int]],
    maximum_equilibration_step: int,
    maximum_run_length: int,
    equilibration_detected: bool,
    time_series_data: np.ndarray,
    dump_trajectory: bool,
    dump_trajectory_fp,
    method: str,
) -> None:
    equilibration_step_array = np.atleast_1d(equilibration_step).astype(int, copy=False)

    # hard limit
    if np.all(equilibration_step_array < maximum_equilibration_step):
        return

    if dump_trajectory:
        kim_edn.dump(time_series_data.tolist(), dump_trajectory_fp)

    msgs = ["\n"]
    number_of_variables = equilibration_step_array.size

    # build per-item messages
    for i, step in enumerate(equilibration_step_array):
        prefix = "" if number_of_variables == 1 else f"for variable number {i + 1} "
        if step < maximum_equilibration_step:
            msgs.append(
                f'The equilibration or "warm-up" period {prefix}is '
                f"detected at step = {step}.\n"
            )
            continue

        if step == maximum_run_length - 1:
            msgs.append(
                f'The equilibration or "warm-up" period {prefix}is not '
                "detected. Check the trajectory data!\n"
            )
        else:
            msgs.append(
                f"The truncation point {prefix}= {step}, returned by {method} is "
                f">= maximum_equilibration_step ({maximum_equilibration_step}) "
                "and is therefore invalid.\n"
            )

    # append global advice
    if equilibration_detected:
        msgs.append(
            "To prevent this error, you can either request a longer maximum "
            "number of allowed steps to reach equilibrium or if you did not "
            "provide this limit you can increase the maximum_run_length.\n"
        )
    else:
        msgs.append(
            "More data is required. To prevent this error, you can request a "
            "longer maximum_run_length.\n"
        )

    raise CRError("".join(msgs))


def _is_within_3sigma(
    sample: np.ndarray,
    pop_mean: Optional[float],
    pop_std: Optional[float],
    pop_cdf: Optional[str],
    pop_args: Optional[tuple],
    pop_loc: Optional[float],
    pop_scale: Optional[float],
) -> bool:
    """
    Return True if sample mean/median is within 3 sigma of population expectation.

    Preconditions (enforced by _validate_population_params):
        - cdf is not None -> mean & std are None
        - cdf is None     -> mean & std are optional and args/loc/scale are None
    """

    if pop_cdf is not None:
        pop_median, _, _, pop_std = get_distribution_stats(
            pop_cdf, pop_args or (), pop_loc, pop_scale
        )

        diff = abs(np.median(sample) - pop_median) if np.isfinite(pop_median) else 0.0
    else:
        if pop_mean is None:
            diff = 0.0
        else:
            diff = abs(sample.mean() - pop_mean)

    sample_std = sample.std()
    if pop_std is not None and np.isfinite(pop_std) and pop_std > 0:
        sample_std = max(sample_std, pop_std)

    return sample_std <= 0 or diff <= 3.0 * sample_std


def _truncated_series(
    tsd: np.ndarray, ndim: int, truncate_index: int, var_idx: int
) -> np.ndarray:
    r"""
    Return a view of the truncated portion of the time series for the given variable.

    The view shares memory with the original ``tsd`` array.
    For single-variable cases (ndim == 1), ``var_idx`` is ignored.
    """
    return tsd[var_idx, truncate_index:] if ndim == 2 else tsd[truncate_index:]


def _equilibration_stage(
    get_trajectory: Callable,
    get_trajectory_args: dict,
    number_of_variables: int,
    initial_run_length: int,
    run_length_factor: float,
    maximum_run_length: int,
    maximum_equilibration_step: int,
    batch_size: int,
    scale: str,
    with_centering: bool,
    with_scaling: bool,
    ignore_end: Union[int, float, None],
    population_mean_list: Sequence[Optional[float]],
    population_standard_deviation_list: Sequence[Optional[float]],
    population_cdf_list: Sequence[Optional[str]],
    population_args_list: Sequence[Optional[tuple]],
    population_loc_list: Sequence[Optional[float]],
    population_scale_list: Sequence[Optional[float]],
    si: str,
    nskip: Optional[int],
    fft: bool,
    minimum_correlation_time: Optional[int],
    number_of_cores: int,
    dump_trajectory: bool,
    dump_trajectory_fp,
) -> tuple[np.ndarray, int, int, list[int], bool]:
    r"""
    Detect the equilibration (warm-up) period for one or more time-series variables.

    This function incrementally acquires trajectory data and determines the number
    of initial steps that must be discarded to reach stationarity. It uses the
    Marginal Standard Error Rule (MSER) as the primary method, optionally applies
    a consistency check against known population parameters (3-sigma rule), and
    refines the estimate using integrated autocorrelation time (IAT) if equilibration
    is successfully detected.

    The process continues extending the simulation until either:
      - Equilibration is detected for all variables, or
      - The maximum_run_length is reached.

    Returns
    -------
    tsd : np.ndarray
        Full trajectory acquired during the process.
        Shape: (number_of_variables, total_steps) or (total_steps,) for single variable.
    run_length: int
        Length of the last trajectory segment acquired.
    total_run_length : int
        Total number of steps in the trajectory.
    equilibration_step : list[int]
        Per-variable index where the equilibrated region begins.
    equilibration_detected : bool
        True if equilibration was successfully detected for all variables.

    Raises
    ------
    CRError
        If any equilibration step exceeds maximum_equilibration_step or MSER returns
        an invalid truncation point.
    """
    # 1. Initialization and first trajectory acquisition

    # 1D or 2D trajectory
    ndim = 2 if number_of_variables > 1 else 1

    # initial running length
    run_length = min(initial_run_length, maximum_run_length)
    total_run_length = run_length

    # time series data temporary array
    tsd = _get_trajectory(
        get_trajectory,
        run_length=run_length,
        ndim=ndim,
        number_of_variables=number_of_variables,
        get_trajectory_args=get_trajectory_args,
    )  # shape: (number_of_variables, nstep) or (nstep,)

    # flag for extra population-based consistency check
    extra_check = any(
        x is not None
        for items in (population_mean_list, population_cdf_list)
        for x in items
    )

    truncated_flag = [False] * number_of_variables
    truncate_index = [0] * number_of_variables  # default 0 if never truncated

    method: str = "MSER"

    # 2. Incremental equilibration detection (primary: MSER + optional 3-sigma check)
    # to estimate the truncation point or "warm-up" period while we have sufficient data
    need_more_data = True
    while need_more_data:
        # apply MSER to each variable independently
        for i in range(number_of_variables):
            if truncated_flag[i]:
                continue

            series = tsd[i] if ndim == 2 else tsd
            truncated_flag[i], truncate_index[i] = mser_m(
                series,
                batch_size=batch_size,
                scale=scale,
                with_centering=with_centering,
                with_scaling=with_scaling,
                ignore_end=ignore_end,
            )

        truncated = all(truncated_flag)

        # if we reached the truncation point using marginal standard error rules
        if truncated and extra_check:
            # experimental feature to make sure of detecting the correct
            # equilibrium or warm-up period
            for i in range(number_of_variables):
                # slice a numpy array, the memory is shared
                series = _truncated_series(tsd, ndim, truncate_index[i], i)
                if not _is_within_3sigma(
                    series,
                    population_mean_list[i],
                    population_standard_deviation_list[i],
                    population_cdf_list[i],
                    population_args_list[i],
                    population_loc_list[i],
                    population_scale_list[i],
                ):
                    truncated = False
                    break

        if truncated:
            need_more_data = False
        else:
            # get the run length
            run_length = _get_run_length(
                run_length, run_length_factor, total_run_length, maximum_run_length
            )

            # we have reached the maximum limit
            if run_length == 0:
                need_more_data = False
            else:
                total_run_length += run_length

                # extend the time series data
                ext_tsd = _get_trajectory(
                    get_trajectory,
                    run_length=run_length,
                    ndim=ndim,
                    number_of_variables=number_of_variables,
                    get_trajectory_args=get_trajectory_args,
                )  # shape: (number_of_variables, nstep) or (nstep,)

                tsd = np.concatenate((tsd, ext_tsd), axis=ndim - 1)

    # 3. Finalize primary result
    equilibration_step: list[int] = list(truncate_index)
    equilibration_detected: bool = bool(truncated)

    # 4. Secondary refinement using Integrated Autocorrelation Time (if successful)
    if equilibration_detected:
        # refine with secondary method (integrated autocorrelation time)

        method = "MSER + Integrated Autocorrelation Time refinement"

        for i in range(number_of_variables):
            # slice a numpy array, the memory is shared
            # between the slice and the original
            series = (
                tsd[i, truncate_index[i]:] if ndim == 2 else tsd[truncate_index[i]:]
            )

            # check to get the more accurate estimate of the
            # equilibrium or warm-up period index
            equilibration_index_estimate, _ = estimate_equilibration_length(
                series,
                si=si,
                nskip=nskip,
                fft=fft,
                minimum_correlation_time=minimum_correlation_time,
                ignore_end=ignore_end,
                number_of_cores=number_of_cores,
            )

            # Correct the equilibration step
            equilibration_step[i] += equilibration_index_estimate

    # 5. Enforce hard limits and final checks
    _check_equilibration_step(
        equilibration_step,
        maximum_equilibration_step,
        maximum_run_length,
        equilibration_detected,
        tsd,
        dump_trajectory,
        dump_trajectory_fp,
        method=method,
    )

    # 6. Return results
    return tsd, run_length, total_run_length, equilibration_step, equilibration_detected

"""Run length control module."""
from inspect import isfunction
import json
import kim_edn
from math import isclose, fabs
import numpy as np
import sys
from typing import Optional, Union

from convergence._default import \
    _DEFAULT_CONFIDENCE_COEFFICIENT, \
    _DEFAULT_CONFIDENCE_INTERVAL_APPROXIMATION_METHOD, \
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS, \
    _DEFAULT_FFT, \
    _DEFAULT_TEST_SIZE, \
    _DEFAULT_TRAIN_SIZE, \
    _DEFAULT_BATCH_SIZE, \
    _DEFAULT_SCALE_METHOD, \
    _DEFAULT_WITH_CENTERING, \
    _DEFAULT_WITH_SCALING, \
    _DEFAULT_IGNORE_END, \
    _DEFAULT_NUMBER_OF_CORES, \
    _DEFAULT_SI, \
    _DEFAULT_NSKIP, \
    _DEFAULT_MINIMUM_CORRELATION_TIME, \
    _DEFAULT_MIN_ABSOLUTE_ACCURACY, \
    _DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL
from convergence import \
    check_population_cdf_args, \
    CVGError, \
    CVGSampleSizeError, \
    cvg_warning, \
    cvg_check, \
    chi_square_test, \
    levene_test, \
    t_test, \
    get_distribution_stats, \
    estimate_equilibration_length, \
    mser_m, \
    ucl_methods


__all__ = [
    'run_length_control',
]


def _convergence_message(
        number_of_variables: int,
        converged: bool,
        total_run_length: int,
        maximum_equilibration_step: int,
        equilibration_detected: bool,
        equilibration_step: Union[int, np.ndarray],
        confidence_coefficient: float,
        relative_accuracy: Union[float, np.ndarray, None],
        absolute_accuracy: Union[float, np.ndarray, None],
        upper_confidence_limit: Union[float, np.ndarray],
        upper_confidence_limit_method: str,
        relative_half_width_estimate: Union[float, np.ndarray],
        time_series_data_mean: Union[float, np.ndarray],
        time_series_data_std: Union[float, np.ndarray],
        effective_sample_size: Union[float, np.ndarray],
        minimum_number_of_independent_samples: int) -> dict:
    """Create convergence message.

    Args:
        number_of_variables (int): the number of variables in the corresponding
            time-series data.
        converged (bool): if we reached convergence or not.
        total_run_length (int): the total number of steps
        maximum_equilibration_step (int): the maximum number of steps as an
            equilibration hard limit.
        equilibration_detected (bool): if we reached equilibration or not!
        equilibration_step (int or 1darray): step number, where the
            equilibration has been achieved
        confidence_coefficient (float): Probability (or confidence interval)
            and must be between 0.0 and 1.0, and represents the confidence for
            calculation of relative halfwidths estimation.
        relative_accuracy (float, or 1darray): a relative half-width
            requirement or the accuracy parameter. Target value for the ratio
            of halfwidth to sample mean.
        absolute_accuracy (float, or 1darray): a half-width
            requirement or the accuracy parameter. Target value for the ratio
            of halfwidth to sample mean. If ``number_of_variables > 1``,
            ``relative_accuracy`` can be a scalar to be used for all variables
            or a 1darray of values of size number_of_variables.
        upper_confidence_limit (float, or 1darray): the upper confidence limit
            of the mean.
        upper_confidence_limit_method (str): Name of the UCL approach used to
            compute the upper_confidence_limit of the mean.
        relative_half_width_estimate(float, or 1darray): estimatemed relative
            half-width from the time-series data.
        time_series_data_mean (float, or 1darray): the mean of time-series data
            for each variable.
        time_series_data_std (float, or 1darray): the std of time-series data
            for each variable.
        effective_sample_size (float, or 1darray): the number of effective
            sample size.
        minimum_number_of_independent_samples (int): the minimum number of
            requested independent samples.

    Returns:
        dict: convergence message

    """
    confidence = '{}%'.format(round(confidence_coefficient * 100, 3))

    if number_of_variables == 1:
        relative_accuracy_ = relative_accuracy
        absolute_accuracy_ = absolute_accuracy

        if relative_accuracy_ is None:
            relative_accuracy_ = 'None'
            relative_half_width_estimate_ = 'None'
        else:
            absolute_accuracy_ = None
            relative_half_width_estimate_ = relative_half_width_estimate

        if absolute_accuracy_ is None:
            absolute_accuracy_ = 'None'

        if minimum_number_of_independent_samples is None:
            minimum_number_of_independent_samples_ = 'None'
        else:
            minimum_number_of_independent_samples_ = int(
                minimum_number_of_independent_samples)

        txt = '{} confidence interval for the mean is '.format(confidence)
        txt += '({}, '.format(time_series_data_mean - upper_confidence_limit)
        txt += '{})'.format(time_series_data_mean + upper_confidence_limit)

        msg = {
            'converged': converged,
            'total_run_length': total_run_length,
            'maximum_equilibration_step': maximum_equilibration_step,
            'equilibration_detected': equilibration_detected,
            'equilibration_step': equilibration_step,
            'confidence': confidence,
            'relative_accuracy': relative_accuracy_,
            'absolute_accuracy': absolute_accuracy_,
            'relative_half_width': relative_half_width_estimate_,
            'upper_confidence_limit': upper_confidence_limit,
            'upper_confidence_limit_method': upper_confidence_limit_method,
            'mean': time_series_data_mean,
            'standard_deviation': time_series_data_std,
            'effective_sample_size': effective_sample_size,
            'requested_sample_size': minimum_number_of_independent_samples_,
            'confidence_interval': txt,
        }
    else:
        msg = {
            'converged': converged,
            'total_run_length': total_run_length,
            'maximum_equilibration_step': maximum_equilibration_step,
            'equilibration_detected': equilibration_detected,
        }

        for i in range(number_of_variables):
            equilibration_detected_ = equilibration_detected or \
                equilibration_step[i] < maximum_equilibration_step

            relative_accuracy_ = relative_accuracy[i]
            absolute_accuracy_ = absolute_accuracy[i]

            if relative_accuracy_ is None:
                relative_accuracy_ = 'None'
                relative_half_width_estimate_ = 'None'
            else:
                absolute_accuracy_ = None
                relative_half_width_estimate_ = relative_half_width_estimate[i]

            if absolute_accuracy_ is None:
                absolute_accuracy_ = 'None'

            if minimum_number_of_independent_samples is None:
                minimum_number_of_independent_samples_ = 'None'
            else:
                minimum_number_of_independent_samples_ = \
                    minimum_number_of_independent_samples

            txt = '{} confidence interval for the mean is '.format(confidence)
            txt += '({}, '.format(time_series_data_mean[i] -
                                  upper_confidence_limit[i])
            txt += '{})'.format(time_series_data_mean[i] +
                                upper_confidence_limit[i])

            msg[i] = {
                'equilibration_detected': equilibration_detected_,
                'equilibration_step': equilibration_step[i],
                'confidence': confidence,
                'relative_accuracy': relative_accuracy_,
                'absolute_accuracy': absolute_accuracy_,
                'relative_half_width': relative_half_width_estimate_,
                'upper_confidence_limit': upper_confidence_limit[i],
                'upper_confidence_limit_method': upper_confidence_limit_method,
                'mean': time_series_data_mean[i],
                'standard_deviation': time_series_data_std[i],
                'effective_sample_size': effective_sample_size[i],
                'requested_sample_size': minimum_number_of_independent_samples_,
                'confidence_interval': txt,
            }
    return msg


def _check_get_trajectory(get_trajectory: callable) -> None:
    if not isfunction(get_trajectory):
        msg = 'the "get_trajectory" input is not a callback function.\n'
        msg += 'One has to provide the "get_trajectory" function as an '
        msg += 'input. It expects to have a specific signature:\n'
        msg += 'get_trajectory(nstep: int) -> 1darray,\n'
        msg += 'where nstep is the number of steps and the function '
        msg += 'should return a time-series data with the requested '
        msg += 'length equals to the number of steps.'
        raise CVGError(msg)


def _get_trajectory(get_trajectory: callable,
                    run_length: int,
                    ndim: int,
                    number_of_variables: int = 1,
                    get_trajectory_args: dict = {}) -> np.ndarray:
    if run_length == 0:
        return np.array([], dtype=np.float64)

    if type(get_trajectory_args) == dict and len(get_trajectory_args) > 0:
        try:
            tsd = get_trajectory(run_length, get_trajectory_args)
        except:
            msg = 'failed to get the time-series data or do the '
            msg += 'simulation for {} number of steps.'.format(run_length)
            raise CVGError(msg)
    else:
        try:
            tsd = get_trajectory(run_length)
        except:
            msg = 'failed to get the time-series data or do the '
            msg += 'simulationfor {} number of steps.'.format(run_length)
            raise CVGError(msg)

    tsd = np.array(tsd, dtype=np.float64, copy=False)

    # Extra check
    if not np.all(np.isfinite(tsd)):
        msg = 'there is/are value/s in the input which is/are non-finite or '
        msg += 'not number.'
        raise CVGError(msg)

    if np.ndim(tsd) != ndim:
        msg = 'the return from the "get_trajectory" function has a wrong '
        msg += 'dimension of {} != 1.'.format(tsd.ndim)
        raise CVGError(msg)

    if ndim == 2 and number_of_variables != np.shape(tsd)[0]:
        msg = 'the return of "get_trajectory" function has a wrong number '
        msg += 'of variables = {} != '.format(np.shape(tsd)[0])
        msg += '{}.\n'.format(number_of_variables)
        msg += 'In a two-dimensional return array of "get_trajectory" '
        msg += 'function, each row corresponds to the time series data '
        msg += 'for one variable.'
        raise CVGError(msg)

    return tsd


def _get_run_length(run_length: int,
                    run_length_factor: float,
                    total_run_length: int,
                    maximum_run_length: int) -> int:
    if 0 in (run_length, total_run_length):
        return run_length

    run_length = min(int(run_length * run_length_factor), maximum_run_length)
    run_length = max(1, run_length)

    total_run_length += run_length

    if total_run_length >= maximum_run_length:
        total_run_length -= run_length
        run_length = maximum_run_length - total_run_length

    return run_length


def _check_accuracy(
        number_of_variables: int,
        relative_accuracy: Union[float, list[float], np.ndarray, None],
        absolute_accuracy: Union[float, list[float], np.ndarray, None]) -> None:
    if number_of_variables == 1:
        if np.size(relative_accuracy) != 1:
            msg = 'For one variable, "relative_accuracy" must be a scalar '
            msg += '(a `float`).'
            raise CVGError(msg)

        if np.size(absolute_accuracy) != 1:
            msg = 'For one variable, "absolute_accuracy" must be a scalar '
            msg += '(a `float`).'
            raise CVGError(msg)

        if relative_accuracy is None:
            cvg_check(absolute_accuracy, 'absolute_accuracy',
                      var_lower_bound=_DEFAULT_MIN_ABSOLUTE_ACCURACY)

    else:
        if np.size(relative_accuracy) != number_of_variables:
            msg = '"relative_accuracy" must be a list of '
            msg += '{} floats.'.format(number_of_variables)
            raise CVGError(msg)

        if np.size(absolute_accuracy) != number_of_variables:
            msg = '"absolute_accuracy" must be a list of '
            msg += '{} floats.'.format(number_of_variables)
            raise CVGError(msg)

        for index, rel_acc in enumerate(relative_accuracy):
            if rel_acc is None:
                cvg_check(absolute_accuracy[index],
                          'absolute_accuracy[{}]'.format(index),
                          var_lower_bound=_DEFAULT_MIN_ABSOLUTE_ACCURACY)


def _check_equilibration_step(equilibration_step: Union[int, list[int]],
                              maximum_equilibration_step: int,
                              maximum_run_length: int,
                              equilibration_detected: bool,
                              time_series_data: Union[list[float], np.ndarray],
                              dump_trajectory: bool,
                              dump_trajectory_fp) -> None:
    equilibration_step_array = np.array(equilibration_step, copy=False)

    hard_limit_crossed = np.any(
        equilibration_step_array >= maximum_equilibration_step)

    if not hard_limit_crossed:
        return

    if dump_trajectory:
        kim_edn.dump(time_series_data.tolist(), dump_trajectory_fp)

    number_of_variables = np.size(equilibration_step)
    if number_of_variables == 1:
        if equilibration_detected:
            msg = 'The equilibration or "warm-up" period is detected '
            msg += 'at step = {}, which '.format(equilibration_step)
            msg += 'is greater than the maximum number of allowed steps '
            msg += 'for the equilibration detection = '
            msg += '{}.\n'.format(maximum_equilibration_step)
        else:
            if equilibration_step == (maximum_run_length - 1):
                msg = 'The equilibration or "warm-up" period is not detected. '
                msg += 'Check the trajectory data!\n'
            else:
                msg = 'The truncation point = {}, '.format(equilibration_step)
                msg += 'returned by MSER is > half of the data set size and '
                msg += 'is invalid.\n'
    else:
        for i in range(number_of_variables):
            if equilibration_step[i] < maximum_equilibration_step:
                msg = 'The equilibration or "warm-up" period for variable '
                msg += 'number {} is detected at step = '.format(i + 1)
                msg += '{}.\n'.format(equilibration_step[i])
            else:
                if equilibration_step[i] == (maximum_run_length - 1):
                    msg = 'The equilibration or "warm-up" period for variable '
                    msg += 'number {}, is not detected. '.format(i + 1)
                    msg = 'Check the trajectory data!\n'
                else:
                    msg = 'The truncation point for variable number '
                    msg += '{} = {}, '.format(i + 1, equilibration_step[i])
                    msg += 'returned by MSER is > half of the data set '
                    msg += 'size and is invalid.\n'

    if equilibration_detected:
        msg += 'To prevent this error, you can either request a '
        msg += 'longer maximum number of allowed steps to reach '
        msg += 'equilibrium or if you did not provide this limit you '
        msg += 'can increase the maximum_run_length.\n'
    else:
        msg += 'More data is required. To prevent this error, you can '
        msg += 'request a longer maximum_run_length.\n'

    raise CVGError(msg)


def _check_population(
        number_of_variables: int,
        population_mean: Union[float, list[float], np.ndarray, None],
        population_standard_deviation: Union[float, list[float], np.ndarray, None],
        population_cdf: Union[str, list[str], None],
        population_args: Union[tuple, list[tuple], None],
        population_loc: Union[float, list[float], np.ndarray, None],
        population_scale: Union[float, list[float], np.ndarray, None]) -> None:

    # Initialize
    if number_of_variables == 1:
        if population_mean is not None and np.size(population_mean) != 1:
            msg = 'For one variable, "population_mean" must be a scalar '
            msg += '(a `float`).'
            raise CVGError(msg)

        if population_standard_deviation is not None and \
                np.size(population_standard_deviation) != 1:
            msg = 'For one variable, "population_standard_deviation" must '
            msg += 'be a scalar (a `float`).'
            raise CVGError(msg)

        if population_cdf is not None:
            if np.size(population_cdf) != 1:
                msg = 'For one variable, "population_cdf" must be a name '
                msg += '(a `str`).'
                raise CVGError(msg)

            check_population_cdf_args(population_cdf=population_cdf,
                                      population_args=population_args)

            if population_mean is not None:
                msg = 'For the non-normally distributed data '
                msg += '"population_mean" should not be provided.\n'
                msg += 'To shift and/or scale the distribution use the '
                msg += '"population_loc" and "population_scale" parameters.'
                raise CVGError(msg)

            if population_standard_deviation is not None:
                msg = 'For the non-normally distributed data '
                msg += '"population_standard_deviation" should not '
                msg += 'be provided.\n'
                msg += 'To shift and/or scale the distribution use the '
                msg += '"population_loc" and "population_scale" parameters.'
                raise CVGError(msg)

            if population_loc is not None and np.size(population_loc) != 1:
                msg = 'For one variable, "population_loc" must be a scalar '
                msg += '(a `float`).'
                raise CVGError(msg)

            if population_scale is not None and np.size(population_scale) != 1:
                msg = 'For one variable, "population_scale" must be a scalar '
                msg += '(a `float`).'
                raise CVGError(msg)

    else:
        if population_mean is not None:
            if np.size(population_mean) != number_of_variables:
                msg = '"population_mean" must be a 1darray/list of size = '
                msg += '{}.'.format(number_of_variables)
                raise CVGError(msg)

        if population_standard_deviation is not None:
            if np.size(population_standard_deviation) != number_of_variables:
                msg = '"population_standard_deviation" must be a 1darray/list '
                msg += 'of size = {}.'.format(number_of_variables)
                raise CVGError(msg)

        if population_cdf is not None:
            if np.size(population_cdf) != number_of_variables:
                msg = '"population_cdf" must be a 1darray/list of size = '
                msg += '{}.'.format(number_of_variables)
                raise CVGError(msg)

            if population_args is not None and \
                    len(population_args) != number_of_variables:
                msg = '"population_args" must be a list of size = '
                msg += '{}.'.format(number_of_variables)
                raise CVGError(msg)

            if population_args is None:
                msg = '"population_args" must be a list of size = '
                msg += '{}.'.format(number_of_variables)
                raise CVGError(msg)

            if population_loc is not None and \
                    np.size(population_loc) != number_of_variables:
                msg = '"population_loc" must be a 1darray/list '
                msg += 'of size = {}.'.format(number_of_variables)
                raise CVGError(msg)

            if population_scale is not None and \
                    np.size(population_scale) != number_of_variables:
                msg = '"population_scale" must be a 1darray/list '
                msg += 'of size = {}.'.format(number_of_variables)
                raise CVGError(msg)

            index = -1

            for pop_cdf, pop_args in zip(population_cdf, population_args):
                index += 1

                check_population_cdf_args(population_cdf=pop_cdf,
                                          population_args=pop_args)

                if pop_cdf is not None:
                    if population_mean is not None and \
                            population_mean[index] is not None:
                        msg = 'For the non-normally distributed data at '
                        msg += 'index = {}, '.format(index + 1)
                        msg += '"population_mean" should not be provided.'
                        msg += '\nTo shift and/or scale the distribution '
                        msg += 'use the "population_loc" and '
                        msg += '"population_scale" parameters.'
                        raise CVGError(msg)

                    if population_standard_deviation is not None and \
                            population_standard_deviation[index] is not None:
                        msg = 'For the non-normally distributed data at '
                        msg += 'index = {}, '.format(index + 1)
                        msg += '"population_standard_deviation" should '
                        msg += 'not be provided.\nTo shift and/or scale '
                        msg += 'the distribution use the "population_loc" '
                        msg += 'and "population_scale" parameters.'
                        raise CVGError(msg)


def _get_array_tolist(input_array: np.ndarray) -> list:
    if input_array is not None:
        if isinstance(input_array, np.ndarray):
            input_array = input_array.tolist()

        if type(input_array) in (list, tuple):
            for index, value in enumerate(input_array):
                if value is not None and not np.isfinite(value):
                    input_array[index] = None
        elif not np.isfinite(input_array):
            input_array = None
    return input_array


def run_length_control(
    get_trajectory: callable,
    get_trajectory_args: Optional(dict) = None,
    *,
    number_of_variables: int = 1,
    initial_run_length: int = 10000,
    run_length_factor: float = 1.0,
    maximum_run_length: int = 1000000,
    maximum_equilibration_step: Optional(int) = None,
    minimum_number_of_independent_samples: Optional(int) = None,
    relative_accuracy: Union[float, list[float], np.ndarray, None] = 0.1,
    absolute_accuracy: Union[float, list[float], np.ndarray, None] = 0.1,
    population_mean: Union[float, list[float], np.ndarray, None] = None,
    population_standard_deviation: Union[float,
                                         list[float], np.ndarray, None] = None,
    population_cdf: Union[str, list[str], None] = None,
    population_args: Union[tuple, list[tuple], None] = None,
    population_loc: Union[float, list[float], np.ndarray, None] = None,
    population_scale: Union[float, list[float], np.ndarray, None] = None,
    # arguments used by different components
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    confidence_interval_approximation_method: str = _DEFAULT_CONFIDENCE_INTERVAL_APPROXIMATION_METHOD,
    heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    fft: bool = _DEFAULT_FFT,
    test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
    train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    scale: str = _DEFAULT_SCALE_METHOD,
    with_centering: bool = _DEFAULT_WITH_CENTERING,
    with_scaling: bool = _DEFAULT_WITH_SCALING,
    ignore_end: Union[int, float, None] = _DEFAULT_IGNORE_END,
    number_of_cores: int = _DEFAULT_NUMBER_OF_CORES,
    si: str = _DEFAULT_SI,
    nskip: int = _DEFAULT_NSKIP,
    minimum_correlation_time: int = _DEFAULT_MINIMUM_CORRELATION_TIME,
    dump_trajectory: bool = False,
    dump_trajectory_fp: str = 'convergence_trajectory.edn',
    fp: Optional(str) = None,
    fp_format: str = 'txt'
) -> str:
    r"""Control the length of the time series data from a simulation run.

    It starts drawing ``initial_run_length`` number of observations (samples)
    by calling the ``get_trajectory`` function in a loop to reach equilibration
    or pass the ``warm-up`` period.

    Note:
        ``get_trajectory`` is a callback function with a specific signature of
        ``get_trajectory(nstep: int) -> 1darray`` if we only have one variable
        or
        ``get_trajectory(nstep: int) -> 2darray`` with the shape of
        (number_of_variables, nstep)

        To use extra arguments in the ``get_trajectory``, one can use the other
        specific signature of
        ``get_trajectory(nstep: int, args: dict) -> 1darray``
        or
        ``get_trajectory(nstep: int, args: dict) -> 2darray`` with the shape of
        (number_of_variables, nstep)

        where all the required variables can be pass thrugh the args dictionary.

        All the values returned from this function should be finite values,
        otherwise the code will stop wih error message explaining the issue.

        Examples:

        >>> rng = np.random.RandomState(12345)
        >>> start = 0
        >>> stop = 0
        >>> def get_trajectory(step):
                global start, stop
                start = stop
                if 100000 < start + step:
                    step = 100000 - start
                stop += step
                data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
                return data

        or,

        >>> targs = {'start': 0, 'stop': 0}
        >>> def get_trajectory(step, targs):
                targs['start'] = targs['stop']
                if 100000 < targs['start'] + step:
                    step = 100000 - targs['start']
                targs['stop'] += step
                data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
                return data

    Then it continues drawing observations until some pre-specified level of
    absolute or relative precision has been reached.

    The ``precision`` is defined as a half-width of the confidence interval
    (CI) of the estimator.

    At each checkpoint, an upper confidence limit (``UCL``) is approximated.
    The drawing of observations is terminated, if UCL is less than the
    pre-specified absolute precision ``absolute_accuracy`` or if the relative
    UCL (UCL divided by the computed sample mean) is less than a pre-specified
    value, ``relative_accuracy``.

    The UCL is calculated as a `confidence_coefficient%` confidence interval
    for the mean, using the portion of the time series data, which is in the
    stationarity region.

    The ``Relative accuracy`` is the confidence interval half-width or UCL
    divided by the sample mean. If the ratio is bigger than
    `relative_accuracy`, the length of the time series is deemed not long
    enough to estimate the mean with sufficient accuracy, which means the run
    should be extended.

    In order to avoid problems caused by sequential UCL evaluation cost, this
    calculation should not be repeated too frequently. Heidelberger and Welch
    (1981) [2]_ suggested increasing the run length by a factor of
    `run_length_factor > 1.5`, each time, so that estimate has the same,
    reasonably large proportion of new data.

    The accuracy parameter `relative_accuracy` specifies the maximum relative
    error that will be allowed in the mean value of time-series data. In other
    words, the distance from the confidence limit(s) to the mean (which is also
    known as the precision, half-width, or margin of error). A value of `0.01`
    is usually used to request two digits of accuracy, and so forth.

    The parameter ``confidence_coefficient`` is the confidence coefficient and
    often, the values 0.95 is used.
    For the confidence coefficient, `confidence_coefficient`, we can use the
    following interpretation,

    If thousands of samples of n items are drawn from a population using
    simple random sampling and a confidence interval is calculated for each
    sample, the proportion of those intervals that will include the true
    population mean is `confidence_coefficient`.

    The ``maximum_run_length`` parameter places an upper bound on how long the
    simulation will run. If the specified accuracy cannot be achieved within
    this time, the simulation will terminate, and a warning message will
    appear in the report.

    The ``maximum_equilibration_step`` parameter places an upper bound on how
    long the simulation will run to reach equilibration or pass the ``warm-up``
    period. If the equilibration or warm-up period cannot be detected within
    this time, the simulation will terminate and a warning message will appear
    in the report.

    Note:
        By default and if not specified on input, the
        ``maximum_equilibration_step`` is defined as half of the
        ``maximum_run_length``.

    Notes:
        By default, the algorithm will use ``relative_accuracy`` as a
        termination criterion, and in case of failure, it switches to use the
        ``absolute_accuracy``.

        If using the ``absolute_accuracy`` is desired, one should set the
        ``relative_accuracy`` to None.

        Examples:

        >>> run_length_control(get_trajectory,
                               number_of_variables=1,
                               relative_accuracy=None
                               absolute_accuracy=0.1)

        The algorithm converts ``relative_accuracy``and ``absolute_accuracy``
        floating numbers to arrays with the shape of (number_of_variables, ),
        when the ``number_of_variables`` bigger than one. By default, it uses
        ``relative_accuracy`` as a termination criterion for the corresponding
        variable number, and in case of failure, it switches to use the
        ``absolute_accuracy``.

        If the ``absolute_accuracy`` is desired for one or some variables, one
        should provide both ``relative_accuracy``and ``absolute_accuracy`` as
        an array. Then it must set the corresponding ``relative_accuracy`` in
        the array to None and set the correct `absolute_accuracy`` at the right
        place in the collection.

        E.g.,
        >>> run_length_control(get_trajectory,
                               number_of_variables=3,
                               relative_accuracy=[0.1, 0.05, None]
                               absolute_accuracy=[0.1, 0.05, 0.1])

        or,
        >>> run_length_control(get_trajectory,
                               number_of_variables=3,
                               relative_accuracy=[None, 0.05, None]
                               absolute_accuracy=[0.1,  0.05, 0.1])

    Note:
        confidence_interval_approximation_method is set to a method to use for
        approximating the upper confidence limit of the mean.

        By default, (``uncorrelated_sample`` approach) uses the independent samples in
        the time-series data to approximate the confidence intervals for the
        mean. The other methods have different approaches.

        E.g., in the ``heidel_welch`` method, it requires no such
        independence assumption. In this spectral approach, the problem of
        dealing with dependent data are largely avoided by working in the
        frequency domain with the sample spectrum (periodogram) of the process.

    Note:
        ``population_mean`` is a variable known (true) mean. Expected value in
        null hypothesis. It is an extra information for normally distributed
        data.

    Note:
        for non-normally distributed data, and as an extra check on the
        convergence one should provide the population info using
        ``population_cdf``, ``population_args``, ``population_loc``, and
        ``population_scale`` for a specific distribution.

    Args:
        get_trajectory (callback function): A callback function with a
            specific signature of ``get_trajectory(nstep: int) -> 1darray``
            if we only have one variable or
            ``get_trajectory(nstep: int) -> 2darray`` with the shape of
            (number_of_variables, nstep)
            Note:
                all the values returned from this function should be finite
                values, otherwise the code will stop wih error message
                explaining the issue.
        get_trajectory_args (dict, optional): Extra arguments passed to the
            get_trajectory function. (default: {})
            To use this option, the dictionary may contain `start` and `stop`
            keywords as well as other keywords which are needed in the
            function.
            ``get_trajectory(nstep, get_trajectory_args) -> 1darray``

        number_of_variables (int, optional): number of variables in the
            corresponding time-series data from get_trajectory callback
            function. (default: 1)
        initial_run_length (int, optional): initial run length.
            (default: 2000)
        run_length_factor (float, optional): run length increasing factor.
            (default: 1.0)
        maximum_run_length (int, optional): the maximum run length represents
            a cost constraint. (default: 1000000)
        maximum_equilibration_step (int, optional): the maximum number of
            steps as an equilibration hard limit. If the algorithm finds
            equilibration_step greater than this limit it will fail. For the
            default None, the function is using ``maximum_run_length // 2`` as
            the maximum equilibration step. (default: None)
        minimum_number_of_independent_samples (int, optional): minimum number
            of independent samples. This is an extra parameter to terminate the
            run after the pre-specified level of absolute or relative precision
            has been reached and there are minimum number of independent
            samples available for further analysis. (default: None)
        relative_accuracy (float, or 1darray, optional): a relative half-width
            requirement or the accuracy parameter. Target value for the ratio
            of halfwidth to sample mean. If ``number_of_variables > 1``,
            ``relative_accuracy`` can be a scalar to be used for all variables
            or a 1darray of values of size number_of_variables. (default: 0.1)
        absolute_accuracy (float, or 1darray, optional): a half-width
            requirement or the accuracy parameter. Target value for the ratio
            of halfwidth to sample mean. If ``number_of_variables > 1``,
            ``relative_accuracy`` can be a scalar to be used for all variables
            or a 1darray of values of size number_of_variables. (default: 0.1)

        population_mean (float, or 1darray, optional): variable known (true)
            mean. Expected value in null hypothesis. (default: None)

            Note:
                For ``number_of_variables > 1``, and if ``population_mean``
                is provided, it should be a list or array of values. It should
                be set to None for variables which we do not intend to use this
                extra measure.

                Examples:

                >>> run_length_control(get_trajectory,
                                       number_of_variables=3,
                                       population_mean=[None, 297., None])

        population_standard_deviation (float, or 1darray, optional): population
            standard deviation. (default: None)

            Note:
                For ``number_of_variables > 1``, and if
                ``population_standard_deviation`` is provided, it should be a
                list or array of values. It should be set to None for variables
                which we do not intend to use this extra measure.

                Examples:

                >>> run_length_control(
                        get_trajectory,
                        number_of_variables=3,
                        population_mean=[None, 297., None],
                        population_standard_deviation=[None, 10., None])

        population_cdf (str, or 1darray, optional): The name of a distribution.
            (default: None)

            Examples:
            >>> run_length_control(
                    get_trajectory,
                    number_of_variables=2,
                    population_cdf=[None, 'gamma'],
                    population_args=[None, (1.99,)],
                    population_loc=[None, None],
                    population_scale=[None, None])

            or,

            >>> run_length_control(
                    get_trajectory,
                    number_of_variables=2,
                    population_mean=[297., None],
                    population_standard_deviation=[10., None],
                    population_cdf=[None, 'gamma'],
                    population_args=[None, (1.99,)],
                    population_loc=[None, None],
                    population_scale=[None, None])

        population_args (tuple, or list of tuples, optional): Distribution
            parameter. (default: None)
        population_loc (float, or 1darray, or None): location of the
            distribution. (default: None)
        population_scale (float, or 1darray, or None): scale of the
            distribution. (default: None)

        confidence_coefficient (float, optional): (or confidence level) and
            must be between 0.0 and 1.0, and represents the confidence for
            calculation of relative halfwidths estimation. (default: 0.95)
        confidence_interval_approximation_method (str, optional) : Method to
            use for approximating the upper confidence limit of the mean.
            One of the ``ucl_methods`` aproaches. (default: 'uncorrelated_sample')
        heidel_welch_number_points (int, optional): the number of points in
            Heidelberger and Welch's spectral method that are used to obtain
            the polynomial fit. The parameter ``heidel_welch_number_points``
            determines the frequency range over which the fit is made.
            (default: 50)
        fft (bool, optional): if True, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        test_size (int, float, optional): if ``float``, should be between 0.0
            and 1.0 and represent the proportion of the periodogram dataset to
            include in the test split. If ``int``, represents the absolute
            number of test samples. (default: None)
        train_size (int, float, optional): if ``float``, should be between
            0.0  and 1.0 and represent the proportion of the preiodogram
            dataset to include in the train split. If `int`, represents the
            absolute number of train samples. (default: None)
        batch_size (int, optional): batch size. (default: 5)
        scale (str, optional): a method to standardize a batched dataset.
            (default: 'translate_scale')
        with_centering (bool, optional): if ``True``, use batched data minus
            the scale metod centering approach. (default: False)
        with_scaling (bool, optional): if ``True``, scale the batched data to
            scale metod scaling approach. (default: False)
        ignore_end (int, or float, or None, optional): if ``int``, it is the
            last few (batch) points that should be ignored. if ``float``,
            should be in ``(0, 1)`` and it is the percent of last (batch)
            points that should be ignored. if `None` it would be set to the
            ``batch_size`` in bacth method and to the one fourth of the total
            number of points elsewhere. (default: None)
        number_of_cores (int, optional): The maximum number of concurrently
            running jobs, such as the number of Python worker processes or the
            size of the thread-pool. If -1 all CPUs are used. If 1 is given, no
            parallel computing code is used at all. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
            one are used. (default: 1)
        si (str, optional): statistical inefficiency method.
            (default: 'statistical_inefficiency')
        nskip (int, optional): the number of data points to skip in estimating
            ucl. (default: 1)
        minimum_correlation_time (int, optional): The minimum amount of
            correlation function to compute in estimating ucl. The algorithm
            terminates after computing the correlation time out to
            minimum_correlation_time when the correlation function first
            goes negative. (default: None)
        dump_trajectory (bool, optional): if ``True``, dump the final trajectory
            data to a file ``dump_trajectory_fp``. (default: False)
        dump_trajectory_fp (str, object with a write(string) method, optional):
            a ``.write()``-supporting file-like object or a name string to open
            a file. (default: 'convergence_trajectory.edn')
        fp (str, object with a write(string) method, optional): if an ``str``
            equals to ``'return'`` the function will return string of the
            analysis results on the length of the time series. Otherwise it
            must be an object with write(string) method. If it is None,
            ``sys.stdout`` will be used which prints objects on the screen.
            (default: None)
        fp_format (str): one of the ``txt``, ``json``, or ``edn`` format.
            (default: 'txt')

    Returns:
        bool or str:
            ``True`` if the length of the time series is long enough to
            estimate the mean with sufficient accuracy or with enough requested
            sample size and ``False`` otherwise.
            If fp is a ``str`` equals to ``'return'`` the function will return
            a string of the analysis results on the length of the time series.

    """
    _check_get_trajectory(get_trajectory)
    cvg_check(number_of_variables, 'number_of_variables', int, 1)
    cvg_check(initial_run_length, 'initial_run_length', int, 1)
    cvg_check(run_length_factor, 'run_length_factor', None, 0)
    cvg_check(maximum_run_length, 'maximum_run_length', int, 1)

    if maximum_equilibration_step is None:
        maximum_equilibration_step = maximum_run_length // 2
        msg = '"maximum_equilibration_step" is not given on input!\nThe '
        msg += 'maximum number of steps as an equilibration hard limit '
        msg += 'is set to {}.'.format(maximum_equilibration_step)
        cvg_warning(msg)

    # Set the hard limit for the equilibration step
    cvg_check(maximum_equilibration_step, 'maximum_equilibration_step', int, 1)
    if maximum_equilibration_step >= maximum_run_length:
        msg = '"maximum_equilibration_step" = '
        msg += '{} must be '.format(maximum_equilibration_step)
        msg += 'less than maximum_run_length = {}.'.format(maximum_run_length)
        raise CVGError(msg)

    if minimum_number_of_independent_samples is not None:
        cvg_check(minimum_number_of_independent_samples,
                  'minimum_number_of_independent_samples', int, 1)

    if fp is None:
        fp = sys.stdout
    elif isinstance(fp, str):
        if fp != 'return':
            msg = 'Keyword argument `fp` is a `str` and not equal to "return".'
            raise CVGError(msg)
        fp = None
    elif not hasattr(fp, 'write'):
        msg = 'Keyword argument `fp` must be either a `str` and equal to '
        msg += '"return", or None, or an object with write(string) method.'
        raise CVGError(msg)

    if fp_format not in ('txt', 'json', 'edn'):
        msg = 'fp format is unknown. Valid formats are:\n\t- '
        msg += '{}'.format('\n\t- '.join(('txt', 'json', 'edn')))
        raise CVGError(msg)

    # Initialize
    if number_of_variables == 1:
        ndim = 1
    else:
        ndim = 2
        if np.size(relative_accuracy) == 1:
            relative_accuracy = [relative_accuracy] * number_of_variables
        relative_accuracy = _get_array_tolist(relative_accuracy)
        if np.size(absolute_accuracy) == 1:
            absolute_accuracy = [absolute_accuracy] * number_of_variables
        absolute_accuracy = _get_array_tolist(absolute_accuracy)
        population_mean = _get_array_tolist(population_mean)
        population_standard_deviation = \
            _get_array_tolist(population_standard_deviation)
        population_cdf = _get_array_tolist(population_cdf)
        population_args = _get_array_tolist(population_args)
        if population_args is None:
            population_args = [None] * number_of_variables
        population_loc = _get_array_tolist(population_loc)
        population_scale = _get_array_tolist(population_scale)

    _check_accuracy(number_of_variables, relative_accuracy, absolute_accuracy)
    _check_population(number_of_variables,
                      population_mean,
                      population_standard_deviation,
                      population_cdf,
                      population_args,
                      population_loc,
                      population_scale)

    if confidence_interval_approximation_method not in ucl_methods:
        msg = 'method "{}" '.format(confidence_interval_approximation_method)
        msg += 'to aproximate confidence interval not found. Valid '
        msg += 'methods are:\n\t- '
        msg += '{}'.format('\n\t- '.join(ucl_methods))
        raise CVGError(msg)

    try:
        ucl_obj = ucl_methods[confidence_interval_approximation_method]()
    except CVGError:
        msg = "Failed to initialize the UCL object."
        raise CVGError(msg)

    if ucl_obj.name == 'heidel_welch':
        ucl_obj.set_heidel_welch_constants(
            confidence_coefficient=confidence_coefficient,
            heidel_welch_number_points=heidel_welch_number_points)

    # Initial running length
    run_length = min(initial_run_length, maximum_run_length)

    equilibration_step = 0
    total_run_length = run_length

    # Extra check flag
    extra_check = population_mean is not None or population_cdf is not None

    need_more_data = True

    # one variable
    if ndim == 1:
        # Time series data temporary array
        tsd = _get_trajectory(get_trajectory,
                              run_length=run_length,
                              ndim=ndim,
                              number_of_variables=number_of_variables,
                              get_trajectory_args=get_trajectory_args)

        # Estimate the truncation point or "warm-up" period
        # while we have sufficient data
        while need_more_data:
            truncated, truncate_index = mser_m(tsd,
                                               batch_size=batch_size,
                                               scale=scale,
                                               with_centering=with_centering,
                                               with_scaling=with_scaling,
                                               ignore_end=ignore_end)

            # if we reached the truncation point using
            # marginal standard error rules
            if truncated and extra_check:
                # Experimental feature to make sure of detecting the
                # correct equilibrium or warm-up period

                # slice a numpy array, the memory is shared
                # between the slice and the original
                time_series_data = tsd[truncate_index:]

                if population_mean is not None:
                    sample_mean = time_series_data.mean()
                    diff = abs(population_mean - sample_mean)

                    sample_std = time_series_data.std()
                    if population_standard_deviation is not None:
                        sample_std = \
                            max(sample_std, population_standard_deviation)

                if population_cdf is not None:
                    population_median, _, _, population_std = \
                        get_distribution_stats(population_cdf,
                                               population_args,
                                               population_loc,
                                               population_scale)

                    sample_median = time_series_data.median()
                    if np.isfinite(sample_median):
                        diff = abs(population_median - sample_median)
                    else:
                        diff = 0

                    sample_std = time_series_data.std()
                    if np.isfinite(population_std):
                        sample_std = max(sample_std, population_std)

                # Estimates further than 3 standard errors away can then
                # easily be flagged as not truncated
                if diff > 3 * sample_std:
                    truncated = False

            # Found the truncation point or "warm-up" period
            if truncated:
                need_more_data = False

            if need_more_data:
                # get the run length
                run_length = _get_run_length(run_length,
                                             run_length_factor,
                                             total_run_length,
                                             maximum_run_length)

                # We have reached the maximum limit
                if run_length == 0:
                    need_more_data = False
                else:
                    total_run_length += run_length

                    _tsd = _get_trajectory(
                        get_trajectory,
                        run_length=run_length,
                        ndim=ndim,
                        number_of_variables=number_of_variables,
                        get_trajectory_args=get_trajectory_args)
                    tsd = np.concatenate((tsd, _tsd))

        del extra_check

        equilibration_step = truncate_index
        equilibration_detected = truncated

        if truncated:
            # slice a numpy array, the memory is shared
            # between the slice and the original
            time_series_data = tsd[truncate_index:]

            # Check to get the more accurate estimate of the
            # equilibrium or warm-up period index
            equilibration_index_estimate, _ = \
                estimate_equilibration_length(
                    time_series_data,
                    si=si,
                    nskip=nskip,
                    fft=fft,
                    minimum_correlation_time=minimum_correlation_time,
                    ignore_end=ignore_end,
                    number_of_cores=number_of_cores)

            # Correct the equilibration step
            equilibration_step += equilibration_index_estimate

        # Check the hard limit
        _check_equilibration_step(equilibration_step,
                                  maximum_equilibration_step,
                                  maximum_run_length,
                                  equilibration_detected,
                                  tsd,
                                  dump_trajectory,
                                  dump_trajectory_fp)

        relative_half_width_estimate = None

        enough_accuracy = False
        need_more_data = True

        while need_more_data:
            # slice a numpy array, the memory is shared
            # between the slice and the original
            time_series_data = tsd[equilibration_step:]
            time_series_data_size = time_series_data.size

            enough_data = True

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
                    minimum_correlation_time=minimum_correlation_time)
            except CVGSampleSizeError:
                # do not have enough data and need more
                enough_data = False
                upper_confidence_limit = None
            except CVGError:
                raise CVGError("Failed to compute the UCL.")

            if enough_data:
                if relative_accuracy is None:
                    enough_accuracy = \
                        upper_confidence_limit < absolute_accuracy
                else:
                    # Estimate the relative half width
                    if isclose(
                            ucl_obj.mean, 0,
                            abs_tol=_DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL
                    ):
                        msg = 'It is not possible to estimate the relative '
                        msg += 'half width for the close to zero mean = '
                        msg += '{}'.format(ucl_obj.mean)
                        raise CVGError(msg)

                    relative_half_width_estimate = upper_confidence_limit / \
                        fabs(ucl_obj.mean)

                    enough_accuracy = \
                        relative_half_width_estimate < relative_accuracy

                if enough_accuracy:
                    if ucl_obj.name != 'uncorrelated_sample':
                        ucl_obj.set_indices(
                            time_series_data,
                            si=si,
                            fft=fft,
                            minimum_correlation_time=minimum_correlation_time)

                    effective_sample_size = time_series_data_size / ucl_obj.si

                    if minimum_number_of_independent_samples is None or \
                            effective_sample_size >= \
                        minimum_number_of_independent_samples:

                        need_more_data = False

                        if population_mean is not None:
                            need_more_data = not t_test(
                                sample_mean=ucl_obj.mean,
                                sample_std=ucl_obj.std,
                                sample_size=ucl_obj.sample_size,
                                population_mean=population_mean,
                                significance_level=1-confidence_coefficient)

                        if not need_more_data and \
                                population_standard_deviation is not None:
                            need_more_data = not chi_square_test(
                                sample_var=ucl_obj.std * ucl_obj.std,
                                sample_size=ucl_obj.sample_size,
                                population_var=population_standard_deviation *
                                population_standard_deviation,
                                significance_level=1-confidence_coefficient)

                        if not need_more_data and \
                                population_cdf is not None:
                            need_more_data = not levene_test(
                                time_series_data=time_series_data[
                                    ucl_obj.indices],
                                population_cdf=population_cdf,
                                population_args=population_args,
                                population_loc=population_loc,
                                population_scale=population_scale,
                                significance_level=1-confidence_coefficient)

            if need_more_data:
                # get the run length
                run_length = _get_run_length(run_length,
                                             run_length_factor,
                                             total_run_length,
                                             maximum_run_length)

                # We have reached the maximum limit
                if run_length == 0:
                    enough_accuracy = False
                    need_more_data = False
                else:
                    total_run_length += run_length

                    _tsd = _get_trajectory(
                        get_trajectory,
                        run_length=run_length,
                        ndim=ndim,
                        number_of_variables=number_of_variables,
                        get_trajectory_args=get_trajectory_args)
                    tsd = np.concatenate((tsd, _tsd))

        if upper_confidence_limit is None:
            raise CVGError("Failed to compute the UCL.")

        if dump_trajectory:
            kim_edn.dump(tsd.tolist(), dump_trajectory_fp)

        converged = enough_accuracy and not need_more_data
        # convert np.bool_ to python bool
        converged = bool(converged)

        if not converged:
            time_series_data = tsd[equilibration_step:]
            time_series_data_size = time_series_data.size

            if ucl_obj.name != 'uncorrelated_sample':
                ucl_obj.set_indices(
                    time_series_data,
                    si=si,
                    fft=fft,
                    minimum_correlation_time=minimum_correlation_time)

            effective_sample_size = time_series_data_size / ucl_obj.si

        msg = _convergence_message(number_of_variables,
                                   converged,
                                   total_run_length,
                                   maximum_equilibration_step,
                                   equilibration_detected,
                                   equilibration_step,
                                   confidence_coefficient,
                                   relative_accuracy,
                                   absolute_accuracy,
                                   upper_confidence_limit,
                                   ucl_obj.name,
                                   relative_half_width_estimate,
                                   ucl_obj.mean,
                                   ucl_obj.std,
                                   effective_sample_size,
                                   minimum_number_of_independent_samples)

    # ndim == 2
    else:
        # Time series data temporary array
        tsd = _get_trajectory(
            get_trajectory,
            run_length=run_length,
            ndim=ndim,
            number_of_variables=number_of_variables,
            get_trajectory_args=get_trajectory_args)

        _truncated = [False] * number_of_variables
        truncate_index = [None] * number_of_variables

        # Estimate the truncation point or "warm-up" period
        # while we have sufficient data
        while need_more_data:
            for i in range(number_of_variables):
                if not _truncated[i]:
                    _truncated[i], truncate_index[i] = \
                        mser_m(tsd[i],
                               batch_size=batch_size,
                               scale=scale,
                               with_centering=with_centering,
                               with_scaling=with_scaling,
                               ignore_end=ignore_end)

            truncated = np.all(_truncated)

            # if we reached the truncation point using
            # marginal standard error rules
            if truncated and extra_check:
                # Experimental feature to make sure of detecting the
                # correct equilibrium or warm-up period

                for i in range(number_of_variables):
                    if truncated:
                        # slice a numpy array, the memory is shared
                        # between the slice and the original
                        time_series_data = tsd[i, truncate_index[i]:]

                        diff = 0
                        sample_std = 1

                        if population_mean is not None and \
                                population_mean[i] is not None:
                            sample_mean = time_series_data.mean()
                            sample_std = time_series_data.std()

                            diff = abs(population_mean[i] - sample_mean)

                            if population_standard_deviation is not None and \
                                    population_standard_deviation[i] is not None:
                                sample_std = max(
                                    sample_std,
                                    population_standard_deviation[i])

                        if population_cdf is not None and \
                           population_cdf[i] is not None:
                            if population_args is not None and \
                                    population_args[i] is not None:
                                args = population_args[i]
                            else:
                                args = None

                            if population_loc is not None and \
                                    population_loc[i] is not None:
                                loc = population_loc[i]
                            else:
                                loc = None

                            if population_scale is not None and \
                                    population_scale[i] is not None:
                                scale = population_scale[i]
                            else:
                                scale = None

                            population_median, _, _, population_std = \
                                get_distribution_stats(population_cdf[i],
                                                       args,
                                                       loc,
                                                       scale)

                            sample_median = time_series_data.median()
                            if np.isfinite(population_median):
                                diff = abs(population_median - sample_median)
                            else:
                                diff = 0

                            sample_std = time_series_data.std()
                            if np.isfinite(population_std):
                                sample_std = max(sample_std, population_std)

                        # Estimates further than 3 standard errors away can then
                        # easily be flagged as not truncated
                        if diff > 3 * sample_std:
                            truncated = False

            # Found the truncation point or "warm-up" period
            if truncated:
                need_more_data = False

            if need_more_data:
                # get the run length
                run_length = _get_run_length(run_length,
                                             run_length_factor,
                                             total_run_length,
                                             maximum_run_length)

                # We have reached the maximum limit
                if run_length == 0:
                    need_more_data = False
                else:
                    total_run_length += run_length

                    _tsd = _get_trajectory(
                        get_trajectory,
                        run_length=run_length,
                        ndim=ndim,
                        number_of_variables=number_of_variables,
                        get_trajectory_args=get_trajectory_args)
                    tsd = np.concatenate((tsd, _tsd), axis=1)

        del extra_check
        del _truncated

        equilibration_step = [index for index in truncate_index]
        equilibration_detected = True if truncated else False

        if truncated:
            for i in range(number_of_variables):
                # slice a numpy array, the memory is shared
                # between the slice and the original
                time_series_data = tsd[i, truncate_index[i]:]

                # Check to get the more accurate estimate of the
                # equilibrium or warm-up period index
                equilibration_index_estimate, _ = \
                    estimate_equilibration_length(
                        time_series_data,
                        si=si,
                        nskip=nskip,
                        fft=fft,
                        minimum_correlation_time=minimum_correlation_time,
                        ignore_end=ignore_end,
                        number_of_cores=number_of_cores)

                # Correct the equilibration step
                equilibration_step[i] += equilibration_index_estimate

        del truncate_index

        # Check the hard limit
        _check_equilibration_step(equilibration_step,
                                  maximum_equilibration_step,
                                  maximum_run_length,
                                  equilibration_detected,
                                  tsd,
                                  dump_trajectory,
                                  dump_trajectory_fp)

        upper_confidence_limit = [None] * number_of_variables
        relative_half_width_estimate = [None] * number_of_variables
        effective_sample_size = [None] * number_of_variables

        _mean = [None] * number_of_variables
        _std = [None] * number_of_variables
        _done = [False] * number_of_variables

        enough_accuracy = False
        need_more_data = True

        while need_more_data:
            enough_data = True

            for i in range(number_of_variables):
                _done[i] = False

                if enough_data:
                    # slice a numpy array, the memory is shared
                    # between the slice and the original
                    time_series_data = tsd[i, equilibration_step[i]:]
                    time_series_data_size = time_series_data.size

                    if population_standard_deviation is not None and \
                            population_standard_deviation[i] is not None:
                        pop_std = population_standard_deviation[i]
                    else:
                        pop_std = None

                    try:
                        upper_confidence_limit[i] = ucl_obj.ucl(
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
                            population_standard_deviation=pop_std,
                            si=si,
                            minimum_correlation_time=minimum_correlation_time)
                    except CVGSampleSizeError:
                        # do not have enough data and need more
                        enough_data = False
                        upper_confidence_limit[i] = None
                    except CVGError:
                        raise CVGError("Failed to compute the ucl.")

                if enough_data:
                    if relative_accuracy[i] is None:
                        enough_accuracy = \
                            upper_confidence_limit[i] < absolute_accuracy[i]
                    else:
                        # Estimate the relative half width
                        if isclose(
                                ucl_obj.mean, 0,
                                abs_tol=_DEFAULT_RELATIVE_HALF_WIDTH_ESTIMATE_ABS_TOL
                        ):
                            msg = 'It is not possible to estimate the relative '
                            msg += 'half width for the close to zero mean = '
                            msg += '{}, for the variable '.format(ucl_obj.mean)
                            msg += 'number = {}.'.format(i + 1)
                            raise CVGError(msg)

                        relative_half_width_estimate[i] = \
                            upper_confidence_limit[i] / fabs(ucl_obj.mean)

                        enough_accuracy = \
                            relative_half_width_estimate[i] < \
                            relative_accuracy[i]

                    if enough_accuracy:
                        if ucl_obj.name != 'uncorrelated_sample':
                            ucl_obj.set_indices(
                                time_series_data,
                                si=si,
                                fft=fft,
                                minimum_correlation_time=minimum_correlation_time)

                        effective_sample_size[i] = \
                            time_series_data_size / ucl_obj.si

                        if minimum_number_of_independent_samples is None or \
                                effective_sample_size[i] >= \
                            minimum_number_of_independent_samples:

                            need_more_data = False

                            if population_mean is not None and \
                                    population_mean[i] is not None:
                                need_more_data = not t_test(
                                    sample_mean=ucl_obj.mean,
                                    sample_std=ucl_obj.std,
                                    sample_size=ucl_obj.sample_size,
                                    population_mean=population_mean[i],
                                    significance_level=1-confidence_coefficient)

                            if not need_more_data and \
                                    population_standard_deviation is not None \
                                    and population_standard_deviation[i] is not None:
                                need_more_data = not chi_square_test(
                                    sample_var=ucl_obj.std * ucl_obj.std,
                                    sample_size=ucl_obj.sample_size,
                                    population_var=population_standard_deviation[i] *
                                    population_standard_deviation[i],
                                    significance_level=1-confidence_coefficient)

                            if not need_more_data and \
                                    population_cdf is not None and \
                                    population_cdf[i] is not None:

                                if population_args is not None and \
                                        population_args[i] is not None:
                                    args = population_args[i]
                                else:
                                    args = None

                                if population_loc is not None and \
                                        population_loc[i] is not None:
                                    loc = population_loc[i]
                                else:
                                    loc = None

                                if population_scale is not None and \
                                        population_scale[i] is not None:
                                    scale = population_scale[i]
                                else:
                                    scale = None

                                need_more_data = not levene_test(
                                    time_series_data=time_series_data[
                                        ucl_obj.indices],
                                    population_cdf=population_cdf[i],
                                    population_args=args,
                                    population_loc=loc,
                                    population_scale=scale,
                                    significance_level=1-confidence_coefficient)

                            if not need_more_data:
                                _mean[i] = ucl_obj.mean
                                _std[i] = ucl_obj.std
                                _done[i] = True

            if not need_more_data:
                enough_accuracy = np.all(_done)
                need_more_data = not enough_accuracy

            if need_more_data:
                # get the run length
                run_length = _get_run_length(run_length,
                                             run_length_factor,
                                             total_run_length,
                                             maximum_run_length)

                # We have reached the maximum limit
                if run_length == 0:
                    enough_accuracy = False
                    need_more_data = False
                else:
                    total_run_length += run_length

                    _tsd = _get_trajectory(
                        get_trajectory,
                        run_length=run_length,
                        ndim=ndim,
                        number_of_variables=number_of_variables,
                        get_trajectory_args=get_trajectory_args)
                    tsd = np.concatenate((tsd, _tsd), axis=1)

        msg = None
        for index, ucl in enumerate(upper_confidence_limit):
            if ucl is None:
                if msg is None:
                    msg = 'For variable number = {}, '.format(index + 1)
                else:
                    msg += '{}, '.format(index + 1)
        if msg is not None:
            msg += '. Failed to compute the UCL.'
            raise CVGError(msg)

        if dump_trajectory:
            kim_edn.dump(tsd.tolist(), dump_trajectory_fp)

        converged = enough_accuracy and not need_more_data

        if not converged:
            for i in range(number_of_variables):
                if not _done[i]:
                    # slice a numpy array, the memory is shared
                    # between the slice and the original
                    time_series_data = tsd[i, equilibration_step[i]:]
                    time_series_data_size = time_series_data.size

                    if population_standard_deviation is not None and \
                            population_standard_deviation[i] is not None:
                        pop_std = population_standard_deviation[i]
                    else:
                        pop_std = None

                    upper_confidence_limit[i] = ucl_obj.ucl(
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
                        population_standard_deviation=pop_std,
                        si=si,
                        minimum_correlation_time=minimum_correlation_time)

                    if ucl_obj.name != 'uncorrelated_sample':
                        ucl_obj.set_si(
                            time_series_data,
                            si=si,
                            fft=fft,
                            minimum_correlation_time=minimum_correlation_time)

                    _mean[i] = ucl_obj.mean
                    _std[i] = ucl_obj.std

                    effective_sample_size[i] = \
                        time_series_data_size / ucl_obj.si

        msg = _convergence_message(number_of_variables,
                                   converged,
                                   total_run_length,
                                   maximum_equilibration_step,
                                   equilibration_detected,
                                   equilibration_step,
                                   confidence_coefficient,
                                   relative_accuracy,
                                   absolute_accuracy,
                                   upper_confidence_limit,
                                   ucl_obj.name,
                                   relative_half_width_estimate,
                                   _mean,
                                   _std,
                                   effective_sample_size,
                                   minimum_number_of_independent_samples)

    # It should return the string
    if fp is None:
        if fp_format == 'json':
            return json.dumps(msg, indent=4)

        return kim_edn.dumps(msg, indent=4)

    # Otherwise it uses fp to print the message
    if fp_format == 'json':
        json.dump(msg, fp, indent=4)
        return converged

    kim_edn.dump(msg, fp, indent=4)
    return converged

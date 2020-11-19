"""Time series module."""

from typing import Callable
import sys
from math import isclose, fabs
import numpy as np
from inspect import isfunction

from .err import CVGError
from .mser_m import mser_m
from .equilibration_length import estimate_equilibration_length
from .statistical_inefficiency import \
    statistical_inefficiency,\
    r_statistical_inefficiency, \
    split_r_statistical_inefficiency, \
    split_statistical_inefficiency, \
    si_methods
from .ucl import HeidelbergerWelch, ucl, subsamples_ucl
from .utils import subsample_index

__all__ = [
    'run_length_control',
]


def run_length_control(get_trajectory,
                       *,
                       n_variables=1,
                       initial_run_length=2000,
                       run_length_factor=1.5,
                       maximum_run_length=1000000,
                       maximum_equilibration_step=None,
                       sample_size=None,
                       relative_accuracy=0.01,
                       population_standard_deviation=None,
                       confidence_coefficient=0.95,
                       heidel_welch_number_points=50,
                       fft=True,
                       test_size=None,
                       train_size=None,
                       batch_size=5,
                       scale='translate_scale',
                       with_centering=False,
                       with_scaling=False,
                       ignore_end_batch=None,
                       si='statistical_inefficiency',
                       nskip=1,
                       minimum_correlation_time=None,
                       ignore_end=None,
                       fp=None):
    """Control the length of the time series data from a simulation run.

    At each checkpoint an upper confidence limit (UCL) is approximated. If
    the relative UCL (UCL divided by the sample mean) is less than a
    prespecified value, `r`elative_accuracy`, the simulation is terminated. 
    Relative accuracy is the confidence  interval  half  width  (UCL) divided
    by sample mean. The UCL is calculated as a `confidence_coefficient%`
    confidence interval for the mean, using the portion of the time series data
    which is in the stationarity region.
    If the ratio is bigger than `relative_accuracy`, the length of the time
    series is deemed not long enough to estimate the mean with sufficient
    accuracy, which means the run should be extended.

    In order to avoid problems caused by sequential UCL evaluation cost, this
    calculation should not be repeated too frequently. Heidelberger and Welch
    (1981) [2]_ suggest increasing the run length by a factor of
    `run_length_factor > 1.5`, each time, so that estimate has the same,
    reasonably large, proportion of new data.

    The accuracy parameter `relative_accuracy` specifies the maximum relative
    error that will be allowed in the mean value of timeseries data. In other
    word, the distance from the confidence limit(s) to the mean. Which is also
    known as the precision, half-width, or margin of error. A value of 0.01
    is usually used to request two digits of accuracy, and so forth.

    The parameter `confidence_coefficient` is the confidence coefficient and
    often, the values 0.95 or 0.99 are used.
    For the confidence coefficient, `confidence_coefficient`, we can use the
    following interpretation,

    If thousands of samples of n items are drawn from a population using
    simple random sampling and a confidence interval is calculated for each
    sample, the proportion of those intervals that will include the true
    population mean is confidence_coefficient.

    The `maximum_run_length` parameter places an upper bound on how long the
    simulation will run. If the specified accuracy cannot be achieved within
    this time, the simulation will terminate and a warning message will
    appear in the report.

    The `maximum_equilibration_step` parameter places an upper bound on how
    long the simulation will run to reach equilibration or pass the "warm-up"
    period. If equilibration or "warm-up" period cannot be detected within
    this time, the simulation will terminate and a warning message will
    appear in the report. By default the `maximum_equilibration_step` is
    defined as half of the `maximum_run_length`.

    Args:
        get_trajectory (callback function): A callback function with a
            specific signature of ``get_trajectory(nstep: int) -> 1darray``
        n_variables (int, optional): number of variables in the corresponding
            time-series data from get_trajectory callback function.
            (default: 1)
        initial_run_length (int, optional): initial run length.
            (default: 2000)
        run_length_factor (float, optional): run length increasing factor.
            (default: 1.5)
        maximum_run_length (int, optional): the maximum run length represents
            a cost constraint. (default: 1000000)
        maximum_equilibration_step (int, optional): the maximum number of
            steps as an equilibration hard limit. If the algorithm finds
            equilibration_step greater than this limit it will fail. For the
            default None, the function is using ``maximum_run_length // 2`` as
            the maximum equilibration step. (default: None)
        sample_size (int, optional): maximum number of independent samples.
            (default: None)
        relative_accuracy (float, or 1darray, optional): a relative half-width 
            requirement or the accuracy parameter. Target value for the ratio
            of halfwidth to sample mean. If ``n_variables > 1``,
            ``relative_accuracy`` can be a scalar to be used for all variables
            or a 1darray of values of size n_variables. (default: 0.01)
        population_standard_deviation (float, or 1darray, optional): population
            standard deviation. If ``n_variables > 1``, and
            ``population_standard_deviation`` is given (not None), then 
            ``population_standard_deviation`` must be a 1darray of values of
            size n_variables (some of those can be None, where the population
            standard deviation is not known.) 
            E.g., [1.0, None] is given for ``n_variables = 2``, where for the
            second variable the population standard deviation is not known.
            (default: None)
        confidence_coefficient (float, optional): Probability (or confidence
            interval) and must be between 0.0 and 1.0, and represents the
            confidence for calculation of relative halfwidths estimation.
            (default: 0.95)
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
        ignore_end_batch (int, or float, or None, optional): if ``int``, it
            is the last few batch points that should be ignored. if ``float``,
            should be in ``(0, 1)`` and it is the percent of last batch points
            that should be ignored. if `None` it would be set to the
            ``batch_size``. (default: None)
        si (str, optional): statistical inefficiency method.
            (default: 'statistical_inefficiency')
        nskip (int, optional): the number of data points to skip in
            estimating ucl. (default: 1)
        minimum_correlation_time (int, optional): The minimum amount of 
            correlation function to compute in estimating ucl. The algorithm 
            terminates after computing the correlation time out to 
            minimum_correlation_time when the correlation function first
            goes negative. (default: None)
        ignore_end (int, or float, or None, optional): if ``int``, it is the
            last few points that should be ignored in estimating ucl. if
            ``float``, should be in ``(0, 1)`` and it is the percent of number
            of points that should be ignored in estimating ucl. If ``None`` it
            would be set to the one fourth of the total number of points.
            (default: None)
        fp (str, object with a write(string) method, optional): if an ``str``
            equals to ``'return'`` the function will return string of the
            analysis results on the length of the time series. Otherwise it
            must be an object with write(string) method. If it is None,
            ``sys.stdout`` will be used which prints objects on the screen.
            (default: None)

    Returns:
        bool or str:
            ``True`` if the length of the time series is long
            enough to estimate the mean with sufficient accuracy and ``False``
            otherwise. If fp is an ``str`` equals to ``'return'`` the function
            will return string of the analysis results on the length of the
            time series.

    """
    if not isfunction(get_trajectory):
        msg = 'the "get_trajectory" input is not a callback function.\n'
        msg += 'One has to provide the "get_trajectory" function as an '
        msg += 'input. It expects to have a specific signature:\n'
        msg += 'get_trajectory(nstep: int) -> 1darray,\n'
        msg += 'where nstep is the number of steps and the function '
        msg += 'should return a time-series data with the requested '
        msg += 'length equals to the number of steps.'
        raise CVGError(msg)

    if not isinstance(maximum_run_length, int):
        msg = 'maximum_run_length must be an `int`.'
        raise CVGError(msg)

    if maximum_run_length < 1:
        msg = 'maximum_run_length must be a positive `int` '
        msg += 'greater than or equal 1'
        raise CVGError(msg)

    if maximum_equilibration_step is None:
        maximum_equilibration_step = maximum_run_length // 2

    # Set the hard limit for the equilibration step
    if not isinstance(maximum_equilibration_step, int):
        msg = 'maximum_equilibration_step must be an `int`.'
        raise CVGError(msg)

    if maximum_equilibration_step < 1 or \
            maximum_equilibration_step >= maximum_run_length:
        msg = 'maximum_equilibration_step = '
        msg += '{} must be a positive '.format(maximum_equilibration_step)
        msg += '`int` greater than or equal 1 and less than '
        msg += 'maximum_run_length = {}.'.format(maximum_run_length)
        raise CVGError(msg)

    if not isinstance(n_variables, int):
        msg = 'n_variables must be an `int`.'
        raise CVGError(msg)

    if n_variables < 1:
        msg = 'n_variables must be a positive `int` greater than or equal 1.'
        raise CVGError(msg)

    if fp is None:
        fp = sys.stdout
    elif isinstance(fp, str):
        if fp != 'return':
            msg = 'Keyword argument `fp` is an `str` and not equal to '
            msg += '"return".'
            raise CVGError(msg)
        fp = None
    elif not hasattr(fp, 'write'):
        msg = 'Keyword argument `fp` must be either an `str` and equal '
        msg += 'to "return", or None, or an object with write(string) '
        msg += 'method.'
        raise CVGError(msg)

    # Initialize
    if n_variables == 1:
        ndim = 1
        if np.size(relative_accuracy) != 1:
            msg = 'relative_accuracy must be a `float`.'
            raise CVGError(msg)

    else:
        ndim = 2

        if np.size(relative_accuracy) == 1:
            relative_accuracy = \
                np.array([relative_accuracy] * n_variables, dtype=np.float64)
        elif np.size(relative_accuracy) != n_variables:
            msg = 'relative_accuracy must be a scalar (a `float`) or a '
            msg += '1darray of size = {}.'.format(n_variables)
            raise CVGError(msg)

        if population_standard_deviation is not None:
            if np.size(population_standard_deviation) != n_variables:
                msg = 'population_standard_deviation must be a 1darray of '
                msg += 'size = {}.'.format(n_variables)
                raise CVGError(msg)

            population_standard_deviation = \
                np.array(population_standard_deviation, copy=False)

    try:
        # Initialize the HeidelbergerWelch object
        heidel_welch = HeidelbergerWelch(
            confidence_coefficient=confidence_coefficient,
            heidel_welch_number_points=heidel_welch_number_points)
    except CVGError:
        msg = "Failed to initialize the HeidelbergerWelch object."
        raise CVGError(msg)

    # Initial length
    run_length = min(initial_run_length, maximum_run_length)
    _run_length = min(int(initial_run_length * run_length_factor),
                      maximum_run_length)

    equilibration_step = 0
    total_run_length = 0

    # Time series data
    t = None

    if ndim == 1:
        # Estimate the equilibration or "warm-up" period
        while True:
            total_run_length += run_length

            if total_run_length >= maximum_run_length:
                total_run_length -= run_length
                run_length = maximum_run_length - total_run_length
                total_run_length = maximum_run_length

            try:
                _t = get_trajectory(step=run_length)
            except:
                msg = 'failed to get the time-series data or do the '
                msg += 'simulation for {} number of '.format(run_length)
                msg += 'steps.'
                raise CVGError(msg)

            # Extra check
            if not np.all(np.isfinite(_t)):
                msg = 'there is/are value/s in the input which is/are '
                msg += 'non-finite or not number.'
                raise CVGError(msg)

            if t is None:
                t = np.array(_t, dtype=np.float64)
                if t.ndim != ndim:
                    msg = 'the return from the "get_trajectory" function '
                    msg += 'has a wrong dimension of {} != 1.'.format(t.ndim)
                    raise CVGError(msg)
            else:
                _t = np.array(_t, copy=False, dtype=np.float64)
                t = np.concatenate((t, _t))

            truncated, truncate_index = \
                mser_m(t,
                       batch_size=batch_size,
                       scale=scale,
                       with_centering=with_centering,
                       with_scaling=with_scaling,
                       ignore_end_batch=ignore_end_batch)

            if truncated:
                break

            # We have reached the maximum limit
            if total_run_length == maximum_run_length:
                break

            run_length = _run_length

        if truncated:
            # slice a numpy array, the memory is shared
            # between the slice and the original
            time_series_data = t[truncate_index:]
            time_series_data_size = time_series_data.size

            equilibration_index_estimate, _ = \
                estimate_equilibration_length(
                    time_series_data,
                    si=si,
                    nskip=nskip,
                    fft=(time_series_data_size > 30 and fft),
                    minimum_correlation_time=minimum_correlation_time)
            equilibration_step = truncate_index + \
                equilibration_index_estimate
        else:
            equilibration_step, _ = \
                estimate_equilibration_length(
                    t,
                    si=si,
                    nskip=nskip,
                    fft=(total_run_length > 30 and fft),
                    minimum_correlation_time=minimum_correlation_time)

        # Check the hard limit
        if equilibration_step >= maximum_equilibration_step:
            msg = 'the equilibration or "warm-up" period is detected '
            msg += 'at step = {}, which '.format(equilibration_step)
            msg += 'is greater than the maximum number of allowed steps '
            msg += 'for the equilibration detection = '
            msg += '{}.'.format(maximum_equilibration_step)
            raise CVGError(msg)

        run_length = _run_length

        si_func = si_methods[si]
        statistical_inefficiency_estimate = None

        while True:
            # slice a numpy array, the memory is shared
            # between the slice and the original
            time_series_data = t[equilibration_step:]
            time_series_data_size = time_series_data.size

            if time_series_data_size < 100 or population_standard_deviation:
                # Compute the statitical inefficiency of a time series
                try:
                    statistical_inefficiency_estimate = si_func(
                        time_series_data,
                        fft=(time_series_data_size > 30 and fft),
                        minimum_correlation_time=minimum_correlation_time)
                except CVGError:
                    statistical_inefficiency_estimate = float(
                        time_series_data_size)

                try:
                    subsample_indices = subsample_index(
                        time_series_data,
                        si=statistical_inefficiency_estimate,
                        fft=(time_series_data_size > 30 and fft),
                        minimum_correlation_time=minimum_correlation_time)
                except CVGError:
                    msg = 'Failed to compute the indices of uncorrelated '
                    msg += 'subsamples of the time series data.'
                    raise CVGError(msg)

                # Get the upper confidence limit
                try:
                    upper_confidence_limit = subsamples_ucl(
                        time_series_data,
                        confidence_coefficient=confidence_coefficient,
                        population_standard_deviation=population_standard_deviation,
                        subsample_indices=subsample_indices,
                        si=statistical_inefficiency_estimate,
                        fft=(subsample_indices.size > 30 and fft),
                        minimum_correlation_time=minimum_correlation_time)
                except CVGError:
                    msg = "Failed to get the upper confidence limit."
                    raise CVGError(msg)

                # Compute the mean
                _mean = np.mean(time_series_data[subsample_indices])
            else:
                # Get the upper confidence limit
                try:
                    upper_confidence_limit = ucl(
                        time_series_data,
                        confidence_coefficient=confidence_coefficient,
                        heidel_welch_number_points=heidel_welch_number_points,
                        fft=fft,
                        test_size=test_size,
                        train_size=train_size,
                        heidel_welch=heidel_welch)
                except CVGError:
                    msg = "Failed to get the upper confidence limit."
                    raise CVGError(msg)

                # Compute the mean
                _mean = np.mean(time_series_data)

            # Estimat the relative half width
            if isclose(_mean, 0, rel_tol=1e-14):
                relative_half_width_estimate = \
                    upper_confidence_limit / 1e-14
            else:
                relative_half_width_estimate = \
                    upper_confidence_limit / fabs(_mean)

            # The run stopping criteria
            if relative_half_width_estimate < relative_accuracy:
                if statistical_inefficiency_estimate is None:
                    # Compute the statitical inefficiency of a time series
                    try:
                        statistical_inefficiency_estimate = si_func(
                            time_series_data,
                            fft=(time_series_data_size > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
                    except:
                        statistical_inefficiency_estimate = float(
                            time_series_data_size)

                    effective_sample_size = time_series_data_size / \
                        statistical_inefficiency_estimate

                    statistical_inefficiency_estimate = None
                else:
                    effective_sample_size = time_series_data_size / \
                        statistical_inefficiency_estimate

                if sample_size is None:
                    # It should stop
                    msg = '=' * 37
                    msg += '\n'
                    msg += 'The equilibration happens at step = '
                    msg += '{}\n'.format(equilibration_step)
                    msg += 'Total run length = {}\n'.format(total_run_length)
                    msg += 'The relative half width with '
                    msg += '{}% '.format(confidence_coefficient * 100)
                    msg += 'confidence of the estimation for the mean meets '
                    msg += 'the required relative accuracy = '
                    msg += '{}\n'.format(relative_accuracy)
                    msg += 'The mean of the time-series data lies in: ('
                    msg += '{} +/- '.format(_mean)
                    msg += '{})\n'.format(upper_confidence_limit)
                    msg += 'The standard deviation of the equilibrated part '
                    msg += 'of the time-series data = '
                    msg += '{}\n'.format(np.std(t[equilibration_step:]))
                    msg += 'Effective sample size = '
                    msg += '{}\n'.format(int(effective_sample_size))
                    msg += '=' * 37
                    msg += '\n'
                    if fp is None:
                        return msg

                    print(msg, file=fp)
                    return True

                # We should check for enough sample size
                if effective_sample_size >= sample_size:
                    # It should stop
                    msg = '=' * 37
                    msg += '\n'
                    msg += 'The equilibration happens at step = '
                    msg += '{}\n'.format(equilibration_step)
                    msg += 'Total run length = '
                    msg += '{}\n'.format(total_run_length)
                    msg += 'The relative half width with '
                    msg += '{}% '.format(confidence_coefficient * 100)
                    msg += 'confidence of the estimation for the mean meets '
                    msg += 'the required relative accuracy = '
                    msg += '{}\n'.format(relative_accuracy)
                    msg += 'The mean of the time-series data lies in: ('
                    msg += '{} +/- '.format(_mean)
                    msg += '{})\n'.format(upper_confidence_limit)
                    msg += 'The standard deviation of the equilibrated '
                    msg += 'part of the time-series data = '
                    msg += '{}\n'.format(np.std(t[equilibration_step:]))
                    msg += 'Effective sample size = '
                    msg += '{} > '.format(int(effective_sample_size))
                    msg += '{}, '.format(sample_size)
                    msg += 'requested number of sample size\n'
                    msg += '=' * 37
                    msg += '\n'
                    if fp is None:
                        return msg

                    print(msg, file=fp)
                    return True

            total_run_length += run_length

            if total_run_length >= maximum_run_length:
                total_run_length -= run_length
                run_length = maximum_run_length - total_run_length
                total_run_length = maximum_run_length

            if run_length == 0:
                # We have reached the maximum limit
                msg = 'the length of the time series data = '
                msg += '{} is not long '.format(maximum_run_length)
                msg += 'enough to estimate the mean with sufficient '
                if sample_size is None:
                    msg += 'accuracy.\n'
                else:
                    msg += 'accuracy or enough requested sample size.\n'
                if fp is None:
                    return msg

                print(msg, file=fp)
                return False

            try:
                _t = get_trajectory(step=run_length)
            except:
                msg = 'failed to get the time-series data or do the '
                msg += 'simulation for {} number of '.format(run_length)
                msg += 'steps.'
                raise CVGError(msg)

            # Extra check
            if not np.all(np.isfinite(_t)):
                msg = 'there is/are value/s in the input which is/are '
                msg += 'non-finite or not number.'
                raise CVGError(msg)

            _t = np.asarray(_t, dtype=np.float64)
            t = np.concatenate((t, _t))

        # We have reached the maximum limit
        msg = 'the length of the time series data = '
        msg += '{} is not long '.format(maximum_run_length)
        msg += 'enough to estimate the mean with sufficient '
        if sample_size is None:
            msg += 'accuracy.\n'
        else:
            msg += 'accuracy or enough requested sample size.\n'
        if fp is None:
            return msg

        print(msg, file=fp)
        return False
    # ndim == 2
    else:
        _truncated = np.array([False] * n_variables)
        truncate_index = np.empty(n_variables, dtype=int)

        # Estimate the equilibration or "warm-up" period
        while True:
            total_run_length += run_length

            if total_run_length >= maximum_run_length:
                total_run_length -= run_length
                run_length = maximum_run_length - total_run_length
                total_run_length = maximum_run_length

            try:
                _t = get_trajectory(step=run_length)
            except:
                msg = 'failed to get the time-series data or do the '
                msg += 'simulation for {} number of '.format(run_length)
                msg += 'steps.'
                raise CVGError(msg)

            # Extra check
            if not np.all(np.isfinite(_t)):
                msg = 'there is/are value/s in the input which is/are '
                msg += 'non-finite or not number.'
                raise CVGError(msg)

            if t is None:
                t = np.array(_t, dtype=np.float64)
                if t.ndim != ndim:
                    msg = 'the return of "get_trajectory" function has a '
                    msg += 'wrong dimension of {} != '.format(t.ndim)
                    msg += '{}.\n'.format(ndim)
                    msg += 'In a two-dimensional return array of '
                    msg += '"get_trajectory" function, each row corresponds '
                    msg += 'to the time series data for one variable.'
                    raise CVGError(msg)
                if n_variables != np.shape(t)[0]:
                    msg = 'the return of "get_trajectory" function has a '
                    msg += 'wrong number of variables = '
                    msg += '{} != '.format(np.shape(t)[0])
                    msg += '{}.\n'.format(n_variables)
                    msg += 'In a two-dimensional return array of '
                    msg += '"get_trajectory" function, each row corresponds '
                    msg += 'to the time series data for one variable.'
                    raise CVGError(msg)
            else:
                _t = np.array(_t, copy=False, dtype=np.float64)
                t = np.concatenate((t, _t), axis=1)

            for i in range(n_variables):
                if not _truncated[i]:
                    _truncated[i], truncate_index[i] = \
                        mser_m(t[i],
                               batch_size=batch_size,
                               scale=scale,
                               with_centering=with_centering,
                               with_scaling=with_scaling,
                               ignore_end_batch=ignore_end_batch)

            truncated = np.all(_truncated)
            if truncated:
                break

            # We have reached the maximum limit
            if total_run_length == maximum_run_length:
                break

            run_length = _run_length

        equilibration_step = np.empty(n_variables, dtype=int)

        if truncated:
            del(_truncated)
            for i in range(n_variables):
                # slice a numpy array, the memory is shared
                # between the slice and the original
                time_series_data = t[i, truncate_index[i]:]
                time_series_data_size = time_series_data.size

                equilibration_index_estimate, _ = \
                    estimate_equilibration_length(
                        time_series_data,
                        si=si,
                        nskip=nskip,
                        fft=(time_series_data_size > 30 and fft),
                        minimum_correlation_time=minimum_correlation_time)
                equilibration_step[i] = truncate_index[i] + \
                    equilibration_index_estimate
        else:
            for i in range(n_variables):
                if _truncated[i]:
                    # slice a numpy array, the memory is shared
                    # between the slice and the original
                    time_series_data = t[i, truncate_index[i]:]
                    time_series_data_size = time_series_data.size

                    equilibration_index_estimate, _ = \
                        estimate_equilibration_length(
                            time_series_data,
                            si=si,
                            nskip=nskip,
                            fft=(time_series_data_size > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
                    equilibration_step[i] = truncate_index[i] + \
                        equilibration_index_estimate
                else:
                    equilibration_step[i], _ = \
                        estimate_equilibration_length(
                            t[i],
                            si=si,
                            nskip=nskip,
                            fft=(total_run_length > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
            del(_truncated)

        # Check the hard limit
        if np.any(equilibration_step > maximum_equilibration_step):
            for i in range(n_variables):
                msg = 'the equilibration or "warm-up" period for '
                msg += 'variable number {} is detected at '.format(i + 1)
                msg += 'step = {}.\n'.format(equilibration_step[i])
                if equilibration_step[i] >= maximum_equilibration_step:
                    msg += '\nThe detected step number is greater than the '
                    msg += 'maximum number of allowed steps = '
                    msg += '{} for '.format(equilibration_step)
                    msg += 'equilibration detection.\n'
            raise CVGError(msg)

        del(truncate_index)

        run_length = _run_length

        si_func = si_methods[si]

        statistical_inefficiency_estimate = \
            np.array([np.nan] * n_variables, dtype=np.float64)
        upper_confidence_limit = np.empty(n_variables, dtype=np.float64)
        _mean = np.empty(n_variables, dtype=np.float64)
        _done = np.array([False] * n_variables)
        relative_half_width_estimate = np.empty(n_variables, dtype=np.float64)
        effective_sample_size = np.empty(n_variables, dtype=np.float64)

        while True:
            for i in range(n_variables):
                # slice a numpy array, the memory is shared
                # between the slice and the original
                time_series_data = t[i, equilibration_step[i]:]
                time_series_data_size = time_series_data.size

                if time_series_data_size < 100 or \
                        population_standard_deviation is not None:

                    # Compute the statitical inefficiency of a time series
                    try:
                        statistical_inefficiency_estimate[i] = si_func(
                            time_series_data,
                            fft=(time_series_data_size > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
                    except CVGError:
                        statistical_inefficiency_estimate[i] = float(
                            time_series_data_size)

                    try:
                        subsample_indices = subsample_index(
                            time_series_data,
                            si=statistical_inefficiency_estimate[i],
                            fft=(time_series_data_size > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
                    except CVGError:
                        msg = 'Failed to compute the indices of uncorrelated '
                        msg += 'subsamples of the time series data.'
                        raise CVGError(msg)

                    # Get the upper confidence limit
                    try:
                        upper_confidence_limit[i] = subsamples_ucl(
                            time_series_data,
                            confidence_coefficient=confidence_coefficient,
                            population_standard_deviation=population_standard_deviation,
                            subsample_indices=subsample_indices,
                            si=statistical_inefficiency_estimate[i],
                            fft=(subsample_indices.size > 30 and fft),
                            minimum_correlation_time=minimum_correlation_time)
                    except CVGError:
                        msg = "Failed to get the upper confidence limit."
                        raise CVGError(msg)

                    # Compute the mean
                    _mean[i] = np.mean(time_series_data[subsample_indices])

                else:
                    try:
                        # Get the upper confidence limit
                        upper_confidence_limit[i] = ucl(
                            time_series_data,
                            confidence_coefficient=confidence_coefficient,
                            heidel_welch_number_points=heidel_welch_number_points,
                            fft=fft,
                            test_size=test_size,
                            train_size=train_size,
                            heidel_welch=heidel_welch)
                    except CVGError:
                        msg = "Failed to get the upper confidence limit."
                        raise CVGError(msg)

                    # Compute the mean
                    _mean[i] = np.mean(time_series_data)

                # Estimat the relative half width
                if isclose(_mean[i], 0, rel_tol=1e-14):
                    relative_half_width_estimate[i] = \
                        upper_confidence_limit[i] / 1e-14
                else:
                    relative_half_width_estimate[i] = \
                        upper_confidence_limit[i] / fabs(_mean[i])

                # The run stopping criteria
                if _done[i]:
                    if relative_half_width_estimate[i] > relative_accuracy[i]:
                        msg += 'for variable number {},\n'.format(i + 1)
                        msg += 'The time series data diverges after meeting '
                        msg += 'the required relative accuracy.'
                        raise CVGError(msg)
                else:
                    if relative_half_width_estimate[i] < relative_accuracy[i]:
                        if np.isnan(statistical_inefficiency_estimate[i]):
                            # Compute the statitical inefficiency of a
                            # time series
                            try:
                                statistical_inefficiency_estimate[i] = \
                                    si_func(
                                    time_series_data,
                                    fft=fft,
                                    minimum_correlation_time=minimum_correlation_time)
                            except CVGError:
                                statistical_inefficiency_estimate[i] = \
                                    float(time_series_data_size)

                            effective_sample_size[i] = time_series_data_size / \
                                statistical_inefficiency_estimate[i]

                            statistical_inefficiency_estimate[i] = np.nan
                        else:
                            effective_sample_size[i] = time_series_data_size / \
                                statistical_inefficiency_estimate[i]

                        if sample_size is None:
                            # It should stop
                            _done[i] = True
                        else:
                            # We should check for enough sample size
                            _done[i] = effective_sample_size[i] >= sample_size

            done = np.all(_done)
            if done:
                # It should stop
                msg = '=' * 37
                msg += '\n'
                for i in range(n_variables):
                    msg += 'for variable number {},\n'.format(i + 1)
                    msg += 'The equilibration happens at step = '
                    msg += '{}\n'.format(equilibration_step[i])
                    msg += 'Total run length = '
                    msg += '{}\n'.format(total_run_length)
                    msg += 'The relative half width with '
                    msg += '{}% '.format(confidence_coefficient * 100)
                    msg += 'confidence of the estimation for the mean meets '
                    msg += 'the required relative accuracy = '
                    msg += '{}\n'.format(relative_accuracy[i])
                    msg += 'The mean of the time-series data lies in: ('
                    msg += '{} +/- '.format(_mean[i])
                    msg += '{})\n'.format(upper_confidence_limit[i])
                    msg += 'The standard deviation of the equilibrated '
                    msg += 'part of the time-series data = '
                    msg += '{}\n'.format(np.std(t[i, equilibration_step[i]:]))
                    if sample_size is None:
                        msg += 'Effective sample size = '
                        msg += '{}\n'.format(int(effective_sample_size[i]))
                    else:
                        msg += 'Effective sample size = '
                        msg += '{} > '.format(int(effective_sample_size[i]))
                        msg += '{}, '.format(sample_size)
                        msg += 'requested number of sample size\n'
                    if i < n_variables - 1:
                        msg += '-' * 37
                        msg += '\n'
                msg += '=' * 37
                msg += '\n'
                if fp is None:
                    return msg

                print(msg, file=fp)
                return True

            total_run_length += run_length

            if total_run_length >= maximum_run_length:
                total_run_length -= run_length
                run_length = maximum_run_length - total_run_length
                total_run_length = maximum_run_length

            if run_length == 0:
                # We have reached the maximum limit
                msg = 'the length of the time series data = '
                msg += '{} is not long '.format(maximum_run_length)
                msg += 'enough to estimate the mean with sufficient '
                if sample_size is None:
                    msg += 'accuracy.\n'
                else:
                    msg += 'accuracy or enough requested sample size.\n'
                if fp is None:
                    return msg

                print(msg, file=fp)
                return False

            try:
                _t = get_trajectory(step=run_length)
            except:
                msg = 'failed to get the time-series data or do the '
                msg += 'simulation for {} number of '.format(run_length)
                msg += 'steps.'
                raise CVGError(msg)

            # Extra check
            if not np.all(np.isfinite(_t)):
                msg = 'there is/are value/s in the input which is/are '
                msg += 'non-finite or not number.'
                raise CVGError(msg)

            _t = np.asarray(_t, dtype=np.float64)
            t = np.concatenate((t, _t), axis=1)

        msg = 'the length of the time series data = '
        msg += '{} '.format(maximum_run_length)
        msg += 'is not long enough to estimate the mean with sufficient '
        if sample_size is None:
            msg += 'accuracy.\n'
        else:
            msg += 'accuracy or enough requested sample size.\n'
        if fp is None:
            return msg

        print(msg, file=fp)
        return False

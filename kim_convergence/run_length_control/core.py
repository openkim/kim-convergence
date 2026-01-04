r"""
Core orchestration of the run-length control algorithm.

This module provides the public entry point ``run_length_control`` and
coordinates the high-level stages of the algorithm:

  - Initial setup and validation of inputs
  - Normalization and validation of per-variable parameters
  - Equilibration detection (warm-up period)
  - Statistical convergence checking (accuracy on mean estimates)
  - Generation and output of the final convergence report

The detailed implementation of each stage is delegated to private helper modules
within the ``kim_convergence.run_length_control`` module for better modularity
and maintainability.

The only public function is ``run_length_control``. All other symbols in this
module and submodules are private implementation details.
"""

import numpy as np
from typing import Any, Callable, Optional, Union

from kim_convergence._default import (
    _DEFAULT_CONFIDENCE_COEFFICIENT,
    _DEFAULT_CONFIDENCE_INTERVAL_APPROXIMATION_METHOD,
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    _DEFAULT_FFT,
    _DEFAULT_TEST_SIZE,
    _DEFAULT_TRAIN_SIZE,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_SCALE_METHOD,
    _DEFAULT_WITH_CENTERING,
    _DEFAULT_WITH_SCALING,
    _DEFAULT_IGNORE_END,
    _DEFAULT_NUMBER_OF_CORES,
    _DEFAULT_SI,
    _DEFAULT_NSKIP,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
)


from ._accuracy import _check_accuracy
from ._convergence import (
    _convergence_message,
    _convergence_stage,
    _output_convergence_report,
)
from ._equilibration import _equilibration_stage
from ._population import _validate_population_params
from ._setup import _setup_algorithm
from ._variable_list_factory import _make_variable_list

__all__ = ["run_length_control"]


def run_length_control(
    get_trajectory: Callable,
    get_trajectory_args: Optional[dict] = None,
    *,
    number_of_variables: int = 1,
    initial_run_length: int = 10000,
    run_length_factor: float = 1.0,
    maximum_run_length: int = 1000000,
    maximum_equilibration_step: Optional[int] = None,
    minimum_number_of_independent_samples: Optional[int] = None,
    relative_accuracy: Union[float, list[Optional[float]], np.ndarray, None] = 0.1,
    absolute_accuracy: Union[float, list[Optional[float]], np.ndarray, None] = 0.1,
    population_mean: Union[float, list[Optional[float]], np.ndarray, None] = None,
    population_standard_deviation: Union[
        float, list[Optional[float]], np.ndarray, None
    ] = None,
    population_cdf: Union[str, list[Optional[str]], None] = None,
    population_args: Union[tuple, list[Optional[tuple]], None] = None,
    population_loc: Union[float, list[Optional[float]], np.ndarray, None] = None,
    population_scale: Union[float, list[Optional[float]], np.ndarray, None] = None,
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
    si: str = _DEFAULT_SI,  # type: ignore[assignment]
    nskip: Optional[int] = _DEFAULT_NSKIP,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
    dump_trajectory: bool = False,
    dump_trajectory_fp: str = "kim_convergence_trajectory.edn",
    fp: Any = None,
    fp_format: str = "txt",
) -> Union[str, bool]:
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
        ...     global start, stop
        ...     start = stop
        ...     if 100000 < start + step:
        ...         step = 100000 - start
        ...     stop += step
        ...     data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
        ...     return data

        or,

        >>> targs = {'start': 0, 'stop': 0}
        >>> def get_trajectory(step, targs):
        ...     targs['start'] = targs['stop']
        ...     if 100000 < targs['start'] + step:
        ...         step = 100000 - targs['start']
        ...     targs['stop'] += step
        ...     data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
        ...     return data

    Then it continues drawing observations until some pre-specified level of
    absolute or relative precision has been reached.

    The relative ``precision`` is defined as a half-width of the estimator's
    confidence interval (CI).

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
    (1981) [heidelberger1981]_ suggested increasing the run length by a factor
    `run_length_factor > 1.5`, each time, so that estimate has the same,
    of reasonably large proportion of new data.

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

    Note:
        By default, the algorithm will use ``relative_accuracy`` as a
        termination criterion, and in case of failure, it switches to use the
        ``absolute_accuracy``.

        If using the ``absolute_accuracy`` is desired, one should set the
        ``relative_accuracy`` to None.

        Examples:

        >>> run_length_control(get_trajectory,
        ...                    number_of_variables=1,
        ...                    relative_accuracy=None
        ...                    absolute_accuracy=0.1)

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
        ...                    number_of_variables=3,
        ...                    relative_accuracy=[0.1, 0.05, None]
        ...                    absolute_accuracy=[0.1, 0.05, 0.1])

        or,

        >>> run_length_control(get_trajectory,
        ...                    number_of_variables=3,
        ...                    relative_accuracy=[None, 0.05, None]
        ...                    absolute_accuracy=[0.1,  0.05, 0.1])

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
                ...                    number_of_variables=3,
                ...                    population_mean=[None, 297., None])

        population_standard_deviation (float, or 1darray, optional): population
            standard deviation. (default: None)

            Note:
                For ``number_of_variables > 1``, and if
                ``population_standard_deviation`` is provided, it should be a
                list or array of values. It should be set to None for variables
                which we do not intend to use this extra measure.

                Examples:

                >>> run_length_control(
                ...     get_trajectory,
                ...     number_of_variables=3,
                ...     population_mean=[None, 297., None],
                ...     population_standard_deviation=[None, 10., None])

        population_cdf (str, or 1darray, optional): The name of a distribution.
            (default: None)

            Examples:
            >>> run_length_control(
            ...     get_trajectory,
            ...     number_of_variables=2,
            ...     population_cdf=[None, 'gamma'],
            ...     population_args=[None, (1.99,)],
            ...     population_loc=[None, None],
            ...     population_scale=[None, None])

            or,

            >>> run_length_control(
            ...     get_trajectory,
            ...     number_of_variables=2,
            ...     population_mean=[297., None],
            ...     population_standard_deviation=[10., None],
            ...     population_cdf=[None, 'gamma'],
            ...     population_args=[None, (1.99,)],
            ...     population_loc=[None, None],
            ...     population_scale=[None, None])

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
            a file. (default: 'kim_convergence_trajectory.edn')
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

    # 1. Setup: validate inputs and initialize UCL object
    maximum_equilibration_step, ucl_obj = _setup_algorithm(
        get_trajectory=get_trajectory,
        number_of_variables=number_of_variables,
        initial_run_length=initial_run_length,
        run_length_factor=run_length_factor,
        maximum_run_length=maximum_run_length,
        maximum_equilibration_step=maximum_equilibration_step,
        minimum_number_of_independent_samples=minimum_number_of_independent_samples,
        confidence_interval_approximation_method=confidence_interval_approximation_method,
        confidence_coefficient=confidence_coefficient,
        heidel_welch_number_points=heidel_welch_number_points,
        number_of_cores=number_of_cores,
        minimum_correlation_time=minimum_correlation_time,
    )

    # 2.0 Normalize per-variable parameters to lists
    relative_accuracy_list: list[Optional[float]] = _make_variable_list(
        relative_accuracy, number_of_variables
    )
    absolute_accuracy_list: list[Optional[float]] = _make_variable_list(
        absolute_accuracy, number_of_variables
    )
    # 2.1 Validate accuracy parameters
    _check_accuracy(number_of_variables, relative_accuracy_list, absolute_accuracy_list)

    # 2.2 Normalize population parameters to lists
    population_mean_list: list[Optional[float]] = _make_variable_list(
        population_mean, number_of_variables
    )
    population_standard_deviation_list: list[Optional[float]] = _make_variable_list(
        population_standard_deviation, number_of_variables
    )
    population_cdf_list: list[Optional[str]] = _make_variable_list(
        population_cdf, number_of_variables
    )
    if number_of_variables == 1:
        # Wrap in tuple so _make_variable_list treats args as a single element,
        # not as a sequence to broadcast
        population_args = (population_args,)
    population_args_list: list[Optional[tuple]] = _make_variable_list(
        population_args, number_of_variables
    )
    population_loc_list: list[Optional[float]] = _make_variable_list(
        population_loc, number_of_variables
    )
    population_scale_list: list[Optional[float]] = _make_variable_list(
        population_scale, number_of_variables
    )
    # 2.3 Validate population parameters
    _validate_population_params(
        number_of_variables,
        population_mean_list,
        population_standard_deviation_list,
        population_cdf_list,
        population_args_list,
        population_loc_list,
        population_scale_list,
    )

    if get_trajectory_args is None:
        get_trajectory_args = {}

    # 3. Equilibration stage: detect stationary region
    tsd, run_length, total_run_length, equilibration_step, equilibration_detected = (
        _equilibration_stage(
            get_trajectory=get_trajectory,
            get_trajectory_args=get_trajectory_args,
            number_of_variables=number_of_variables,
            initial_run_length=initial_run_length,
            run_length_factor=run_length_factor,
            maximum_run_length=maximum_run_length,
            maximum_equilibration_step=maximum_equilibration_step,
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
            ignore_end=ignore_end,
            population_mean_list=population_mean_list,
            population_standard_deviation_list=population_standard_deviation_list,
            population_cdf_list=population_cdf_list,
            population_args_list=population_args_list,
            population_loc_list=population_loc_list,
            population_scale_list=population_scale_list,
            si=si,
            nskip=nskip,
            fft=fft,
            minimum_correlation_time=minimum_correlation_time,
            number_of_cores=number_of_cores,
            dump_trajectory=dump_trajectory,
            dump_trajectory_fp=dump_trajectory_fp,
        )
    )

    # 4. Convergence stage: achieve required accuracy
    (
        converged,
        total_run_length,
        mean,
        std,
        effective_sample_size,
        upper_confidence_limit,
        relative_half_width_estimate,
        relative_accuracy_undefined,
    ) = _convergence_stage(
        get_trajectory=get_trajectory,
        get_trajectory_args=get_trajectory_args,
        number_of_variables=number_of_variables,
        tsd=tsd,
        equilibration_step=equilibration_step,
        run_length=run_length,
        total_run_length=total_run_length,
        run_length_factor=run_length_factor,
        maximum_run_length=maximum_run_length,
        minimum_number_of_independent_samples=minimum_number_of_independent_samples,
        minimum_correlation_time=minimum_correlation_time,
        relative_accuracy_list=relative_accuracy_list,
        absolute_accuracy_list=absolute_accuracy_list,
        population_mean_list=population_mean_list,
        population_standard_deviation_list=population_standard_deviation_list,
        population_cdf_list=population_cdf_list,
        population_args_list=population_args_list,
        population_loc_list=population_loc_list,
        population_scale_list=population_scale_list,
        ucl_obj=ucl_obj,
        confidence_coefficient=confidence_coefficient,
        heidel_welch_number_points=heidel_welch_number_points,
        fft=fft,
        batch_size=batch_size,
        scale=scale,
        with_centering=with_centering,
        with_scaling=with_scaling,
        test_size=test_size,
        train_size=train_size,
        si=si,
        dump_trajectory=dump_trajectory,
        dump_trajectory_fp=dump_trajectory_fp,
    )

    # 5. Build convergence report
    cmsg = _convergence_message(
        number_of_variables=number_of_variables,
        converged=converged,
        total_run_length=total_run_length,
        maximum_equilibration_step=maximum_equilibration_step,
        equilibration_detected=equilibration_detected,
        equilibration_step=equilibration_step,
        confidence_coefficient=confidence_coefficient,
        relative_accuracy=relative_accuracy_list,
        absolute_accuracy=absolute_accuracy_list,
        upper_confidence_limit=upper_confidence_limit,
        upper_confidence_limit_method=confidence_interval_approximation_method,
        relative_half_width_estimate=relative_half_width_estimate,
        time_series_data_mean=mean,
        time_series_data_std=std,
        effective_sample_size=effective_sample_size,
        minimum_number_of_independent_samples=minimum_number_of_independent_samples,
        relative_accuracy_undefined=relative_accuracy_undefined,
    )

    # 6. Output report (or return string)
    return _output_convergence_report(
        cmsg=cmsg, converged=converged, fp=fp, fp_format=fp_format
    )

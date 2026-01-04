r"""Time series statistical inefficiency module.

The statistical inefficiency is the limiting number of steps to obtain
uncorrelated configurations.
"""

from math import isclose
import numpy as np
from typing import Optional, Union

from kim_convergence._default import (
    _DEFAULT_ABS_TOL,
    _DEFAULT_FFT,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
    _DEFAULT_SI,
)
from kim_convergence import (
    auto_covariance,
    auto_correlate,
    cross_correlate,
    CRError,
    CRSampleSizeError,
)


__all__ = [
    "geyer_r_statistical_inefficiency",
    "geyer_split_r_statistical_inefficiency",
    "geyer_split_statistical_inefficiency",
    "integrated_auto_correlation_time",
    "si_methods",
    "statistical_inefficiency",
]


def statistical_inefficiency(
    x: Union[np.ndarray, list[float]],
    y: Union[np.ndarray, list[float], None] = None,
    *,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
) -> float:
    r"""Compute the statistical inefficiency.

    The statistical inefficiency :math:`si` of the observable :math:`x`
    of a time series :math:`\{X\}_{t=0}^n` is formally defined as,

    .. math::

        si &\equiv 1 + 2\tau \\
        \tau &\equiv \sum_{t=0}^n {\left( 1 - \frac{t}{n} \right) C\left(t\right)} \\
        C\left(t\right) &\equiv \frac{<x(X_{t_0})x(X_{t_0+t})> - {<x>}^2}{<x^2>-{<x>}^2}

    where :math:`\tau` denotes the integrated auto-correlation time and
    :math:`C\left(t\right)` is the normalized fluctuation auto-correlation
    function of the observable :math:`x`

    Note:
        The behavior is updated. Suppose the time series data is an array of
        (constant) numbers with standard deviation close to zero within
        `abs_tol=1e-18`, where `abs(a) <= max(1e-9 * abs(a), abs_tol)`.
        In that case, this function returns the statistical inefficiency as the
        size of the time series data array.

    Args:
        x (array_like, 1d): time series data.
        y (array_like, 1d, optional): time series data.
            If it is passed to this function, the cross-correlation of
            timeseries x and y will be estimated instead of the
            auto-correlation of timeseries x. (default: None)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of correlation
            function to compute. The algorithm terminates after computing the
            correlation time out to minimum_correlation_time when the
            correlation function first goes negative. (default: None)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`1 + 2\tau`, where :math:`\tau` denotes the
            integrated auto-correlation time).

    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise CRError("x is not an array of one-dimension.")

    # Get the length of the timeseries
    x_size = x.size

    if x_size < 2:
        raise CRSampleSizeError(
            f"{x_size} input data points are not sufficient to be used by "
            "this method."
        )

    # minimum amount of correlation function to compute
    if not isinstance(minimum_correlation_time, int):
        if minimum_correlation_time is None:
            minimum_correlation_time = 3
        else:
            raise CRError("minimum_correlation_time must be an `int`.")
    elif minimum_correlation_time < 1:
        raise CRError("minimum_correlation_time must be a positive `int`.")

    fft = fft and x_size > 30

    if y is None or y is x:
        # Special case if timeseries is constant.
        _std = np.std(x)

        if not np.isfinite(_std):
            raise CRError(
                "there is at least one value in the input array which is "
                "non-finite or not-number."
            )

        if isclose(_std, 0, abs_tol=_DEFAULT_ABS_TOL):
            return x_size

        del _std

        # Calculate the discrete-time normalized fluctuation
        # auto correlation function
        _corr = auto_correlate(x, fft=fft)[1:]
    else:
        # Calculate the discrete-time normalized fluctuation
        # cross correlation function
        _corr = cross_correlate(x, y, fft=fft)[1:]

    _time = np.arange(1.0, 0.0, -1.0 / float(x_size))[1:]

    end_ind = min(_corr.size, _time.size)

    # slice a numpy array, the memory is shared
    # between the slice and the original
    corr = _corr[:end_ind]
    time = _time[:end_ind]

    mct = 1.0 - min(minimum_correlation_time, x_size) / float(x_size)

    try:
        ind = np.where((corr <= 0) & (time < mct))[0][0]
    except IndexError:
        ind = end_ind

    # Compute the integrated auto-correlation time
    _tau_eq = corr[:ind] * time[:ind]

    tau_eq = np.sum(_tau_eq)

    # Compute the statistical inefficiency
    si = 1.0 + 2.0 * float(tau_eq)

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)


# Must be corrected in future
# .. [vehtari2019] Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
# .. [mcstaness] https://mc-stan.org/docs/2_22/reference-manual/effective-sample-size-section.html
# .. [gelman2014] Gelman et al. BDA (2014) Formula 11.8


def geyer_r_statistical_inefficiency(
    x: Union[np.ndarray, list[float]],
    y: Union[np.ndarray, list[float], None] = None,
    *,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,  # unused (API compatibility)
) -> float:
    r"""Compute the statistical inefficiency.

    Compute the statistical inefficiency using the Geyer's [geyer1992]_,
    [geyer2011]_ initial monotone sequence criterion.

    Note:
        The behavior is updated. Suppose the time series data is an array of
        (constant) numbers with standard deviation close to zero within
        `abs_tol=1e-18`, where `abs(a) <= max(1e-9 * abs(a), abs_tol)`.
        In that case, this function returns the statistical inefficiency as the
        size of the time series data array.

    Note:
        The effective sample size is computed by:

        .. math::

            \hat{N}_{eff} &= \frac{N}{si} \\
            si &= -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}

        where :math:`N` is the number of data points.
        :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`, where
        :math:`\hat{\rho}_t'` is the estimated auto-correlation at lag
        :math:`t'`, and :math:`m` is the last integer for which
        :math:`\hat{P}_{t'}` is still positive (largest :math:`m` such that
        :math:`\hat{P}_{t'} > 0,~t'=1,\cdots,m`). The initial monotone
        sequence is obtained by further reducing :math:`\hat{P}_{t'}` to the
        minimum of the preceding ones so that the estimated sequence is
        monotone.

        The current implementation is similar to Stan [mcstan]_, which
        uses Geyer's initial monotone sequence criterion (Geyer, 1992
        [geyer1992]_; Geyer, 2011 [geyer2011]_).

    Args:
        x (array_like, 1d): time series data. Using this method, statistical
            inefficiency can not be estimated with less than four data points.
        y (array_like, 1d, optional): time series data. If it is passed to this
            function, the cross-correlation of timeseries x and y will be
            estimated instead of the auto-correlation of timeseries x.
            (default: None)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`si = -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}`, where
            :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`)

    Note:
        minimum_correlation_time is accepted for API compatibility but is not
        used by this method.
    """
    x = np.asarray(x)

    # Check inputs
    if x.ndim != 1:
        raise CRError("x is not an array of one-dimension.")

    x_size = x.size

    if x_size < 4:
        raise CRSampleSizeError(
            f"{x_size} input data points are not sufficient to be used by "
            "this method."
        )

    fft = fft and x_size > 30

    if y is None or y is x:
        # Special case if timeseries is constant.
        _std = np.std(x)

        if not np.isfinite(_std):
            raise CRError(
                "there is at least one value in the input "
                "array which is non-finite or not-number."
            )

        if isclose(_std, 0, abs_tol=_DEFAULT_ABS_TOL):
            return x_size

        del _std

        # Calculate the discrete-time normalized fluctuation
        # auto correlation function
        corr = auto_correlate(x, fft=fft)
    else:
        # Calculate the discrete-time normalized fluctuation
        # cross correlation function
        corr = cross_correlate(x, y, fft=fft)

    rho_hat = corr - 1.0 / (x_size - 1.0)
    rho_hat[0] = 1.0

    rho_hat_s = np.zeros([x_size], dtype=np.float64)
    rho_hat_s[0:2] = rho_hat[0:2]

    # Convert estimators into Geyer's initial positive sequence. Loop only
    # until x_size - 4 to leave the last pair of auto-correlations as a bias term
    # that reduces variance in the case of antithetical chain.

    # The difficulty is that for large values of s the sample correlation is
    # too noisy. Instead we compute a partial sum, starting from lag 0 and
    # continuing until the sum of autocorrelation estimates for two
    # successive lags rho_hat_even + rho_hat_odd is negative. We use this
    # positive partial sum as an estimate of `\sum_{s=1}^{\inf}{\rho_s}`.
    # Putting this all together yields the estimate for effective sample
    # size. where the estimated autocorrelations rho_hat_s are computed and
    # max_s is the first odd positive integer for which
    # rho_hat_even + rho_hat_odd is negative.

    _sum = rho_hat[0] + rho_hat[1]
    s = 1
    while s < (x_size - 4) and _sum > 0.0:
        _sum = rho_hat[s + 1] + rho_hat[s + 2]
        if _sum >= 0.0:
            rho_hat_s[s + 1] = rho_hat[s + 1]
            rho_hat_s[s + 2] = rho_hat[s + 2]
        s += 2

    # this is used in the improved estimate, which reduces variance in
    # antithetic case: see tau_hat below
    if s > 1:
        if rho_hat[s - 2] > 0.0:
            rho_hat_s[s + 1] = rho_hat[s - 2]

    max_s = s

    # Convert Geyer's initial positive sequence into an initial monotone
    # sequence
    for s in range(1, max_s - 2, 2):
        _sum = rho_hat_s[s - 1] + rho_hat_s[s]
        if (rho_hat_s[s + 1] + rho_hat_s[s + 2]) > _sum:
            rho_hat_s[s + 1] = _sum / 2.0
            rho_hat_s[s + 2] = rho_hat_s[s + 1]

    # Geyer's truncated estimator for the asymptotic variance. Improved
    # estimate reduces variance in antithetic case
    si = -1.0 + 2.0 * rho_hat_s[:max_s].sum() + rho_hat_s[max_s + 1]

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)


def geyer_split_r_statistical_inefficiency(
    x: Union[np.ndarray, list[float]],
    y: Union[np.ndarray, list[float], None] = None,
    *,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,  # unused (API compatibility)
) -> float:
    r"""Compute the statistical inefficiency.

    Compute the statistical inefficiency using the split-r method of
    Geyer's [geyer1992]_, [geyer2011]_ initial monotone sequence criterion.

    Note:
        The effective sample size is computed by:

        .. math::

            \hat{N}_{eff} &= \frac{N}{si} \\
            si &= -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}

        where :math:`N` is the number of data points.
        :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`, where
        :math:`\hat{\rho}_t'` is the estimated auto-correlation at lag
        :math:`t'`, and :math:`m` is the last integer for which
        :math:`\hat{P}_{t'}` is still positive (largest :math:`m` such that
        :math:`\hat{P}_{t'} > 0,~t'=1,\cdots,m`). The initial monotone
        sequence is obtained by further reducing :math:`\hat{P}_{t'}` to the
        minimum of the preceding ones so that the estimated sequence is
        monotone.

        The current implementation is similar to Stan [mcstan]_, which
        uses Geyer's initial monotone sequence criterion (Geyer, 1992
        [geyer1992]_; Geyer, 2011 [geyer2011]_).

    Args:
        x (array_like, 1d): time series data.
            Using this method, statistical inefficiency can not be estimated
            with less than eight data points.
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`si = -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}`, where
            :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`)

    Note:
        minimum_correlation_time is accepted for API compatibility but is not
        used by this method.
    """
    if y is not None:
        raise CRError(
            "The split-r method splits the x time-series data and does not use y."
        )

    x = np.asarray(x)
    x_size = x.size
    if x_size < 8:
        raise CRSampleSizeError(
            f"{x_size} input data points are not sufficient to be used by "
            "this method."
        )
    x_size //= 2
    return geyer_r_statistical_inefficiency(x[:x_size], x[x_size:2 * x_size], fft=fft)


def geyer_split_statistical_inefficiency(
    x: Union[np.ndarray, list[float]],
    y: Union[np.ndarray, list[float], None] = None,
    *,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,  # unused (API compatibility)
) -> float:
    r"""Compute the statistical inefficiency.

    Computes the effective sample size. The value returned is the minimum of
    effective sample size and the data size times log10(data size).

    Note:
        Note that the effective sample size can not be estimated with less than
        four samples.

    Note:
        The behavior is updated. Suppose the time series data is an array of
        (constant) numbers with standard deviation close to zero within
        `abs_tol=1e-18`, where `abs(a) <= max(1e-9 * abs(a), abs_tol)`.
        In that case, this function returns the statistical inefficiency as the
        size of the time series data array.

    Args:
        x (array_like, 1d): time series data.
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency

    Note:
        minimum_correlation_time is accepted for API compatibility but is not
        used by this method.
    """
    if y is not None:
        raise CRError(
            "The split-r method splits the x time-series data and does not use y."
        )

    x = np.asarray(x)

    # Check inputs
    if x.ndim != 1:
        raise CRError("x is not an array of one-dimension.")

    x_size = x.size
    if x_size < 8:
        raise CRSampleSizeError(
            f"{x_size} input data points are not sufficient to be used by "
            "this method."
        )

    # Special case if timeseries is constant.
    _std = np.std(x)

    if not np.isfinite(_std):
        raise CRError(
            "there is at least one value in the input array which is "
            "non-finite or not-number."
        )

    if isclose(_std, 0, abs_tol=_DEFAULT_ABS_TOL):
        return x_size

    del _std

    x_size //= 2

    fft = fft and x_size > 30

    acov_1 = auto_covariance(x[:x_size], fft=fft)
    acov_2 = auto_covariance(x[x_size:2 * x_size], fft=fft)

    chain_mean_1 = np.mean(x[:x_size])
    chain_mean_2 = np.mean(x[x_size:2 * x_size])

    n_n_1 = float(x_size) / (x_size - 1.0)
    n_1_n = (x_size - 1.0) / float(x_size)

    chain_var_1 = acov_1[0] * n_n_1
    chain_var_2 = acov_2[0] * n_n_1

    mean_var = (chain_var_1 + chain_var_2) / 2.0

    var_plus = mean_var * n_1_n
    var_plus += np.var([chain_mean_1, chain_mean_2])

    var_plus_inv = 1.0 / var_plus

    rho_hat_s = np.zeros([x_size], dtype=np.float64)

    acov_s_1 = acov_1[1]
    acov_s_2 = acov_2[1]

    rho_hat_even = 1.0
    rho_hat_s[0] = rho_hat_even

    rho_hat_odd = 1.0 - (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv
    rho_hat_s[1] = rho_hat_odd

    # Convert raw auto-covariance estimators into Geyer's initial positive
    # sequence. Loop only until x_size - 4 to leave the last pair of
    # auto-correlations as a bias term that reduces variance in the case of
    # antithetical chain.

    _sum = rho_hat_even + rho_hat_odd

    s = 1
    while s < (x_size - 4) and _sum > 0.0:
        acov_s_1 = acov_1[s + 1]
        acov_s_2 = acov_2[s + 1]

        rho_hat_even = 1.0 - (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv

        acov_s_1 = acov_1[s + 2]
        acov_s_2 = acov_2[s + 2]

        rho_hat_odd = 1.0 - (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv

        _sum = rho_hat_even + rho_hat_odd

        if _sum >= 0.0:
            rho_hat_s[s + 1] = rho_hat_even
            rho_hat_s[s + 2] = rho_hat_odd

        s += 2

    max_s = s

    # this is used in the improved estimate, which reduces variance in
    # antithetic case: see tau_hat below
    if rho_hat_even > 0.0:
        rho_hat_s[max_s + 1] = rho_hat_even

    # Convert Geyer's initial positive sequence into an initial monotone
    # sequence
    for s in range(1, max_s - 2, 2):
        _sum = rho_hat_s[s - 1] + rho_hat_s[s]
        if (rho_hat_s[s + 1] + rho_hat_s[s + 2]) > _sum:
            rho_hat_s[s + 1] = _sum / 2.0
            rho_hat_s[s + 2] = rho_hat_s[s + 1]

    x_size *= 2

    # Geyer's truncated estimator for the asymptotic variance. Improved
    # estimate reduces variance in antithetic case
    si = -1.0 + 2.0 * rho_hat_s[:max_s].sum() + rho_hat_s[max_s + 1]

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)


si_methods = {
    "statistical_inefficiency": statistical_inefficiency,
    "geyer_r_statistical_inefficiency": geyer_r_statistical_inefficiency,
    "geyer_split_r_statistical_inefficiency": geyer_split_r_statistical_inefficiency,
    "geyer_split_statistical_inefficiency": geyer_split_statistical_inefficiency,
}


def integrated_auto_correlation_time(
    x: Union[np.ndarray, list[float]],
    y: Union[np.ndarray, list[float], None] = None,
    *,
    si: Union[str, float, int, None] = _DEFAULT_SI,
    fft: bool = _DEFAULT_FFT,
    minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,
) -> float:
    r"""Estimate the integrated auto-correlation time.

    The statistical inefficiency :math:`si` of the observable :math:`x`
    of a time series :math:`\left \{X\right \}_{t=0}^n` is formally defined as,
    :math:`si \equiv 1 + 2\tau`, where :math:`\tau` denotes the integrated
    auto-correlation time.

    Args:
        x (array_like, 1d): time series data.
        y (array_like, 1d, optional): time series data. (default: None)
            If it is passed to this function, the cross-correlation of
            timeseries x and y will be estimated instead of the
            auto-correlation of timeseries x.
        si (float, or str, optional): estimated statistical inefficiency, or a
            method of computing the statistical inefficiency. (default: None)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: True)
        minimum_correlation_time (int, optional): minimum amount of correlation
            function to compute. The algorithm terminates after computing the
            correlation time out to minimum_correlation_time when the
            correlation function first goes negative. (default: None)

    Returns:
        float: integrated auto-correlation time.
            estimated :math:`\tau` (the integrated auto-correlation time)

    """
    if si is None:
        # Compute the statistical inefficiency
        si = statistical_inefficiency(
            x, y=y, fft=fft, minimum_correlation_time=minimum_correlation_time
        )

    elif isinstance(si, str):
        if si not in si_methods:
            raise CRError(
                f"method {si} not found. Valid statistical inefficiency (si) "
                "methods are:\n\t- " + "\n\t- ".join(si_methods)
            )

        si_func = si_methods[si]

        # Compute the statistical inefficiency
        si = si_func(x, y=y, fft=fft, minimum_correlation_time=minimum_correlation_time)

    elif si < 1.0:
        raise CRError(
            "statistical inefficiency (si) must be greater than or equal one."
        )

    # Compute the integrated auto-correlation time
    tau = (si - 1.0) / 2.0

    return tau

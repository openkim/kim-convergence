"""Statistical inefficiency module."""

import numpy as np

from .err import CVGError
from .stats import auto_covariance, auto_correlate, cross_correlate

__all__ = [
    'statistical_inefficiency',
    'r_statistical_inefficiency',
    'split_r_statistical_inefficiency',
    'split_statistical_inefficiency',
    'integrated_auto_correlation_time',
    'si_methods',
]


def statistical_inefficiency(x, y=None, *, fft=False, mct=None):
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

    Args:
        x (array_like, 1d): time series data.
        y (array_like, 1d, optional): time series data.
            If it is passed to this function, the cross-correlation of
            timeseries x and y will be estimated instead of the
            auto-correlation of timeseries x. (default: None)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: False)
        mct (int, optional): minimum amount of correlation function to compute.
            The algorithm terminates after computing the correlation time out
            to mct when the correlation function first goes negative.
            (default: None)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`1 + 2\tau`, where :math:`\tau` denotes the
            integrated auto-correlation time).

    """
    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    # Get the length of the timeseries
    n = x.size

    if n < 2:
        msg = '{} number of input data points is not '.format(n)
        msg += 'sufficient to be used by this method.'
        raise CVGError(msg)

    # minimum amount of correlation function to compute
    if not isinstance(mct, int):
        if mct is None:
            mct = 5
        else:
            msg = 'mct must be an `int`.'
            raise CVGError(msg)
    elif mct < 1:
        msg = 'mct must be a positive `int`.'
        raise CVGError(msg)

    if y is None or y is x:
        # Special case if timeseries is constant.
        _std = np.std(x)
        if np.isclose(_std, 0, atol=1e-08):
            return 1.0
        elif not np.isfinite(_std):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)
        del _std

        # Calculate the discrete-time normalized fluctuation
        # auto correlation function
        corr = auto_correlate(x, fft=fft)[1:]
    else:
        # Calculate the discrete-time normalized fluctuation
        # cross correlation function
        corr = cross_correlate(x, y, fft=fft)[1:]

    t = np.arange(1., 0., -1.0 / float(n))[1:]

    mct = 1. - min(mct, n) / float(n)

    try:
        ind = np.where((corr <= 0) & (t < mct))[0][0]
    except IndexError:
        ind = n

    # Compute the integrated auto-correlation time
    tau_eq = corr[:ind] * t[:ind]
    tau_eq = np.sum(tau_eq)

    # Compute the statistical inefficiency
    si = 1.0 + 2.0 * tau_eq

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)

# .. [11] Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
# .. [12] https://mc-stan.org/docs/2_22/reference-manual/effective-sample-size-section.html
# .. [13] Gelman et al. BDA (2014) Formula 11.8


def r_statistical_inefficiency(x, y=None, *, fft=False, mct=None):
    r"""Compute the statistical inefficiency.

    Compute the statistical inefficiency using the Geyer’s [8]_, [9]_ initial
    monotone sequence criterion.

    Args:
        x (array_like, 1d): time series data. Using this method, statistical
            inefficiency can not be estimated with less than four data points.
        y (array_like, 1d, optional): time series data. If it is passed to this
            function, the cross-correlation of timeseries x and y will be
            estimated instead of the auto-correlation of timeseries x.
            (default: None)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`si = -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}`, where
            :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`)

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

        The current implementation is similar to Stan [10]_, which
        uses Geyer's initial monotone sequence criterion (Geyer, 1992 [8]_;
        Geyer, 2011 [9]_).

    References:
        .. [8] Geyer, Charles J. (1992). "Practical Markov Chain Monte Carlo."
               Statistical Science, 473–83.
        .. [9] Geyer, Charles J. (2011). "Introduction to Markov Chain Monte
               Carlo." In Handbook of Markov Chain Monte Carlo, edited by
               Steve Brooks, Andrew Gelman, Galin L. Jones, and Xiao-Li Meng,
               3–48. Chapman; Hall/CRC.
        .. [10] https://mc-stan.org/

    """
    x = np.array(x, copy=False)

    # Check inputs
    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    n = x.size

    if n < 4:
        msg = '{} number of input data points is not '.format(n)
        msg += 'sufficient to be used by this method.'
        raise CVGError(msg)

    if y is None or y is x:
        # Special case if timeseries is constant.
        _std = np.std(x)
        if np.isclose(_std, 0, atol=1e-08):
            return 1.0
        elif not np.isfinite(_std):
            msg = 'there is at least one value in the input array which is '
            msg += 'non-finite or not-number.'
            raise CVGError(msg)
        del _std

        # Calculate the discrete-time normalized fluctuation
        # auto correlation function
        corr = auto_correlate(x, fft=fft)
    else:
        # Calculate the discrete-time normalized fluctuation
        # cross correlation function
        corr = cross_correlate(x, y, fft=fft)

    rho_hat = corr - 1.0 / (n - 1.0)
    rho_hat[0] = 1.0

    rho_hat_s = np.zeros([n], dtype=np.float64)
    rho_hat_s[0:2] = rho_hat[0:2]

    # Convert estimators into Geyer's initial positive sequence. Loop only
    # until n - 4 to leave the last pair of auto-correlations as a bias term
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
    while s < (n - 4) and _sum > 0.0:
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
            rho_hat_s[s + 1] = _sum / 2.
            rho_hat_s[s + 2] = rho_hat_s[s + 1]

    # Geyer's truncated estimator for the asymptotic variance. Improved
    # estimate reduces variance in antithetic case
    si = -1.0 + 2.0 * rho_hat_s[:max_s].sum() + rho_hat_s[max_s + 1]

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)


def split_r_statistical_inefficiency(x, y=None, *, fft=False, mct=None):
    r"""Compute the statistical inefficiency.

    Compute the statistical inefficiency using the split-r method of
    Geyer’s [8]_, [9]_ initial monotone sequence criterion.

    Args:
        x (array_like, 1d): time series data.
            Using this method, statistical inefficiency can not be estimated
            with less than eight data points.
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency
            (equal to :math:`si = -1 + 2 \sum_{t'=0}^m \hat{P}_{t'}`, where
            :math:`\hat{P}_{t'} = \hat{\rho}_{2t'} + \hat{\rho}_{2t'+1}`)

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

        The current implementation is similar to Stan [10]_, which
        uses Geyer's initial monotone sequence criterion (Geyer, 1992 [8]_;
        Geyer, 2011 [9]_).

    """
    if y is not None:
        msg = 'The split-r method, splits the x time-series data '
        msg += 'and do not use y.'
        raise CVGError(msg)

    x = np.array(x, copy=False)
    n = x.size
    if n < 8:
        msg = '{} number of input data points is not '.format(n)
        msg += 'sufficient to be used by this method.'
        raise CVGError(msg)
    n //= 2
    return r_statistical_inefficiency(x[:n], x[n:2 * n], fft=fft)


def split_statistical_inefficiency(x, y=None, *, fft=False, mct=None):
    r"""Compute the statistical inefficiency.

    Computes the effective sample size. The value returned is the minimum of
    effective sample size and the data size times log10(data size).

    Note that the effective sample size can not be estimated with less than
    four samples.

    Args:
        x (array_like, 1d): time series data.
        fft (bool, optional): if ``True``, use FFT convolution. FFT should be
            preferred for long time series. (default: False)

    Returns:
        float: estimated statistical inefficiency.
            :math:`si >= 1` is the estimated statistical inefficiency

    """
    if y is not None:
        msg = 'the split-r method, splits the x time-series data '
        msg += 'and do not use y.'
        raise CVGError(msg)

    x = np.array(x, copy=False)

    # Check inputs
    if x.ndim != 1:
        msg = 'x is not an array of one-dimension.'
        raise CVGError(msg)

    n = x.size
    if n < 8:
        msg = '{} number of input data points is not sufficient '.format(n)
        msg += 'to be used by this method.'
        raise CVGError(msg)

    n //= 2

    # Special case if timeseries is constant.
    _std = np.std(x)
    if np.isclose(_std, 0, atol=1e-08):
        return 1.0
    elif not np.isfinite(_std):
        msg = 'there is at least one value in the input array which is '
        msg += 'non-finite or not-number.'
        raise CVGError(msg)
    del _std

    acov_1 = auto_covariance(x[:n], fft=True)
    acov_2 = auto_covariance(x[n:2 * n], fft=True)

    chain_mean_1 = np.mean(x[:n])
    chain_mean_2 = np.mean(x[n:2 * n])

    n_n_1 = float(n) / (n - 1.0)
    n_1_n = (n - 1.0) / float(n)

    chain_var_1 = acov_1[0] * n_n_1
    chain_var_2 = acov_2[0] * n_n_1

    mean_var = (chain_var_1 + chain_var_2) / 2.0

    var_plus = mean_var * n_1_n
    var_plus += np.var([chain_mean_1, chain_mean_2])

    var_plus_inv = 1.0 / var_plus

    rho_hat_s = np.zeros([n], dtype=np.float64)

    acov_s_1 = acov_1[1]
    acov_s_2 = acov_2[1]

    rho_hat_even = 1.0
    rho_hat_s[0] = rho_hat_even

    rho_hat_odd = 1.0 - \
        (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv
    rho_hat_s[1] = rho_hat_odd

    # Convert raw auto-covariance estimators into Geyer's initial positive
    # sequence. Loop only until n - 4 to leave the last pair of
    # auto-correlations as a bias term that reduces variance in the case of
    # antithetical chain.

    _sum = rho_hat_even + rho_hat_odd

    s = 1
    while s < (n - 4) and _sum > 0.0:
        acov_s_1 = acov_1[s + 1]
        acov_s_2 = acov_2[s + 1]

        rho_hat_even = 1.0 - \
            (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv

        acov_s_1 = acov_1[s + 2]
        acov_s_2 = acov_2[s + 2]

        rho_hat_odd = 1.0 - \
            (mean_var - (acov_s_1 + acov_s_2) / 2.0) * var_plus_inv

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
            rho_hat_s[s + 1] = _sum / 2.
            rho_hat_s[s + 2] = rho_hat_s[s + 1]

    n *= 2

    # Geyer's truncated estimator for the asymptotic variance. Improved
    # estimate reduces variance in antithetic case
    si = -1.0 + 2.0 * rho_hat_s[:max_s].sum() + rho_hat_s[max_s + 1]

    # Statistical inefficiency (si) must be greater than or equal one.
    return max(1.0, si)


si_methods = {
    'statistical_inefficiency': statistical_inefficiency,
    'r_statistical_inefficiency': r_statistical_inefficiency,
    'split_r_statistical_inefficiency': split_r_statistical_inefficiency,
    'split_statistical_inefficiency': split_statistical_inefficiency,
}


def integrated_auto_correlation_time(x, y=None, *, si=None, fft=False, mct=None):
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
            preferred for long time series. (default: {False})
        mct (int, optional): minimum amount of correlation function to compute.
            (default: None) The algorithm terminates after computing the
            correlation time out to mct when the correlation function first
            goes negative.

    Returns:
        float: integrated auto-correlation time.
            estimated :math:`\tau` (the integrated auto-correlation time)

    """
    if si is None:
        try:
            # Compute the statistical inefficiency
            si = statistical_inefficiency(x, y=y, fft=fft, mct=mct)
        except:
            si = 1.0
    elif isinstance(si, str):
        if si not in si_methods:
            msg = 'method {} not found. Valid statistical '.format(si)
            msg += 'inefficiency (si) methods are:\n\t- '
            msg += '{}'.format('\n\t- '.join(si_methods))
            raise CVGError(msg)

        si_func = si_methods[si]
        try:
            # Compute the statistical inefficiency
            si = si_func(x, y=y, fft=fft, mct=mct)
        except:
            si = 1.0
    elif si < 1.0:
        msg = 'statistical inefficiency (si) must be greater than '
        msg += 'or equal one.'
        raise CVGError(msg)

    # Compute the integrated auto-correlation time
    tau = (si - 1.0) / 2.0

    return tau

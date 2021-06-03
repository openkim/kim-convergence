"""Upper Confidence Limit (UCL) module.

Upper Confidence Limit (UCL): The upper boundary (or limit) of a confidence
interval of a parameter of interest such as the population mean.

A confidence interval is how much uncertainty there is with any particular
statistic.
Confidence limits for the mean are interval estimates. Interval estimates are
often desirable because instead of a single estimate for the mean, a confidence
interval generates a lower and upper limit. It indicates how much uncertainty
there is in our estimation of the true mean. The narrower the gap, the more
precise our estimate is. We use a confidence level to express confidence limits.
Choosing the confidence level is somewhat arbitrary, but 90 %, 95 %, and 99 %
intervals are standard, and 95 % is the most commonly used.

Note:
    One should note that a 95 % confidence interval does not mean a 95 %
    probability of containing the true mean. The interval computed from a
    sample either has the true mean, or it does not. The confidence level is
    simply the proportion of samples of a given size that may be expected to
    contain the true mean. For a 95 % confidence interval, if many samples are
    collected and the confidence interval computed, in the long run, about 95 %
    of these intervals would contain the true mean.

References:
    .. [18] https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm

"""

from .ucl_base import UCLBase

from .spectral import \
    HeidelbergerWelch, \
    heidelberger_welch_ucl, \
    heidelberger_welch_ci, \
    heidelberger_welch_relative_half_width_estimate

from .uncorrelated_samples import \
    UncorrelatedSamples, \
    uncorrelated_samples_ucl, \
    uncorrelated_samples_ci, \
    uncorrelated_samples_relative_half_width_estimate

from .n_skart import \
    N_SKART, \
    n_skart_ucl, \
    n_skart_ci, \
    n_skart_relative_half_width_estimate

from .mser_m import \
    MSER_m, \
    mser_m_ucl, \
    mser_m_ci, \
    mser_m_relative_half_width_estimate, \
    mser_m

from .mser_m_y import \
    MSER_m_y, \
    mser_m_y_ucl, \
    mser_m_y_ci, \
    mser_m_y_relative_half_width_estimate

ucl_methods = {
    'heidel_welch': HeidelbergerWelch,
    'uncorrelated_sample': UncorrelatedSamples,
    'n_skart': N_SKART,
    'mser_m': MSER_m,
    'mser_m_y': MSER_m_y,
}


__all__ = [
    'UCLBase',
    'HeidelbergerWelch',
    'heidelberger_welch_ucl',
    'heidelberger_welch_ci',
    'heidelberger_welch_relative_half_width_estimate',
    'UncorrelatedSamples',
    'uncorrelated_samples_ucl',
    'uncorrelated_samples_ci',
    'uncorrelated_samples_relative_half_width_estimate',
    'N_SKART',
    'n_skart_ucl',
    'n_skart_ci',
    'n_skart_relative_half_width_estimate',
    'MSER_m',
    'mser_m_ucl',
    'mser_m_ci',
    'mser_m_relative_half_width_estimate',
    'mser_m',
    'MSER_m_y',
    'mser_m_y_ucl',
    'mser_m_y_ci',
    'mser_m_y_relative_half_width_estimate',
    'ucl_methods',
]

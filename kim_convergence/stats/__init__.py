r"""stats module."""

from .beta_dist import beta, betacf, betai, betai_cdf_ccdf, betai_cdf
from .normal_dist import s_normal_inv_cdf, normal_inv_cdf, normal_interval
from .normal_test import t_test, chi_square_test
from .nonnormal_test import (
    ContinuousDistributions,
    ContinuousDistributionsNumberOfRequiredArguments,
    ContinuousDistributionsArgumentRequirement,
    check_population_cdf_args,
    get_distribution_stats,
    levene_test,
    ks_test,
    kruskal_test,
    wilcoxon_test,
)
from .randomness_test import randomness_test
from .t_dist import t_cdf_ccdf, t_cdf, t_inv_cdf, t_interval
from .tools import (
    get_fft_optimal_size,
    auto_covariance,
    auto_correlate,
    cross_covariance,
    cross_correlate,
    modified_periodogram,
    periodogram,
    int_power,
    moment,
    skew,
)
from .zero_rc_bounds import ZERO_RC_BOUNDS
from .zero_rc import ZERO_RC


__all__ = [
    # beta_dist
    "beta",
    "betacf",
    "betai",
    "betai_cdf_ccdf",
    "betai_cdf",
    # normal_dist
    "s_normal_inv_cdf",
    "normal_inv_cdf",
    "normal_interval",
    # normal_test
    "chi_square_test",
    "t_test",
    # nonnormal_test
    "ContinuousDistributions",
    "ContinuousDistributionsNumberOfRequiredArguments",
    "ContinuousDistributionsArgumentRequirement",
    "check_population_cdf_args",
    "get_distribution_stats",
    "levene_test",
    "kruskal_test",
    "ks_test",
    "wilcoxon_test",
    # randomness_test
    "randomness_test",
    # t_dist
    "t_cdf_ccdf",
    "t_cdf",
    "t_inv_cdf",
    "t_interval",
    # tools
    "get_fft_optimal_size",
    "auto_covariance",
    "auto_correlate",
    "cross_covariance",
    "cross_correlate",
    "modified_periodogram",
    "periodogram",
    "int_power",
    "moment",
    "skew",
    # zero_rc_bounds
    "ZERO_RC_BOUNDS",
    # zero_rc
    "ZERO_RC",
]

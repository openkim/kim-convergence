r"""Outlier module."""

import numpy as np
from typing import Optional, Union

from .err import CRError


__all__ = ["outlier_methods", "outlier_test"]


outlier_methods = (
    "iqr",
    "boxplot",
    "extreme_iqr",
    "extreme_boxplot",
    "z_score",
    "standard_score",
    "modified_z_score",
)
r"""Methods to decide what are outliers in the data.

- iqr or boxplot:
  The interquartile range ('iqr') or 'boxplot' is a graphical display for
  illustrating the data's behavior in the middle and at the distributions'
  ends. The lower and upper quartiles Q1 and Q3, respectively (defined as the
  25th and 75th percentiles) and the difference (Q3 - Q1) is called the
  interquartile range.

  In the 'iqr' or 'boxplot' method, a point beyond an inner-fence on either
  side is considered an outlier. The inner-fence is the multiplication of '1.5'
  and the interquartile range (:math:`1.5 \times \text{IQR}`).

- extreme_iqr or extreme_boxplot:
  In the `extreme_iqr` method, a point beyond an outer-fence is considered an
  outlier. The outer-fence is the multiplication of 3 and the interquartile
  range (:math:`3 \times \text{IQR}`).

- z_score or standard_score:
    The 'z_score', or 'standard_score', describes a point compared to the
    data's mean and standard deviation. The goal is to remove the effects of
    the location and scale of the data. This approach maps the data onto a
    distribution whose mean is 0 and whose standard deviation is 1. Anything
    that is too far from zero (the threshold is 3 or -3) is considered an
    outlier.

- modified_z_score:
    'modified_z_score' is a method to improve the 'z_score' approach in small
    datasets (usually when the dataset has fewer than 12 points). It uses the
    median and MAD rather than the mean and standard deviation, which are
    robust central tendency and dispersion measures, respectively.

"""


def outlier_test(
    x: Union[np.ndarray, list[float]], outlier_method: str = "iqr"
) -> Optional[np.ndarray]:
    r"""Test to detect what are outliers in the data.

    The intuitive definition for the concept of an outlier in the data is a
    point that significantly deviates from its expected value. Therefore, given
    a time series (or a random sample from a population), a point can be
    declared an outlier if the distance to its expected value is higher than a
    predefined threshold (:math:`|x_i - E(x)| > \tau`), where :math:`x_i` is
    the observed data point, and :math:`E(x)` is its expected value.

    The methods based on this strategy are the most common approaches in the
    literature. These methods intend to detect outliers, but it is up to the
    analyst to decide if the detected points are real outliers. Thus it is
    necessary to characterize standard data points before removing any outliers
    detected by these approaches.

    Args:
        x (array_like, 1d): Time series data.
        outlier_method (str, optional): Method for outlier detection.
            (default: 'iqr')

    Returns:
        Optional[ndarray]
            Indices of outliers; None if no outliers found.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise CRError("x is not an array of one-dimension.")

    if not np.all(np.isfinite(x)):
        raise CRError(
            "there is at least one value in the input array which is "
            "non-finite or not-number."
        )

    if isinstance(outlier_method, str):
        if outlier_method not in outlier_methods:
            raise CRError(
                f"method {outlier_method} not found. Valid methods to "
                "detect outliers are:\n\t- " + "\n\t- ".join(outlier_methods)
            )
    else:
        raise CRError("Input outlier_method is not a `str`.")

    if outlier_method in ("iqr", "boxplot"):
        lower_quartile, upper_quartile = np.quantile(x, [0.25, 0.75])
        difference = upper_quartile - lower_quartile
        lower_inner_fence = lower_quartile - (difference * 1.5)
        upper_inner_fence = upper_quartile + (difference * 1.5)
        outliers_indices = np.where((x < lower_inner_fence) | (x > upper_inner_fence))
    elif outlier_method in ("extreme_iqr", "extreme_boxplot"):
        lower_quartile, upper_quartile = np.quantile(x, [0.25, 0.75])
        difference = upper_quartile - lower_quartile
        lower_outer_fence = lower_quartile - (difference * 3.0)
        upper_outer_fence = upper_quartile + (difference * 3.0)
        outliers_indices = np.where((x < lower_outer_fence) | (x > upper_outer_fence))
    elif outlier_method in ("z_score", "standard_score"):
        x_mean = np.mean(x)
        x_std = np.std(x)
        if np.isclose(x_std, 0.0):
            # All values identical -> no outlier can be declared
            return None
        z_score = (x - x_mean) / x_std
        outliers_indices = np.where(np.abs(z_score) > 3)
    elif outlier_method == "modified_z_score":
        x_median = np.median(x)
        x_median_absolute_deviation = np.median(np.abs(x - x_median))
        if np.isclose(x_median_absolute_deviation, 0.0):
            # All values identical -> no outlier can be declared
            return None
        modified_z_score = 0.6745 * (x - x_median) / x_median_absolute_deviation
        outliers_indices = np.where(np.abs(modified_z_score) > 3.5)

    if np.size(outliers_indices):
        return outliers_indices[0]

    return None

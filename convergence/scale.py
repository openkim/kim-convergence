"""Scale module to standardize a dataset."""

from math import isclose, fabs, log10
import numpy as np

from .err import CVGError
from ._default import _DEFAULT_ABS_TOL


__all__ = [
    'MinMaxScale',
    'minmax_scale',
    'TranslateScale',
    'translate_scale',
    'StandardScale',
    'standard_scale',
    'RobustScale',
    'robust_scale',
    'MaxAbsScale',
    'maxabs_scale',
    'scale_methods',
]


class MinMaxScale():
    r"""Standardize/Transform a dataset by scaling it to a given range.

    This estimator scales and translates a dataset such that it is in the given
    range, e.g. between zero and one.

    The transformation is given by::

    .. math::

        \nonumber x\_std = \frac{x - np.min(x)}{np.max(x) - np.min(x)} \\
        \nonumber scaled\_x = x\_std * (max - min) + min

    where min, max = feature_range.

    Args:
        feature_range (tuple, optional): tuple (min, max). (default: (0, 1))
            Desired range of transformed data.

    Examples
    --------
    >>> from convergence import MinMaxScale, minmax_scale
    >>> data = [-1., 3.]
    >>> mms = MinMaxScale()
    >>> scaled_x = mms.scale(data)
    >>> print(scaled_x)
    [0. 1.]
    >>> x = mms.inverse(scaled_x)
    >>> print(x)
    [-1.  3.]
    >>> data = [-1., 3., 100.]
    >>> scaled_x = minmax_scale(data)
    >>> print(x)
    [0. 0.03960396 1.]
    >>> x = mms.inverse(scaled_x)
    >>> print(x)
    [ -1. 3. 100.]

    """

    def __init__(self, *, feature_range=(0, 1)):
        if feature_range[0] >= feature_range[1]:
            msg = "Minimum of desired feature range must be smaller "
            msg += "than maximum. Got {}".format(str(feature_range))
            raise CVGError(msg)
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None

    def scale(self, x):
        """Standardize a dataset by scaling it to a given range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Scaled dataset to a given range

        """
        x = np.array(x, copy=False)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CVGError(msg)

        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input '
            msg += 'array which is non-finite or not-number.'
            raise CVGError(msg)

        self.data_min = np.min(x)
        self.data_max = np.max(x)
        self.data_range = self.data_max - self.data_min

        if isclose(self.data_range, 0, abs_tol=_DEFAULT_ABS_TOL):
            msg = 'the data_range of the input array is almost zero within '
            msg += '{} precision numbers.'.format(int(fabs(log10(_DEFAULT_ABS_TOL))))
            raise CVGError(msg)

        self.scale_ = \
            (self.feature_range[1] - self.feature_range[0]) / self.data_range

        self.min_ = self.feature_range[0] - self.data_min * self.scale_

        scaled_x = x * self.scale_
        scaled_x += self.min_
        return scaled_x

    def inverse(self, x):
        """Undo the scaling of dataset to its original range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Transformed data.

        """
        if self.min_ is None:
            msg = "internal data-dependent state are not set, you need "
            msg += "to scale an array before trying to inverse it."
            raise CVGError(msg)
        x = np.array(x, copy=False)
        inverse_scaled_x = x - self.min_
        inverse_scaled_x /= self.scale_
        return inverse_scaled_x


def minmax_scale(x, *, with_centering=True, with_scaling=True,
                 feature_range=(0, 1)):
    r"""Standardize/Transform a dataset by scaling it to a given range.

    This estimator scales and translates a dataset such that it is in the given
    range, e.g. between zero and one.

    The transformation is given by::

    .. math::

        \nonumber x\_std = \frac{x - np.min(x)}{np.max(x) - np.min(x)} \\
        \nonumber scaled\_x = x\_std * (max - min) + min

    where min, max = feature_range.

    Args:
        x (array_like, 1d): Time series data.
        feature_range (tuple, optional): tuple (min, max). (default: (0, 1))
            Desired range of transformed data.

    Returns:
        1darray: Scaled dataset to a given range

    """
    mms = MinMaxScale(feature_range=feature_range)
    return mms.scale(x)


class TranslateScale():
    r"""Standardize a dataset.

    Standardize a dataset by translating the data set so that :math:`x[0]=0`
    and rescaled by overall averages so that the numbers are of O(1) with a
    good spread. (default: True)

    The translate and scale of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - x_0)}{u}

    where :math:`x_0` is :math:`x[0]` or :math:`0` if `with_centering=False`,
    and `u` is the mean of the samples or :math:`1` if `with_scaling=False`.

    Args:
        with_centering (bool, optional): If True, use x minus its first
            element. (default: True)
        with_scaling (bool, optional): If True, scale the data to overall
            averages so that the numbers are of O(1) with a good spread.
            (default: True)

    Examples
    --------
    >>> from convergence import TranslateScale
    >>> data = [1., 2., 2., 2., 3.]
    >>> tsc = TranslateScale()
    >>> scaled_x = tsc.scale(data)
    >>> print(scaled_x)
    [0. 1. 1. 1. 2.]
    >>> x = tsc.inverse(scaled_x)
    >>> print(x)
    [1. 2. 2. 2. 3.]

    """

    def __init__(self, *, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_ = None
        self.scale_ = None

    def scale(self, x):
        """Standardize a dataset by scaling it to a given range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Scaled dataset to a given range

        """
        x = np.array(x, copy=False)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CVGError(msg)

        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input '
            msg += 'array which is non-finite or not-number.'
            raise CVGError(msg)

        if self.with_centering:
            self.center_ = x[0]
            # Fluctuations
            scaled_x = x - self.center_
        else:
            scaled_x = np.array(x, copy=True)

        if self.with_scaling:
            self.scale_ = np.mean(scaled_x)

            if not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
                scaled_x /= self.scale_

        return scaled_x

    def inverse(self, x):
        """Undo the scaling of dataset to its original range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Transformed data.

        """
        if self.scale_ is None and self.center_ is None:
            msg = "internal data-dependent state are not set, you need "
            msg += "to scale an array before trying to inverse it."
            raise CVGError(msg)

        if self.with_scaling and not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
            x = np.array(x, copy=False)
            inverse_scaled_x = x * self.scale_
        else:
            inverse_scaled_x = np.array(x, copy=True)

        if self.with_centering:
            inverse_scaled_x += self.center_

        return inverse_scaled_x


def translate_scale(x, *, with_centering=True, with_scaling=True):
    r"""Standardize a dataset.

    Standardize a dataset by translating the data set so that :math:`x[0]=0`
    and rescaled by overall averages so that the numbers are of O(1) with a
    good spread. (default: True)

    The translate and scale of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - x_0)}{u}

    where :math:`x_0` is :math:`x[0]` or :math:`0` if `with_centering=False`,
    and `u` is the mean of the samples or :math:`1` if `with_scaling=False`.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, use x minus its first
            element. (default: True)
        with_scaling (bool, optional): If True, scale the data to overall
            averages so that the numbers are of O(1) with a good spread.
            (default: True)

    Returns:
        1darray: Scaled dataset

    """
    tsc = TranslateScale(with_centering=with_centering,
                         with_scaling=with_scaling)
    return tsc.scale(x)


class StandardScale():
    r"""Standardize a dataset.

    Standardize a dataset by removing the mean and scaling to unit variance.
    The standard score of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - u)}{s}

    where `u` is the mean of the samples or :math:`0` if `with_centering=False`
    , and `s` is the standard deviation of the samples or :math:`1` if
    `with_scaling=False`.

    Centering and scaling happen independently.

    Args:
        with_centering (bool, optional): If True, use x minus its mean, or
            center the data before scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data to unit
            variance (or equivalently, unit standard deviation).
            (default: True)

    Notes:
        If set explicitly `with_centering=False` (only variance scaling will
        be performed on x). We use a biased estimator for the standard
        deviation.

    Examples
    --------
    >>> from convergence import StandardScale
    >>> data = [-0.5, 6]
    >>> ssc = StandardScale()
    >>> scaled_x = ssc.scale(data)
    >>> print(scaled_x)

    >>> x = ssc.inverse(scaled_x)
    >>> print(x)

    """

    def __init__(self, *, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.std_ = None
        self.mean_ = None
        self.mean_1 = None
        self.mean_2 = None

    def scale(self, x):
        """Standardize a dataset.

        Args:
            x (array_like, 1d): The data to center and scale.

        Returns:
            1darray: Scaled and/or Centered dataset.

        """
        x = np.array(x, copy=False)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CVGError(msg)

        if self.with_centering:
            self.mean_ = np.mean(x)

            # Fluctuations
            scaled_x = x - self.mean_

            self.mean_1 = np.mean(scaled_x)

            # Verify that mean_1 is 'close to zero'. If x contains very
            # large values, mean_1 can also be very large, due to a lack of
            # scaled_x is a view on the original array
            # Numerical issues were encountered when centering the data
            # and might not be solved. Dataset may contain too large values.
            # You may need to prescale your features.

            if not np.isfinite(self.mean_1):
                msg = 'there is at least one value in the input array which is '
                msg += 'non-finite or not-number.'
                raise CVGError(msg)

            if not isclose(self.mean_1, 0, abs_tol=_DEFAULT_ABS_TOL):
                scaled_x -= self.mean_1

        else:
            scaled_x = np.array(x, copy=True)

        if self.with_scaling:
            self.std_ = np.std(x)

            if not np.isfinite(self.std_):
                msg = 'there is at least one value in the input array which is '
                msg += 'non-finite or not-number.'
                raise CVGError(msg)

            if not isclose(self.std_, 0, abs_tol=_DEFAULT_ABS_TOL):
                scaled_x /= self.std_

            if self.with_centering:
                self.mean_2 = np.mean(scaled_x)

                # If mean_2 is not 'close to zero', it comes from the fact that
                # std_ is very small so that mean_2 = mean_1/std_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of np.mean(x). A solution is then
                # to subtract the mean again.
                # Numerical issues were encountered when centering the data
                # and might not be solved. Dataset may contain too large values.
                # You may need to prescale your features.

                if not isclose(self.mean_2, 0, abs_tol=_DEFAULT_ABS_TOL):
                    scaled_x -= self.mean_2

        return scaled_x

    def inverse(self, x):
        """Undo the scaling of dataset to its original range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Transformed data.

        """
        if self.mean_ is None and self.std_ is None:
            msg = "internal data-dependent state are not set, you need "
            msg += "to scale an array before trying to inverse it."
            raise CVGError(msg)

        if self.with_scaling:
            x = np.array(x, copy=False)
            if self.mean_2 is not None:
                inverse_scaled_x = x + self.mean_2
            inverse_scaled_x *= self.std_
        else:
            inverse_scaled_x = np.array(x, copy=True)

        if self.with_centering:
            if not isclose(self.mean_1, 0, abs_tol=_DEFAULT_ABS_TOL):
                inverse_scaled_x += self.mean_1
            inverse_scaled_x += self.mean_

        return inverse_scaled_x


def standard_scale(x, *, with_centering=True, with_scaling=True):
    r"""Standardize a dataset.

    Standardize a dataset by removing the mean and scaling to unit variance.
    The standard score of a sample `x` is calculated as:

    .. math::

        z = \frac{(x - u)}{s}

    where `u` is the mean of the samples or :math:`0` if `with_centering=False`
    , and `s` is the standard deviation of the samples or :math:`1` if
    `with_scaling=False`.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, use x minus its mean, or
            center the data before scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data to unit
            variance (or equivalently, unit standard deviation).
            (default: True)

    Returns:
        1darray: scaled dataset

    Notes:
        If set explicitly `with_centering=False` (only variance scaling will
        be performed on x). We use a biased estimator for the standard
        deviation.

    """
    ssc = StandardScale(with_centering=with_centering,
                        with_scaling=with_scaling)
    return ssc.scale(x)


class RobustScale():
    r"""Standardize a dataset.

    Standardize a dataset by centering to the median and component wise scale
    according to the inter-quartile range. These features are robust to
    outliers.

    This way removes the median and scales the data according to the quantile
    range. The Interquartile Range is the range between the 1st quartile
    (25th quantile) and the 3rd quartile (75th quantile).

    Centering and scaling happen independently.

    Args:
        with_centering (bool, optional): If True, center the data before
            scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data.
            (default: True)
        quantile_range (tuple, or list, optional): (q_min, q_max),
            0.0 < q_min < q_max < 100.0
            (default: (25.0, 75.0) = (1st quantile, 3rd quantile))

    Examples
    --------
    >>> from convergence import RobustScale
    >>> data = [ 4.,  1., -2.]
    >>> rsc = RobustScale()
    >>> scaled_x = rsc.scale(data)
    >>> print(scaled_x)
    [ 1.22474487  0.         -1.22474487]
    >>> x = rsc.inverse(scaled_x)
    >>> print(x)
    [ 4.  1. -2.]
    """

    def __init__(self, *,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        if not isinstance(quantile_range, tuple) or \
                not isinstance(quantile_range, list):
            msg = 'invalid quantile range: {}.'.format(str(quantile_range))
            raise CVGError(msg)

        if len(quantile_range) != 2:
            msg = 'invalid quantile range: {}.'.format(str(quantile_range))
            raise CVGError(msg)

        q_min, q_max = quantile_range
        if not 0 <= q_min <= q_max <= 100:
            msg = 'invalid quantile range: {}.'.format(str(quantile_range))
            raise CVGError(msg)

        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def scale(self, x):
        """Compute the median and quantiles to be used for scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        x = np.array(x, copy=False)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CVGError(msg)

        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input '
            msg += 'array which is non-finite or not-number.'
            raise CVGError(msg)

        if self.with_centering:
            self.center_ = np.median(x)
            scaled_x = x - self.center_
        else:
            scaled_x = np.array(x, copy=True)

        if self.with_scaling:
            quantiles = np.percentile(x, self.quantile_range)

            self.scale_ = quantiles[1] - quantiles[0]

            if not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
                scaled_x /= self.scale_

        return scaled_x

    def inverse(self, x):
        """Undo the scaling of dataset to its original range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Transformed data.

        """
        if self.center_ is None and self.scale_ is None:
            msg = "internal data-dependent state are not set, you need "
            msg += "to scale an array before trying to inverse it."
            raise CVGError(msg)

        if self.with_scaling and not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
            x = np.array(x, copy=False)
            inverse_scaled_x = x * self.scale_
        else:
            inverse_scaled_x = np.array(x, copy=True)

        if self.with_centering:
            inverse_scaled_x += self.center_

        return inverse_scaled_x


def robust_scale(x, *, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0)):
    """Standardize a dataset.

    Standardize a dataset by centering to the median and component wise scale
    according to the inter-quartile range.

    Args:
        x (array_like, 1d): The data to center and scale.
        with_centering (bool, optional): If True, center the data before
            scaling. (default: True)
        with_scaling (bool, optional): If True, scale the data.
            (default: True)
        quantile_range (tuple, or list, optional): (q_min, q_max),
            0.0 < q_min < q_max < 100.0
            (default: (25.0, 75.0) = (1st quantile, 3rd quantile))

    Returns:
        1darray: scaled dataset

    """
    rsc = RobustScale(with_centering=with_centering,
                      with_scaling=with_scaling,
                      quantile_range=quantile_range)
    return rsc.scale(x)


class MaxAbsScale():
    r"""Standardize a dataset to the [-1, 1] range.

    Standardize a dataset to the [-1, 1] range such that the maximal absolute
    value in the data set will be 1.0.

    Examples
    --------
    >>> from convergence import MaxAbsScale
    >>> data = [ 4.,  1., -9.]
    >>> mas = MaxAbsScale()
    >>> scaled_x = mas.scale(data)
    >>> print(scaled_x)
    [ 0.44444444  0.11111111 -1.        ]
    >>> x = mas.inverse(scaled_x)
    >>> print(x)
    [ 4.  1. -9.]
    """

    def __init__(self):
        self.scale_ = None

    def scale(self, x):
        """
        Online computation of max absolute value of X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Transformer instance.
        """
        x = np.array(x, copy=False)

        if x.ndim != 1:
            msg = 'x is not an array of one-dimension.'
            raise CVGError(msg)

        if not np.all(np.isfinite(x)):
            msg = 'there is at least one value in the input '
            msg += 'array which is non-finite or not-number.'
            raise CVGError(msg)

        self.scale_ = np.max(np.abs(x))

        if not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
            scaled_x = x / self.scale_
        else:
            scaled_x = np.array(x, copy=True)

        return scaled_x

    def inverse(self, x):
        """Undo the scaling of dataset to its original range.

        Args:
            x (array_like, 1d): Time series data.

        Returns:
            1darray: Transformed data.

        """
        if self.scale_ is None:
            msg = "internal data-dependent state are not set, you need "
            msg += "to scale an array before trying to inverse it."
            raise CVGError(msg)

        if not isclose(self.scale_, 0, abs_tol=_DEFAULT_ABS_TOL):
            x = np.array(x, copy=False)
            inverse_scaled_x = x * self.scale_
        else:
            inverse_scaled_x = np.array(x, copy=True)

        return inverse_scaled_x


def maxabs_scale(x, *, with_centering=True, with_scaling=True):
    """Standardize a dataset to the [-1, 1] range.

    Standardize a dataset to the [-1, 1] range such that the maximal absolute
    value in the data set will be 1.0.

    Args:
        x (array_like, 1d): The data to center and scale.

    Returns:
        1darray: scaled dataset

    """
    mas = MaxAbsScale()
    return mas.scale(x)


scale_methods = {
    'minmax_scale': minmax_scale,
    'translate_scale': translate_scale,
    'standard_scale': standard_scale,
    'robust_scale': robust_scale,
    'maxabs_scale': maxabs_scale,
}

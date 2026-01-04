r"""Heidelberger and Welch UCL module."""

from math import sqrt
import numpy as np
from numpy.linalg import pinv, norm, inv
from typing import Optional, Union

from kim_convergence._default import (
    _DEFAULT_CONFIDENCE_COEFFICIENT,
    _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_FFT,
    _DEFAULT_SCALE_METHOD,
    _DEFAULT_WITH_CENTERING,
    _DEFAULT_WITH_SCALING,
    _DEFAULT_TEST_SIZE,
    _DEFAULT_TRAIN_SIZE,
    _DEFAULT_POPULATION_STANDARD_DEVIATION,
    _DEFAULT_SI,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
    _DEFAULT_UNCORRELATED_SAMPLE_INDICES,
    _DEFAULT_SAMPLE_METHOD,
)
from kim_convergence.ucl import UCLBase
from kim_convergence import (
    batch,
    CRError,
    CRSampleSizeError,
    cr_warning,
    modified_periodogram,
    t_inv_cdf,
    train_test_split,
)


__all__ = [
    "HeidelbergerWelch",
    "heidelberger_welch_ucl",
    "heidelberger_welch_ci",
    "heidelberger_welch_relative_half_width_estimate",
]


class HeidelbergerWelch(UCLBase):
    r"""Heidelberger and Welch algorithm.

    Heidelberger and Welch (1981) [heidelberger1981]_ Object.

    Attributes:
        heidel_welch_set (bool): Flag indicating if the Heidelberger and Welch
            constants are set.
        heidel_welch_k (int) : The number of points that are used to obtain the
            polynomial fit in Heidelberger and Welch's spectral method.
        heidel_welch_n (int) : The number of time series data points or number
            of batches in Heidelberger and Welch's spectral method.
        heidel_welch_p (float) : Probability.
        a_matrix (ndarray) : Auxiliary matrix.
        a_matrix_1_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the first degree polynomial fit in Heidelberger and
            Welch's spectral method.
        a_matrix_2_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the second degree polynomial fit in Heidelberger and
            Welch's spectral method.
        a_matrix_3_inv (ndarray) : The (Moore-Penrose) pseudo-inverse of a
            matrix for the third degree polynomial fit in Heidelberger and
            Welch's spectral method.
        heidel_welch_c1_1 (float) : Heidelberger and Welch's C1 constant for
            the first degree polynomial fit.
        heidel_welch_c1_2 (float) : Heidelberger and Welch's C1 constant for
            the second degree polynomial fit.
        heidel_welch_c1_3 (float) : Heidelberger and Welch's C1 constant for
            the third degree polynomial fit.
        heidel_welch_c2_1 (float) : Heidelberger and Welch's C2 constant for
            the first degree polynomial fit.
        heidel_welch_c2_2 (float) : Heidelberger and Welch's C2 constant for
            the second degree polynomial fit.
        heidel_welch_c2_3 (float) : Heidelberger and Welch's C2 constant for
            the third degree polynomial fit.
        tm_1 (float) : t_distribution inverse cumulative distribution function
            for C2_1 degrees of freedom.
        tm_2 (float) : t_distribution inverse cumulative distribution function
            for C2_2 degrees of freedom.
        tm_3 (float) : t_distribution inverse cumulative distribution function
            for C2_3 degrees of freedom.

    """

    def __init__(self):
        """Initialize the Heidelberger and Welch class.

        Initialize a HeidelbergerWelch object and set the constants.

        """
        UCLBase.__init__(self)

        self.name = "heidel_welch"

        self.heidel_welch_set: bool = False
        self.heidel_welch_k: Optional[int] = None
        self.heidel_welch_n: Optional[int] = None
        self.heidel_welch_p: Optional[float] = None
        self.a_matrix: Optional[np.ndarray] = None
        self.a_matrix_1_inv: Optional[np.ndarray] = None
        self.a_matrix_2_inv: Optional[np.ndarray] = None
        self.a_matrix_3_inv: Optional[np.ndarray] = None
        self.heidel_welch_c1_1: Optional[float] = None
        self.heidel_welch_c1_2: Optional[float] = None
        self.heidel_welch_c1_3: Optional[float] = None
        self.heidel_welch_c2_1: Optional[float] = None
        self.heidel_welch_c2_2: Optional[float] = None
        self.heidel_welch_c2_3: Optional[float] = None
        self.tm_1: Optional[float] = None
        self.tm_2: Optional[float] = None
        self.tm_3: Optional[float] = None

    def set_heidel_welch_constants(
        self,
        *,
        confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
        heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    ):
        r"""Set Heidelberger and Welch constants globally.

        Set the constants necessary for application of the Heidelberger and
        Welch's [heidelberger1981]_ confidence interval generation method.

        Args:
            confidence_coefficient (float): probability (or confidence
                interval) and must be between 0.0 and 1.0. (default: 0.95)
            heidel_welch_number_points (int): the number of points in
                Heidelberger and Welch's spectral method that are used to
                obtain the polynomial fit. The parameter
                ``heidel_welch_number_points`` determines the frequency range
                over which the fit is made. (default: 50)

        """
        if confidence_coefficient <= 0.0 or confidence_coefficient >= 1.0:
            raise CRError(
                f"confidence_coefficient = {confidence_coefficient} is not in "
                "the range (0.0 1.0)."
            )

        if self.heidel_welch_set and heidel_welch_number_points == self.heidel_welch_k:
            if confidence_coefficient != self.heidel_welch_p:
                assert isinstance(self.heidel_welch_c2_1, float)
                assert isinstance(self.heidel_welch_c2_2, float)
                assert isinstance(self.heidel_welch_c2_3, float)
                p_up = (1 + confidence_coefficient) / 2
                self.tm_1 = t_inv_cdf(p_up, self.heidel_welch_c2_1)
                self.tm_2 = t_inv_cdf(p_up, self.heidel_welch_c2_2)
                self.tm_3 = t_inv_cdf(p_up, self.heidel_welch_c2_3)
                self.heidel_welch_p = confidence_coefficient
            return

        if isinstance(heidel_welch_number_points, int):
            if heidel_welch_number_points < 25:
                raise CRError(
                    "wrong number of points heidel_welch_number_points = "
                    f"{heidel_welch_number_points} is given to obtain the "
                    "polynomial fit. According to Heidelberger, and Welch, "
                    "(1981), this procedure at least needs to have 25 points."
                )
        else:
            raise CRError(
                f"heidel_welch_number_points = {heidel_welch_number_points} "
                "is the number of points and should be a positive `int`."
            )

        self.heidel_welch_k = heidel_welch_number_points
        self.heidel_welch_n = heidel_welch_number_points * 4
        self.heidel_welch_p = confidence_coefficient

        # Auxiliary matrix
        aux_array = np.arange(1, self.heidel_welch_k + 1) * 4 - 1.0
        aux_array /= 2.0 * self.heidel_welch_n

        self.a_matrix = np.empty((self.heidel_welch_k, 4), dtype=np.float64)

        self.a_matrix[:, 0] = np.ones((self.heidel_welch_k), dtype=np.float64)
        self.a_matrix[:, 1] = aux_array
        self.a_matrix[:, 2] = aux_array * aux_array
        self.a_matrix[:, 3] = self.a_matrix[:, 2] * aux_array

        # The (Moore-Penrose) pseudo-inverse of a matrix.
        # Calculate the generalized inverse of a matrix using
        # its singular-value decomposition (SVD) and including all
        # large singular values.
        self.a_matrix_1_inv = pinv(self.a_matrix[:, :2])
        self.a_matrix_2_inv = pinv(self.a_matrix[:, :3])
        self.a_matrix_3_inv = pinv(self.a_matrix)

        # Heidelberger and Welch (1981) constants Table 1
        _sigma2 = (
            0.645
            * inv(np.dot(np.transpose(self.a_matrix[:, :2]), self.a_matrix[:, :2]))[
                0, 0
            ]
        )
        self.heidel_welch_c1_1 = np.exp(-_sigma2 / 2.0)
        # Heidelberger and Welch's C2 constant for
        # the first degree polynomial fit.
        self.heidel_welch_c2_1 = int(np.rint(2.0 / (np.exp(_sigma2) - 1.0)))

        _sigma2 = (
            0.645
            * inv(np.dot(np.transpose(self.a_matrix[:, :3]), self.a_matrix[:, :3]))[
                0, 0
            ]
        )
        self.heidel_welch_c1_2 = np.exp(-_sigma2 / 2.0)
        # Heidelberger and Welch's C2 constant for
        # the second degree polynomial fit.
        self.heidel_welch_c2_2 = int(np.rint(2.0 / (np.exp(_sigma2) - 1.0)))

        _sigma2 = 0.645 * inv(np.dot(np.transpose(self.a_matrix), self.a_matrix))[0, 0]
        self.heidel_welch_c1_3 = np.exp(-_sigma2 / 2.0)
        # Heidelberger and Welch's C2 constant for
        # the third degree polynomial fit.
        self.heidel_welch_c2_3 = int(np.rint(2.0 / (np.exp(_sigma2) - 1.0)))

        p_up = (1 + confidence_coefficient) / 2
        self.tm_1 = t_inv_cdf(p_up, self.heidel_welch_c2_1)
        self.tm_2 = t_inv_cdf(p_up, self.heidel_welch_c2_2)
        self.tm_3 = t_inv_cdf(p_up, self.heidel_welch_c2_3)

        # reset the UCL Base class
        self.reset()

        # Set the flag
        self.heidel_welch_set = True

    def unset_heidel_welch_constants(self):
        """Unset the Heidelberger and Welch flag."""
        # Unset the flag
        self.heidel_welch_set = False
        self.heidel_welch_k = None
        self.heidel_welch_n = None
        self.heidel_welch_p = None
        self.a_matrix = None
        self.a_matrix_1_inv = None
        self.a_matrix_2_inv = None
        self.a_matrix_3_inv = None
        self.heidel_welch_c1_1 = None
        self.heidel_welch_c1_2 = None
        self.heidel_welch_c1_3 = None
        self.heidel_welch_c2_1 = None
        self.heidel_welch_c2_2 = None
        self.heidel_welch_c2_3 = None
        self.tm_1 = None
        self.tm_2 = None
        self.tm_3 = None

        # reset the UCL Base class
        self.reset()

    def get_heidel_welch_constants(self) -> tuple:
        """Get the Heidelberger and Welch constants."""
        return (
            self.heidel_welch_set,
            self.heidel_welch_k,
            self.heidel_welch_n,
            self.heidel_welch_p,
            self.a_matrix,
            self.a_matrix_1_inv,
            self.a_matrix_2_inv,
            self.a_matrix_3_inv,
            self.heidel_welch_c1_1,
            self.heidel_welch_c1_2,
            self.heidel_welch_c1_3,
            self.heidel_welch_c2_1,
            self.heidel_welch_c2_2,
            self.heidel_welch_c2_3,
            self.tm_1,
            self.tm_2,
            self.tm_3,
        )

    def is_heidel_welch_set(self) -> bool:
        """Return `True` if the flag is set to `True`."""
        return self.heidel_welch_set

    def get_heidel_welch_knp(self) -> tuple:
        """Get the heidel_welch_number_points, n, and confidence_coefficient."""
        return self.heidel_welch_k, self.heidel_welch_n, self.heidel_welch_p

    def get_heidel_welch_auxilary_matrices(self) -> tuple:
        """Get the Heidelberger and Welch auxilary matrices."""
        return (
            self.a_matrix,
            self.a_matrix_1_inv,
            self.a_matrix_2_inv,
            self.a_matrix_3_inv,
        )

    def get_heidel_welch_c1(self) -> tuple:
        """Get the Heidelberger and Welch C1 constants."""
        return self.heidel_welch_c1_1, self.heidel_welch_c1_2, self.heidel_welch_c1_3

    def get_heidel_welch_c2(self) -> tuple:
        """Get the Heidelberger and Welch C2 constants."""
        return self.heidel_welch_c2_1, self.heidel_welch_c2_2, self.heidel_welch_c2_3

    def get_heidel_welch_tm(self) -> tuple:
        """Get the Heidelberger and Welch t_distribution ppf.

        Get the Heidelberger and Welch t_distribution ppf for C2 degrees of
        freedom.
        """
        return self.tm_1, self.tm_2, self.tm_3

    def _ucl_impl(
        self,
        time_series_data: Union[np.ndarray, list[float]],
        *,
        confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
        heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
        fft: bool = _DEFAULT_FFT,
        test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
        train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        equilibration_length_estimate: int = _DEFAULT_EQUILIBRATION_LENGTH_ESTIMATE,  # unused (API compatibility)
        scale: str = _DEFAULT_SCALE_METHOD,
        with_centering: bool = _DEFAULT_WITH_CENTERING,
        with_scaling: bool = _DEFAULT_WITH_SCALING,
        population_standard_deviation: Optional[
            float
        ] = _DEFAULT_POPULATION_STANDARD_DEVIATION,  # unused (API compatibility)
        si: Union[str, float, int, None] = _DEFAULT_SI,  # unused (API compatibility)
        minimum_correlation_time: Optional[int] = _DEFAULT_MINIMUM_CORRELATION_TIME,  # unused (API compatibility)
        uncorrelated_sample_indices: Union[
            np.ndarray, list[int], None
        ] = _DEFAULT_UNCORRELATED_SAMPLE_INDICES,  # unused (API compatibility)
        sample_method: Optional[str] = _DEFAULT_SAMPLE_METHOD,  # unused (API compatibility)
    ) -> float:
        r"""Approximate the upper confidence limit of the mean using spectral methods.

        The Heidelberger-Welch method [heidelberger1981]_, [heidelberger1983]_
        provides an approximately unbiased estimate of the upper confidence
        limit (half-width) of a `confidence_coefficient%` confidence interval
        for the time-series mean.

        Mathematical Basis
        ------------------

        For a covariance stationary time series, the variance of the sample mean
        can be expressed in terms of the spectral density at zero frequency:

        .. math::

            \text{Var}(\bar{X}_N) = \frac{2\pi S(0)}{N} + O\left(\frac{1}{N^2}\right)

        where :math:`S(0)` is the spectral density at zero frequency and :math:`N`
        is the sample size.

        The method estimates :math:`S(0)` from the modified periodogram using
        polynomial regression and applies bias correction factors :math:`C_1` and
        :math:`C_2` from Table 1 of Heidelberger & Welch (1981) [heidelberger1981]_.

        Algorithm Description
        ---------------------

        1. **Batching**: The time series is divided into :math:`n = 4k` batches,
           where :math:`k` = ``heidel_welch_number_points``
        2. **Periodogram**: Compute the modified periodogram of batch means
        3. **Polynomial Fit**: Fit polynomials of degrees 1-3 to the log periodogram
        4. **Model Selection**: Choose the best-fitting polynomial degree based on
           test error (if test/train split is used) or residual norm
        5. **Variance Estimation**: Estimate :math:`S(0)` from the intercept of the
           best-fitting polynomial
        6. **UCL Calculation**: Compute the upper confidence limit as:

           .. math::

              UCL = t_{C_2}(\alpha) \cdot \sqrt{\frac{C_1 \cdot \widehat{S(0)}}{n}}

           where:

           - :math:`t_{C_2}(\alpha)` is the t-distribution critical value with
             :math:`C_2` degrees of freedom at confidence level :math:`\alpha`
           - :math:`C_1, C_2` are bias correction constants from Heidelberger & Welch
           - :math:`\widehat{S(0)}` is the estimated spectral density at zero frequency
           - :math:`n` is the number of batches

        Interpretation
        --------------

        The resulting confidence interval :math:`[\hat{\mu} - UCL, \hat{\mu} + UCL]`
        provides an approximate `confidence_coefficient%` confidence interval for
        the true population mean. As :math:`N \to \infty`, the UCL converges to 0
        and the sample mean converges to the population mean.

        .. note::
            A 95% confidence interval means that if the experiment were repeated
            many times, approximately 95% of the computed intervals would contain
            the true population mean. It does **not** mean there is a 95% probability
            that the specific computed interval contains the true mean.

        Key Parameters
        --------------

        - ``heidel_welch_number_points`` (k): Controls the frequency resolution and
          number of points used for polynomial fitting. Larger values provide
          smoother spectral estimates but require more data.
        - ``confidence_coefficient``: The desired coverage probability of the
          confidence interval (e.g., 0.95 for 95% confidence).
        - ``fft``: Use FFT for periodogram computation (recommended for N > 30).

        Performance Characteristics
        ---------------------------

        - **Time Complexity**: :math:`O(N \log N)` due to FFT-based periodogram
        - **Space Complexity**: :math:`O(N + k)` for data and periodogram storage
        - **Minimum Data**: Requires at least :math:`4k` data points
          (default: 200 points with k=50)

        Example
        -------

        .. code-block:: python

           from kim_convergence.ucl import HeidelbergerWelch
           import numpy as np

           # Generate sample data
           data = np.random.normal(loc=10, scale=2, size=5000)

           # Initialize algorithm
           hw = HeidelbergerWelch()

           # Compute UCL
           ucl = hw.ucl(
               data,
               confidence_coefficient=0.95,
               heidel_welch_number_points=50,
               fft=True
           )

           print(f"95% Upper Confidence Limit: {ucl:.4f}")

        Args:
            time_series_data (array_like, 1d): time series data.
            confidence_coefficient (float, optional): probability (or confidence
                interval) and must be between 0.0 and 1.0, and represents the
                confidence for calculation of relative halfwidths estimation.
                (default: 0.95)
            heidel_welch_number_points (int, optional): the number of points
                that are used to obtain the polynomial fit. The parameter
                ``heidel_welch_number_points`` determines the frequency range
                over which the fit is made. (default: 50)
            fft (bool, optional): if ``True``, use FFT convolution. FFT should
                be preferred for long time series. (default: True)
            test_size (int, float, optional): if ``float``, should be between
                0.0 and 1.0 and represent the proportion of the periodogram
                dataset to include in the test split. If ``int``, represents
                the absolute number of test samples. (default: None)
            train_size (int, float, optional): if ``float``, should be between
                0.0 and 1.0 and represent the proportion of the preiodogram
                dataset to include in the train split. If ``int``, represents
                the absolute number of train samples. (default: None)

        Returns:
            float: upper_confidence_limit
                The approximately unbiased estimate of variance of the sample
                mean, based on the degree of the fitted polynomial.

        Raises:
            CRError: If inputs are invalid or computation fails
            CRSampleSizeError: If insufficient data points

        Note:
            - If both ``test_size`` and ``train_size`` are None, no train-test
              split is performed (uses entire periodogram for fitting).
            - In the ucl method, it is mandatory to compute and set the
              ``mean`` and ``std`` property.
            - Constants :math:`C_1, C_2` are pre-computed based on
              ``heidel_welch_number_points`` and ``confidence_coefficient``.

        """

        time_series_data = np.asarray(time_series_data)

        if time_series_data.ndim != 1:
            raise CRError("time_series_data is not an array of one-dimension.")

        # We compute once and use it during iterations
        if (
            not self.heidel_welch_set
            or heidel_welch_number_points != self.heidel_welch_k
            or confidence_coefficient != self.heidel_welch_p
        ):
            self.set_heidel_welch_constants(
                confidence_coefficient=confidence_coefficient,
                heidel_welch_number_points=heidel_welch_number_points,
            )

        time_series_data_size = time_series_data.size

        assert isinstance(self.heidel_welch_n, int)
        if time_series_data_size < self.heidel_welch_n:
            msg = (
                f"{time_series_data_size} input data points are not "
                'sufficient to be used by this method.\n"HeidelbergerWelch" '
                f"at least needs {self.heidel_welch_n} data points."
            )
            cr_warning(msg)
            raise CRSampleSizeError(msg)

        number_batches = self.heidel_welch_n
        batch_size = time_series_data_size // number_batches

        processed_sample_size = number_batches * batch_size

        # Batch the data
        x_batch = batch(
            time_series_data[:processed_sample_size],
            batch_size=batch_size,
            scale=scale,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

        # Compute the mean & std of the batched data
        # to be used later in the CI method
        self.mean = time_series_data.mean()
        self.std = x_batch.std()
        self.sample_size = number_batches

        # Compute the periodogram of the sequence x_batch
        period = modified_periodogram(
            x_batch, fft=(fft and number_batches > 30), with_mean=False
        )

        left_range = range(0, period.size, 2)
        right_range = range(1, period.size, 2)

        # Compute the log of the average of adjacent periodogram values
        avg_period_lg = period[left_range] + period[right_range]
        avg_period_lg *= 0.5
        avg_period_lg = np.log(avg_period_lg)
        avg_period_lg += 0.27

        assert isinstance(self.a_matrix_1_inv, np.ndarray)
        assert isinstance(self.a_matrix_2_inv, np.ndarray)
        assert isinstance(self.a_matrix_3_inv, np.ndarray)
        assert isinstance(self.a_matrix, np.ndarray)

        # Using ordinary least squares, and fit a polynomial to the data
        if test_size is None and train_size is None:
            # Least-squares solution
            least_sqr_sol_1 = np.matmul(self.a_matrix_1_inv, avg_period_lg)
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps1 = norm(
                avg_period_lg - np.matmul(self.a_matrix[:, :2], least_sqr_sol_1)
            )

            # Least-squares solution
            least_sqr_sol_2 = np.matmul(self.a_matrix_2_inv, avg_period_lg)
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps2 = norm(
                avg_period_lg - np.matmul(self.a_matrix[:, :3], least_sqr_sol_2)
            )

            # Least-squares solution
            least_sqr_sol_3 = np.matmul(self.a_matrix_3_inv, avg_period_lg)
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps3 = norm(avg_period_lg - np.matmul(self.a_matrix, least_sqr_sol_3))
        else:
            ind_train, ind_test = train_test_split(
                avg_period_lg, train_size=train_size, test_size=test_size
            )

            # Least-squares solution
            least_sqr_sol_1 = np.matmul(
                self.a_matrix_1_inv[:, ind_train], avg_period_lg[ind_train]
            )
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps1 = norm(
                avg_period_lg[ind_test]
                - np.matmul(self.a_matrix[ind_test, :2], least_sqr_sol_1)
            )

            # Least-squares solution
            least_sqr_sol_2 = np.matmul(
                self.a_matrix_2_inv[:, ind_train], avg_period_lg[ind_train]
            )
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps2 = norm(
                avg_period_lg[ind_test]
                - np.matmul(self.a_matrix[ind_test, :3], least_sqr_sol_2)
            )

            # Least-squares solution
            least_sqr_sol_3 = np.matmul(
                self.a_matrix_3_inv[:, ind_train], avg_period_lg[ind_train]
            )
            # Error of solution ||avg_period_lg - a_matrix*time_series_data||
            eps3 = norm(
                avg_period_lg[ind_test]
                - np.matmul(self.a_matrix[ind_test, :], least_sqr_sol_3)
            )

        # Find the best fit
        best_fit_index = np.argmin((eps1, eps2, eps3))

        if best_fit_index == 0:
            # get unbiased_estimate, which is an unbiased estimate of
            # log(confidence_coefficient(0)).
            unbiased_estimate = least_sqr_sol_1[0]
            heidel_welch_c = self.heidel_welch_c1_1
            hwl_tm = self.tm_1
        elif best_fit_index == 1:
            # get unbiased_estimate, which is an unbiased estimate of
            # log(confidence_coefficient(0)).
            unbiased_estimate = least_sqr_sol_2[0]
            heidel_welch_c = self.heidel_welch_c1_2
            hwl_tm = self.tm_2
        else:
            # get unbiased_estimate, which is an unbiased estimate of
            # log(confidence_coefficient(0)).
            unbiased_estimate = least_sqr_sol_3[0]
            heidel_welch_c = self.heidel_welch_c1_3
            hwl_tm = self.tm_3

        # The variance of the sample mean of a covariance stationary sequence is
        # given approximately by confidence_coefficient(O)/N, the spectral density
        # at zero frequency divided by the sample size.

        # Calculate the approximately unbiased estimate of the variance of the
        # sample mean
        sigma_sq = heidel_welch_c * np.exp(unbiased_estimate) / number_batches

        # The standard deviation of the mean within the dataset. The
        # standard_error_of_mean provides a measurement for spread. The smaller
        # the spread the more accurate.
        standard_error_of_mean = sqrt(sigma_sq)

        assert isinstance(hwl_tm, float)
        self.upper_confidence_limit = hwl_tm * standard_error_of_mean
        return float(self.upper_confidence_limit)


def heidelberger_welch_ucl(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    fft: bool = _DEFAULT_FFT,
    test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
    train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
    obj: Optional[HeidelbergerWelch] = None,
) -> float:
    """Approximate the upper confidence limit of the mean."""
    heidelberger_welch = HeidelbergerWelch() if obj is None else obj
    upper_confidence_limit = heidelberger_welch.ucl(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        heidel_welch_number_points=heidel_welch_number_points,
        fft=fft,
        test_size=test_size,
        train_size=train_size,
    )
    return upper_confidence_limit


def heidelberger_welch_ci(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    fft: bool = _DEFAULT_FFT,
    test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
    train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
    obj: Optional[HeidelbergerWelch] = None,
) -> tuple[float, float]:
    r"""Approximate the confidence interval of the mean.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or
            confidence interval) and must be between 0.0 and 1.0, and
            represents the confidence for calculation of relative
            halfwidths estimation. (default: 0.95)
        heidel_welch_number_points (int, optional): the number of points
            that are used to obtain the polynomial fit. The parameter
            ``heidel_welch_number_points`` determines the frequency range
            over which the fit is made. (default: 50)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should
            be preferred for long time series. (default: True)
        test_size (int, float, optional): if ``float``, should be between
            0.0 and 1.0 and represent the proportion of the periodogram
            dataset to include in the test split. If ``int``, represents
            the absolute number of test samples. (default: None)
        train_size (int, float, optional): if ``float``, should be between
            0.0 and 1.0 and represent the proportion of the preiodogram
            dataset to include in the train split. If ``int``, represents
            the absolute number of train samples. (default: None)
        obj (HeidelbergerWelch, optional): instance of ``HeidelbergerWelch``
            (default: None)

    Returns:
        float, float: confidence interval.
            The estimate of confidence Limits for the mean.

    """
    heidelberger_welch = HeidelbergerWelch() if obj is None else obj
    confidence_limits = heidelberger_welch.ci(
        time_series_data=time_series_data,
        confidence_coefficient=confidence_coefficient,
        heidel_welch_number_points=heidel_welch_number_points,
        fft=fft,
        test_size=test_size,
        train_size=train_size,
    )
    return confidence_limits


def heidelberger_welch_relative_half_width_estimate(
    time_series_data: Union[np.ndarray, list[float]],
    *,
    confidence_coefficient: float = _DEFAULT_CONFIDENCE_COEFFICIENT,
    heidel_welch_number_points: int = _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    fft: bool = _DEFAULT_FFT,
    test_size: Union[int, float, None] = _DEFAULT_TEST_SIZE,
    train_size: Union[int, float, None] = _DEFAULT_TRAIN_SIZE,
    obj: Optional[HeidelbergerWelch] = None,
) -> float:
    r"""Get the relative half width estimate.

    The relative half width estimate is the confidence interval
    half-width or upper confidence limit (UCL) divided by the sample mean.

    The UCL is calculated as a `confidence_coefficient%` confidence
    interval for the mean, using the portion of the time series data, which
    is in the stationarity region.

    Args:
        time_series_data (array_like, 1d): time series data.
        confidence_coefficient (float, optional): probability (or
            confidence interval) and must be between 0.0 and 1.0, and
            represents the confidence for calculation of relative
            halfwidths estimation. (default: 0.95)
        heidel_welch_number_points (int, optional): the number of points
            that are used to obtain the polynomial fit. The parameter
            ``heidel_welch_number_points`` determines the frequency range
            over which the fit is made. (default: 50)
        fft (bool, optional): if ``True``, use FFT convolution. FFT should
            be preferred for long time series. (default: True)
        test_size (int, float, optional): if ``float``, should be between
            0.0 and 1.0 and represent the proportion of the periodogram
            dataset to include in the test split. If ``int``, represents
            the absolute number of test samples. (default: None)
        train_size (int, float, optional): if ``float``, should be between
            0.0 and 1.0 and represent the proportion of the preiodogram
            dataset to include in the train split. If ``int``, represents
            the absolute number of train samples. (default: None)
        obj (HeidelbergerWelch, optional): instance of ``HeidelbergerWelch``
            (default: None)

    Returns:
        float: relative half width estimate

    """
    heidelberger_welch = HeidelbergerWelch() if obj is None else obj
    try:
        relative_half_width_estimate = heidelberger_welch.relative_half_width_estimate(
            time_series_data=time_series_data,
            confidence_coefficient=confidence_coefficient,
            heidel_welch_number_points=heidel_welch_number_points,
            fft=fft,
            test_size=test_size,
            train_size=train_size,
        )
    except CRError as e:
        raise CRError("Failed to get the relative_half_width_estimate.") from e
    return relative_half_width_estimate

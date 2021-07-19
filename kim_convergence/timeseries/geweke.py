"""Geweke method module."""

import numpy as np

from convergence import CVGError

__all__ = [
    'geweke',
]


def geweke(x: np.ndarray,
           *,
           first: float = 0.1,
           last: float = 0.5,
           intervals: int = 20):
    r"""Compute z-scores for convergence diagnostics.

    # Compare the mean of the first % of series with the mean of the last % of
    # series. x is divided into a number of segments for which this difference
    # is computed. If the series is converged, this score should oscillate
    # between -1 and 1.

    # The Geweke diagnostic tests the null hypothesis that the Markov chain is
    # in the stationary distribution and produces z-statistics for each
    # estimated parameter.

    # Parameters
    # ----------
    # x : 1D array-like
    #   The trace of some stochastic parameter.
    # first : float
    #   The fraction of series at the beginning of the trace.
    # last : float
    #   The fraction of series at the end to be compared with the section
    #   at the beginning.
    # intervals : int
    #   The number of segments.
    # Returns
    # -------
    # scores : list [[]]
    #   Return a list of [i, score], where i is the starting index for each interval and score the
    #   Geweke score on the interval.
    # Notes
    # -----
    # The Geweke score on some series x is computed by:
    #   .. math::
    #       \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    # where :math:`E` stands for the mean, :math:`V` the variance,
    # :math:`x_s` a section at the start of the series and
    # :math:`x_e` a section at the end of the series.
    # References
    # ----------
    # * Geweke (1992)

    """
    if first + last >= 1:
        msg = 'Invalid intervals for Geweke convergence analysis:'
        msg += '({}, {}), '.format(first, last)
        msg += 'where {} + {} >= 1'.format(first, last)
        raise CVGError(msg)

    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            msg = 'Invalid intervals for Geweke convergence analysis:'
            msg += '({}, {})'.format(first, last)
            raise CVGError(msg)

    x = np.array(x, copy=False)

    if x.ndim != 1:
        msg = "x is not an array of one-dimension."
        raise CVGError(msg)

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.linspace(
        0, last_start_idx, num=intervals, endpoint=True, dtype=int)

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]
        z_score = first_slice.mean() - last_slice.mean()
        scale_ = first_slice.var() + last_slice.var()
        if scale_ > 0.0:
            z_score /= np.sqrt(scale_)
        zscores.append([start, z_score])

    return np.array(zscores)

import unittest
import numpy as np

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError


class TimeseriesModule:
    """Test timeseries module components."""

    def test_set_heidel_welch_constants(self):
        """Test set_heidel_welch_constants function."""
        cvg.set_heidel_welch_constants()

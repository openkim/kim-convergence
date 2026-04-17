"""Single-threaded FFT implementation using pocketfft.

This module provides single-threaded FFT for environments where
multi-threading causes deadlocks (e.g., LAMMPS with MPI).
"""

try:
    from ._stfft_core import rfft, irfft
    STFFT_AVAILABLE = True
except ImportError:
    STFFT_AVAILABLE = False

    def rfft(x, n=None):
        raise ImportError(
            "Single-threaded FFT extension is not built. "
            "Reinstall kim-convergence with a C++ compiler available."
        )

    def irfft(x, n=None):
        raise ImportError(
            "Single-threaded FFT extension is not built. "
            "Reinstall kim-convergence with a C++ compiler available."
        )


__all__ = ['rfft', 'irfft', 'STFFT_AVAILABLE']


# Single-Threaded FFT Extension

This is a custom-compiled pocketfft extension for environments where
multi-threading causes deadlocks (e.g., LAMMPS with MPI).

## What This Does

Provides single-threaded FFT without any OpenMP/threading, solving
deadlock issues in multi-process simulation codes.

## Implementation

- Uses pocketfft (same FFT backend as NumPy 1.20+)
- Compiled without OpenMP flags
- Header-only, no external dependencies
- Explicitly single-threaded (nthreads=1 hardcoded)

## Build

This extension is built automatically when installing kim-convergence:

```bash
pip install kim-convergence
```

Or for development:

```bash
pip install -e .
```

## Usage

Set the environment variable to use the single-threaded backend:

```bash
export KIM_CONV_STFFT=1
mpirun -np 20 lmp -in in.my_simulation
```

If the extension is not built, a warning is issued and NumPy's FFT is used.

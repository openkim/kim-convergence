# kim-convergence

[![Python package](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml/badge.svg)](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml)
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/kim-convergence.svg)](https://anaconda.org/conda-forge/kim-convergence)
[![PyPI](https://img.shields.io/pypi/v/kim-convergence.svg)](https://pypi.python.org/pypi/kim-convergence)
[![License](https://img.shields.io/badge/license-LGPL--2.1--or--later-blue)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/kim-convergence/badge/?version=latest)](https://kim-convergence.readthedocs.io/en/latest/?badge=latest)

**Stop guessing how long your simulation should run.**

`kim-convergence` auto-detects equilibration and stops precisely when your data
is statistically reliable.

## Why Use kim-convergence?

Manual estimation of simulation length is unreliable and wasteful. Different
observables converge at different rates, and visual inspection can't guarantee
statistical reliability. This package automates it all using proven methods like
MSER-m for equilibration and adaptive confidence intervals for precision.

### The Challenge: Estimating Simulation Length

<table>
  <tr>
    <td> <img src="./doc/files/vid1_T.gif?raw=true" width="200" height="200"> </td>
    <td> <img src="./doc/files/vid1_P.gif?raw=true" width="200" height="200"> </td>
    <td> <img src="./doc/files/vid1_V.gif?raw=true" width="200" height="200"> </td>
  </tr>
</table>

<table>
  <tr>
    <td> <img src="./doc/files/vid2_T.gif?raw=true" width="200" height="200"> </td>
    <td> <img src="./doc/files/vid2_P.gif?raw=true" width="200" height="200 "> </td>
    <td> <img src="./doc/files/vid2_V.gif?raw=true" width="200" height="200 "> </td>
  </tr>
</table>

**Top row:** 10 ps simulation | **Bottom row:** 50 ps simulation

**Key observations:**

- Different properties (e.g., temperature, pressure, volume) converge at varying
  speeds.
- Visual checks alone can't ensure statistical reliability. Where running too
  short biases results and too long wastes resources.
- `kim-convergence` solves this by detecting equilibration and controlling run
  length based on your accuracy needs.

[Jump to Quick Start](#quick-start) or [docs](https://kim-convergence.readthedocs.io/) for theory and examples.

It is desirable to simulate the minimum amount of time necessary to reach an
acceptable amount of uncertainty in the quantity of interest.

**The first place you save time is by cutting the warm-up correctly.**

## How Do You Spot the Warm-Up?

<table>
  <tr>
    <td> <img src="./doc/files/vid1_T_Eq.gif?raw=true" width="200" height="200" alt="Temperature equilibration detection animation"> </td>
  </tr>
</table>

Equilibration = the transient stretch until your sim forgets its starting point.

- Cut too early -> biased averages
- Cut too late -> wasted cycles

`kim-convergence` runs **MSER-m** automatically: it scans truncation points and
picks the one that minimizes the batch-means standard error.

## Key Features

- **Auto equilibration detection** – MSER-m + autocorrelation refinement
- **Adaptive run length** – quits the instant every observable hits your
  accuracy target
- **Proven CI engines** – MSER-m, Heidelberger-Welch, N-SKART, uncorrelated
  samples
- **Time-series toolkit** – statistical inefficiency, autocorrelation, effective
  sample size, uncorrelated subsampling
- **Zero-friction integration** – callbacks for LAMMPS & OpenMM using
  one-function API for custom codes
- **Multi-observable** – per-variable accuracy. The run ends only when **all**
  converge

## ⚙️ Installation

Install with one command:

```bash
pip install kim-convergence

# or:

conda install -c conda-forge kim-convergence
```

<a id="quick-start"></a>
## 🚀 Quick Start

```python
import numpy as np
import kim_convergence as cr

def trajectory(nstep):
    # Fake temperature data from a simulation (in K)
    return np.random.normal(300, 5, nstep)  # Mean 300 K, std 5 K

if __name__ == "__main__":
    result = cr.run_length_control(
        get_trajectory=trajectory,
        relative_accuracy=0.01,        # Target ±1 % precision on the mean
        maximum_run_length=200_000,
        fp="return",
    )

    print(result) # txt with converged length, UCL, effective sample size …
```

That’s it—head to the [docs](https://kim-convergence.readthedocs.io/) for
platform notes, developer install, LAMMPS/OpenMM integration, multi-observable
examples, etc.

## Documentation

For installation instructions, usage examples, theoretical background, best
practices, troubleshooting tips, and the full API reference, see the
[documentation](https://kim-convergence.readthedocs.io/).

## Contributing

Bug reports, feature requests, pull requests:
[GitHub Issues](https://github.com/openkim/kim-convergence/issues)

Guidelines: [Contributing Guide](https://kim-convergence.readthedocs.io/en/latest/contributing.html)

## License

LGPL-2.1-or-later — see [LICENSE](LICENSE).

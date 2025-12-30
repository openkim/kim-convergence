# kim-convergence utility module

[![Python package](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml/badge.svg)](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml)
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/kim-convergence.svg)](https://anaconda.org/conda-forge/kim-convergence)
[![PyPI](https://img.shields.io/pypi/v/kim-convergence.svg)](https://pypi.python.org/pypi/kim-convergence)
[![License](https://img.shields.io/badge/license-LGPL--2.1--or--later-blue)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/kim-convergence/badge/?version=latest)](https://kim-convergence.readthedocs.io/en/latest/?badge=latest)

## How do you automatically estimate the length of the simulation required?

### Problem: Estimating Simulation Length

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

**Key observations:**

- Different observables converge at different rates
- Visual inspection alone cannot determine statistical reliability
- Running too short leads to bias, too long wastes computational resources

It is desirable to simulate the minimum amount of time necessary to reach an
acceptable amount of uncertainty in the quantity of interest.

## How do you automatically estimate the length of the warm-up period required?

<table>
  <tr>
    <td> <img src="./doc/files/vid1_T_Eq.gif?raw=true" width="200" height="200"> </td>
  </tr>
</table>

Welcome to **kim-convergence** module!

`kim-convergence` package solves this by providing
**automatic equilibration detection** and **adaptive run length control** based
on statistical confidence intervals.

## Features

- **Automatic Equilibration Detection**: Identify when simulations reach
  steady-state using MSER-m and related algorithms
- **Adaptive Run Length Control**: Extend simulations only until desired
  statistical accuracy is achieved
- **Multiple UCL Methods**: MSER-m, Heidelberger-Welch, N-SKART, and
  uncorrelated samples
- **Time Series Analysis**: Statistical inefficiency, autocorrelation, effective
  sample size
- **Integration Support**: Callbacks for LAMMPS, OpenMM, and custom simulators
- **Multiple Observables**: Handle different convergence rates for different
  quantities

## Installing kim-convergence

### Requirements

You need Python 3.9 or later to run `kim-convergence`. You can have multiple
Python versions (2.x and 3.x) installed on the same system without problems.

To install Python 3 for different Linux flavors, macOS and Windows, packages
are available at\
[https://www.python.org/getit/](https://www.python.org/getit/)

### Using pip

**pip** is the most popular tool for installing Python packages, and the one
included with modern versions of Python.

`kim-convergence` can be installed with `pip`:

```sh
pip install kim-convergence
```

**NOTE:**

Depending on your Python installation, you may need to use `pip3` instead of
`pip`.

```sh
pip3 install kim-convergence
```

Depending on your configuration, you may have to run `pip` like this:

```sh
python3 -m pip install kim-convergence
```

### Using pip (GIT Support)

`pip` currently supports cloning over `git`

```sh
pip install git+https://github.com/openkim/kim-convergence.git
```

For more information and examples, see the [pip install](https://pip.pypa.io/en/stable/reference/pip_install/#id18) reference.

### Using conda

**conda** is the package management tool for Anaconda Python installations.

Installing `kim-convergence` from the `conda-forge` channel can be achieved by
adding `conda-forge` to your channels with:

```sh
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once the `conda-forge` channel has been enabled, `kim-convergence` can be
installed with:

```sh
conda install kim-convergence
```

It is possible to list all of the versions of `kim-convergence` available on
your platform with:

```sh
conda search kim-convergence --channel conda-forge
```

## Basic Usage

Basic usage involves importing kim-convergence and use the utility to control
the length of the time series data from a simulation run or a sampling approach,
or a dump file from the previously done simulation.

The main requirement is a `get_trajectory` function. **`get_trajectory`** is a
callback function with a specific signature of

```get_trajectory(nstep: int) -> 1darray```

if we only have one variable or,

```get_trajectory(nstep: int) -> 2darray```

with the shape of return array as,

```(number_of_variables, nstep)```.

For example,

```py
rng = np.random.RandomState(12345)
stop = 0

def get_trajectory(step: int) -> np.ndarray:
  global stop
  start = stop
  if 100000 < start + step:
    step = 100000 - start
  stop += step
  data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
  return data
```

**NOTE:**

To use extra arguments in calling the ``get_trajectory`` function, one can use
the other specific signature of

```get_trajectory(nstep: int, args: dict) -> 1darray```

or

```get_trajectory(nstep: int, args: dict) -> 2darray```,

where all the extra required parameters and arguments can be provided with the
args.

```py
rng = np.random.RandomState(12345)
args = {'stop': 0, 'maximum_steps': 100000}

def get_trajectory(step: int, args: dict) -> np.ndarray:
  start = args['stop']
  if args['maximum_steps'] < start + step:
    step = args['maximum_steps'] - start
  args['stop'] += step
  data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
  return data
```

---

Then call the `run_length_control` function as below,

```py
import kim_convergence as cr

msg = cr.run_length_control(
  get_trajectory=get_trajectory,
  number_of_variables=1,
  initial_run_length=1000,
  maximum_run_length=100000,
  relative_accuracy=0.01,
  fp_format='json'
)
```

or

```py
import kim_convergence as cr

msg = cr.run_length_control(
  get_trajectory=get_trajectory,
  get_trajectory_args=args,
  number_of_variables=1,
  initial_run_length=1000,
  maximum_run_length=100000,
  relative_accuracy=0.01,
  fp_format='json'
)
```

An estimate produced by a simulation typically has an accuracy requirement and
is an input to the utility. This requirement means that the experimenter wishes
to run the simulation only until an estimate meets this accuracy requirement.
Running the simulation less than this length would not provide the information
needed while running it longer would be a waste of computing time. In the above
example, the accuracy requirement is specified as the relative accuracy.

In case of having more than one variable,

```py
rng = np.random.RandomState(12345)
stop = 0

def get_trajectory(step: int) -> np.ndarray:
  global stop
  start = stop
  if 100000 < start + step:
    step = 100000 - start
  stop += step
  data = np.ones((3, step)) * 10 + (rng.random_sample(3 * step).reshape(3, step) - 0.5)
  return data
```

Then call the `run_length_control` function as below,

```py
import kim_convergence as cr

msg = cr.run_length_control(
  get_trajectory=get_trajectory,
  number_of_variables=3,
  initial_run_length=1000,
  maximum_run_length=100000,
  relative_accuracy=0.01,
  fp_format='json'
)
```

**NOTE:**

All the values returned from this `get_trajectory` function should be finite
values, otherwise the code will stop wih error message explaining the issue.

```py
ERROR(@_get_trajectory): there is/are value/s in the input which is/are non-finite or not number.
```

Thus, one should remove infinit values or Not a Number (NaN) values from the
returning array within the `get_trajectory` function.

---

The run-length control procedure employs `initial_run_length` parameter. It
begins at time 0 and starts calling the `get_trajectory` function with the
provided number of steps (e.g. ```initial_run_length=1000```). At this point,
and with no assumptions about the distribution of the observable of interest,
it tries to estimate an equilibration time. Failing to find the transition
point will request more data and call the `get_trajectory` function until it
finds the equilibration time or hits the maximum run length limit
(e.g. ```maximum_run_length=100000```).

At this point, and after finding an optimal equilibration time, the confidence
interval (CI) generation method is applied to the set of available data points.
If the resulting confidence interval met the provided accuracy value
(e.g. ```relative_accuracy=0.01```), the simulation is terminated. If not, the
simulation is continued by requesting more data and calling the `get_trajectory`
function again and again until it does. This procedure continues until the
criteria is met or it reaches the maximum run length limit.

The `relative_accuracy` as mentioned above, is the relative precision and
defined as a half-width of the estimator's confidence interval or an
approximated upper confidence limit (UCL) divided by the computed sample mean.

The UCL is calculated as a `confidence_coefficient%` confidence interval for
the mean, using the portion of the time series data, which is in the stationary
region. If the ratio is bigger than `relative_accuracy`, the length of the time
series is deemed not long enough to estimate the mean with sufficient accuracy,
which means the run should be extended.

The accuracy parameter `relative_accuracy` specifies the maximum relative error
that will be allowed in the mean value of the data point series. In other words,
the distance from the confidence limit(s) to the mean (which is also known as
the precision, half-width, or margin of error). A value of `0.01` is usually
used to request two digits of accuracy, and so forth.

The parameter ```confidence_coefficient``` is the confidence coefficient and
often, the values ```0.95``` is used. For the confidence coefficient,
`confidence_coefficient`, we can use the following interpretation, If thousands
of samples of n items are drawn from a population using simple random sampling
and a confidence interval is calculated for each sample, the proportion of
those intervals that will include the true population mean is
`confidence_coefficient`.

## Documentation

Complete documentation is available at: https://kim-convergence.readthedocs.io/

The documentation includes:

- **Getting Started**: Installation and basic usage
- **Best Practices**: Guidelines for accuracy requirements and method selection
- **Theory**: Statistical background and algorithm details
- **Examples**: Copy-paste ready code snippets
- **API Reference**: Complete function documentation
- **Troubleshooting**: Common issues and solutions

## Contact us

If something is not working as you think it should or would like it to, please
get in touch with us! Further, if you have an algorithm or any idea that you
would want to try using the kim-convergence, please get in touch with us, we
would be glad to help!

[![Gitter](https://badges.gitter.im/openkim/kim-convergence.svg)](https://gitter.im/openkim/kim-convergence?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Contributing

Contributions are very welcome.

## Copyright

Copyright (c) 2021-2026, Regents of the University of Minnesota.\
All Rights Reserved

## Contributors

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

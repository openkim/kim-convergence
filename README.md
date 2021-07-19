# kim-convergence utility module

[![Python package](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml/badge.svg)](https://github.com/openkim/kim-convergence/actions/workflows/python-package.yml)
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/kim-convergence.svg)](https://anaconda.org/conda-forge/kim-convergence)
[![PyPI](https://img.shields.io/pypi/v/kim-convergence.svg)](https://pypi.python.org/pypi/kim-convergence)
[![License](https://img.shields.io/badge/license-LGPLv2-blue)](LICENSE)

## How do you automatically estimate the length of the simulation required?

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

It is desirable to simulate the minimum amount of time necessary to reach an
acceptable amount of uncertainty in the quantity of interest.

## How do you automatically estimate the length of the warm-up period required?

<table>
  <tr>
    <td> <img src="./doc/files/vid1_T_Eq.gif?raw=true" width="200" height="200"> </td>
  </tr>
</table>

Welcome to **kim-convergence** module!

The kim-convergence package is designed to help in automatic equilibration
detection & run length control.

## Document

<span style="font-size:300%; color:red; font-weight: 900;">!WORK IN PROGRESS!</span>

## Installing kim-convergence

### Requirements

You need Python 3.7 or later to run `kim-convergence`. You can have multiple
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

**Note:**

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

The basic use is to control the length of the time series data from a
simulation run. For example, one can start drawing ``initial_run_length`` data
points (the number of observations or samples) by calling the ``get_trajectory``
function in a loop to reach equilibration or pass the ``warm-up`` period.

**Note** ``get_trajectory`` is a callback function with a specific signature of
`get_trajectory(nstep: int) -> 1darray` if we only have one variable or
`get_trajectory(nstep: int) -> 2darray` with the shape of
`(number_of_variables, nstep)`.

All the values returned from this function should be finite values, otherwise
the code will stop wih error message explaining the issue.

Example:

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

To use extra arguments in calling the ``get_trajectory`` function, one can use
the other specific signature of
`get_trajectory(nstep: int, args: dict) -> 1darray` or
`get_trajectory(nstep: int, args: dict) -> 2darray` with the shape of
`(number_of_variables, nstep)` where all the required variables can be pass
thrugh the args dictionary.

```py
  rng = np.random.RandomState(12345)
  targs = {'stop': 0}

  def get_trajectory(step: int, targs: dict) -> np.ndarray:
    start = targs['stop']
    if 100000 < start + step:
      step = 100000 - start
    targs['stop'] += step
    data = np.ones(step) * 10 + (rng.random_sample(step) - 0.5)
    return data
```

Then the code continues drawing observations until some pre-specified level of
``absolute`` or ``relative`` precision has been reached. The relative
``precision`` is defined as a half-width of the estimator's confidence interval
(CI). At each call, an upper confidence limit (``UCL``) is approximated.
If UCL is less than the pre-specified absolute precision ``absolute_accuracy``
or if the relative UCL (UCL divided by the computed sample mean) is less than a
pre-specified value ``relative_accuracy``, the drawing of observations is
terminated.

The UCL is calculated as a `confidence_coefficient%` confidence interval for
the mean, using the portion of the time series data, which is in the
stationary region.

The ``Relative accuracy`` is the confidence interval half-width or UCL divided
by the sample mean. If the ratio is bigger than `relative_accuracy`, the length
of the time series is deemed not long enough to estimate the mean with
sufficient accuracy, which means the run should be extended.

In order to avoid problems caused by sequential UCL evaluation cost, this
calculation should not be repeated too frequently. Heidelberger and Welch (1981)
[2]_ suggested increasing the run length by a factor of
`run_length_factor > 1.5`, each time, so that estimate has the same, reasonably
large proportion of new data.

The accuracy parameter `relative_accuracy` specifies the maximum relative error
that will be allowed in the mean value of time-series data. In other words, the
distance from the confidence limit(s) to the mean (which is also known as the
precision, half-width, or margin of error). A value of `0.01` is usually used to
request two digits of accuracy, and so forth.

The parameter ``confidence_coefficient`` is the confidence coefficient and
often, the values 0.95 is used. For the confidence coefficient,
`confidence_coefficient`, we can use the following interpretation,

If thousands of samples of n items are drawn from a population using simple
random sampling and a confidence interval is calculated for each sample, the
proportion of those intervals that will include the true population mean is
`confidence_coefficient`.

## Contact us

If something is not working as you think it should or would like it to, please
get in touch with us! Further, if you have an algorithm or any idea that you
would want to try using the kim-convergence, please get in touch with us, we
would be glad to help!

[![Join the chat at https://gitter.im/kim-convergence](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/kim-convergence/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Contributing

Contributions are very welcome.

## Copyright

Copyright (c) 2021, Regents of the University of Minnesota.\
All Rights Reserved

## Contributors

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

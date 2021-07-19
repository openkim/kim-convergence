# kim-convergence utility module

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

## Copyright

Copyright (c) 2021, Regents of the University of Minnesota.\
All Rights Reserved

## Contributing

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

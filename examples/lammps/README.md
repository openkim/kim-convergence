# LAMMPS examples for kim-convergence utility module

The kim-convergence package is designed to help in automatic equilibration
detection & run length control. You can use this package in your script and
post-process the LAMMPS output files (log file or dump files), or you can use
the provided Python script
[`run_length_control.py`](https://github.com/openkim/kim-convergence/blob/main/examples/lammps/run_length_control.py)
in this folder and use it in your LAMMPS input script.

For the latter, the LAMMPS uses the
[`python`](https://docs.lammps.org/python.html) command to define and execute a
Python function. The Python code for the run length control is included in an
auxiliary file `run_length_control.py.`

The function has arguments that mapped to LAMMPS variables (also defined in the
input script), and it can return a value to a LAMMPS variable. This is a
mechanism for the input script to pass information to a Python code, ask Python
to execute the code, and return data to the input script. The
`run_length_control` function imports other Python modules and contains
`"call back"` to LAMMPS through its Python-wrapped library interface, in the
manner described in the
[LAMMPS Python run doc page](https://docs.lammps.org/Python_run.html).
By this mechanism, it issues LAMMPS input script commands.

## `run_length_control` use

syntax

```shell
python run_length_control input ... format format_args file run_length_control.py
python run_length_control invoke
```

where `...` are the input arguments as,

```shell
args = N SELF i2 i3 ... iN
```

``N`` is the number of inputs to the `run_length_control` function.

``SELF`` is a reference to LAMMPS itself and is accessed later by the Python
function.

``i2`` is an integer number and specifies on what timesteps the input values
will be used. Only timesteps that are a multiple of `i2`, including timestep
`0`, will contribute values.

Note that `SELF` and `i2` are mandatory inputs to the `run_length_control`
function. The rest of `i3 ... iN` can be strings and get parsed to the correct
value in the Python function.

``format`` is a keyword followed by ``format_args``. Where `format_args` is a
string with N characters. The order of characters corresponds to the N inputs
and each character (i,f,s,p) corresponds in order to an
input value `i` = integer, `f` = floating point, `s` = string, and `p` = SELF.

Each character defines the type of the corresponding input value of the Python
function and affects the type conversion that is performed internally as data
is passed back and forth between LAMMPS and Python.

**NOTE:**

Each input value after `i2` can be the result of a `compute` or a `fix` or the
evaluation of an equal-style or vector-style `variable`.

In each case, the `compute`, `fix`, or `variable` must produce a global
quantity, not a per-atom or local quantity. And the global quantity must be a
scalar, not a vector or array.

``Computes`` that produce global quantities are those which do not have the word
atom in their style name. Only a few fixes produce global quantities.

``Variables of style equal or vector`` are the only ones that can be used as an
input here. ``Variables of style atom`` cannot be used, since they produce
per-atom values.

Each input value following the argument `lb`, or `lbound` and `ub`, or
`ubound` must previously be defined in the input script as the evaluation of an
equal-style `variable`.

### Examples

For example, to use the `pea` variable defined below as a control value to the
Python function,

```shell
variable natoms equal "count(all)"
variable pea    equal "c_thermo_pe/v_natoms"
```

you can pass the variable and invoke the Python function as,

```shell
python run_length_control input 4 SELF 1 variable pea format piss file run_length_control.py
python run_length_control invoke
```

where `4` is the number of inputs to the Python function.

- `SELF` is a reference to the LAMMPS
- `1` specifies to use the variable `pea` every steps
- `variable` is a string indicating to the Python function that the next input
  is a LAMMPS variable
- `pea` is a string name of the LAMMPS varable

The format for `SELF 1 variable pea` would be `piss`.

Another example is to use te `pea` variable and the `temp` compute in LAMMPS as
a control values,

```shell
python run_length_control input 6 SELF 1 variable pea compute thermo_temp format pissss file run_length_control.py
python run_length_control invoke
```

where `6` is the number of inputs to the Python function.

- `SELF` is a reference to the LAMMPS
- `1` specifies to use the variable `pea` and compute `thermo_temp` every steps
- `variable` is a string indicating to the Python function that the next input
  is a LAMMPS variable
- `pea` is a string name of the LAMMPS varable
- `compute` is a string indicating to the Python function that the next input
  is a LAMMPS compute
- `thermo_temp` is a LAMMPS compute with the ID of `thermo_temp` and is created
  when LAMMPS starts up

The format for `SELF 1 variable pea compute thermo_temp` would be `pissss`.

In the third example, we use another variable `t_ub` as,

```shell
variable t_ub equal 0.71
```

to control the upper bound on the computed temperature during the simulation as,

```shell
python run_length_control input 8 SELF 1 variable pea compute thermo_temp ub t_ub format pissssss file run_length_control.py
python run_length_control invoke
```

where `8` is the number of inputs to the Python function.

- `SELF` is a reference to the LAMMPS
- `1` specifies to use the variable `pea` and compute `thermo_temp` every steps
- `variable` is a string indicating to the Python function that the next input
  is a LAMMPS variable
- `pea` is a string name of the LAMMPS varable
- `compute` is a string indicating to the Python function that the next input
  is a LAMMPS compute
- `thermo_temp` is a LAMMPS compute with the ID of `thermo_temp` and is created
  when LAMMPS starts up
- `ub` or `ubound` is a string. It is a key indicating to the Python function
  that the previously defined `compute`, `fix`, or `variable` has an upper
  bound check and stops the simulation if the value crosses this bound.
  Here `ub` indicates that there is an upper bound to the LAMMPS compute with
  the ID of `thermo_temp`
- `t_ub` is a LAMMPS variable equals with `0.71`

The format for `SELF 1 variable pea compute thermo_temp ub t_ub` would be
`pissssss`.

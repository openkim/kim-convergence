"""Run length control example by LAMMPS."""

import numpy as np
from lammps import lammps
import convergence as cvg

# Initial run length
cvg_initial_run_length = 1000
# Run length increasing factor
cvg_run_length_factor = 1
# The maximum run length represents a cost constraint.
cvg_maximum_run_length = 1000 * cvg_initial_run_length
# The maximum number of steps as an equilibration hard limit. If the
# algorithm finds equilibration_step greater than this limit it will fail.
# For the default None, the function is using `cvg_maximum_run_length // 2` as
# the maximum equilibration step.
cvg_maximum_equilibration_step = None
# Maximum number of independent samples.
cvg_sample_size = None
# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1, relative_accuracy can be a
# scalar to be used for all variables or a 1darray of values of size
# n_variables.
cvg_relative_accuracy = 0.01
# Probability (or confidence interval) and must be between 0.0 and 1.0, and
# represents the confidence for calculation of relative halfwidths
# estimation.
cvg_p = 0.95
# The heidel_welch_number_points is the number of points Heidelberger and
# Welch's spectral method that are used to obtain the polynomial fit. The
# parameter heidel_welch_number_points determines the frequency range over
# which the fit is made.
cvg_k = 50
# If True, use FFT convolution. FFT should be preferred for long time series.
cvg_fft = True
# If `float`, should be between 0.0 and 1.0 and represent the proportion of
# the periodogram dataset to include in the test split. If `int`, represents
# the absolute number of test samples.
cvg_test_size = None
# If `float`, should be between 0.0 and 1.0 and represent the proportion of
# the preiodogram dataset to include in the train split. If `int`, represents
# the absolute number of train samples.
cvg_train_size = None
# batch size.
cvg_batch_size = 5
# A method to standardize a batched dataset.
cvg_scale = "translate_scale"
# If True, use batched data minus the scale metod centering approach.
cvg_with_centering = False
# If True, scale the batched data to scale metod scaling approach.
cvg_with_scaling = False
# if `int`, it is the last few batch points that should be ignored. if
# `float`, should be in `(0, 1)` and it is the percent of last batch points
# that should be ignored. if `None` it would be set to the `batch_size`.
cvg_ignore_end_batch = None
# Statistical inefficiency method.
cvg_si = 'statistical_inefficiency'
# The number of data points to skip in estimating ucl.
cvg_nskip = 1
# The minimum amount of correlation function to compute in estimating ucl.
# The algorithm terminates after computing the correlation time out to
# minimum_correlation_time when the correlation function first goes negative.
cvg_mct = None
# if `int`, it is the last few points that should be ignored in estimating
# ucl. if `float`, should be in `(0, 1)` and it is the percent of number of
# points that should be ignored in estimating ucl. If `None` it would be
# set to the one fourth of the total number of points.
cvg_ignore_end = None


start = 0
stop = 0
nstep = 0
initialstep = 0


def run_length_control(lmpptr, nevery, *argv):
    """Control the length of the LAMMPS simulation run.

    Arguments:
        lmpptr {pointer} -- LAMMPS pointer to a previously created LAMMPS
            object.
        nevery {int} -- use input values every this many timesteps. It
            specifies on what timesteps the input values will be used in
            order to be stored. Only timesteps that are a multiple of nevery,
            including timestep 0, will contribute values.

    Note:
        Each input value throug argv can be the result of a `compute` or
        a `fix` or the evaluation of an equal-style or vector-style `variable`.
        In each case, the `compute`, `fix`, or `variable` must produce a
        global quantity, not a per-atom or local quantity. And the global
        quantity must be a scalar, not a vector or array.

        Computes that produce global quantities are those which do not have
        the word atom in their style name. Only a few fixes produce global
        quantities.

        Variables of style equal or vector are the only ones that can be used
        as an input here. Variables of style atom cannot be used, since they
        produce per-atom values.

        Each input value through argv following the argument `lb`, or `lbound`
        and `ub`, or `ubound` must previously be defined in the input script
        as the evaluation of an equal-style `variable`.

    """
    lmp = lammps(ptr=lmpptr)

    if not isinstance(nevery, int):
        msg = 'nevery is not an `int`.'
        raise cvg.CVGError(msg)
    elif nevery < 1:
        msg = 'nevery is not a positive `int`.'
        raise cvg.CVGError(msg)

    msg = 'fix cvg_fix all vector {} '.format(nevery)

    # new keyword
    prefix = None
    ctrl_map = {}

    # Number of arguments
    arguments_map = {}
    n_arguments = 0
    _n_arguments = len(argv)
    prefix = 'v_'
    i = 0
    while i < _n_arguments:
        arg = argv[i]

        if not isinstance(arg, str):
            msg = '{} is not an `str`.'.format(str(arg))
            raise cvg.CVGError(msg)

        # The values following the argument `variable` must previously be
        # defined in the input script (`v_`).
        if arg == 'variable':
            prefix = 'v_'
            i += 1
            continue
        # The values following the argument `compute` must previously be
        # defined in the input script (`c_`).
        elif arg == 'compute':
            prefix = 'c_'
            i += 1
            continue
        # The values following the argument `fix` must previously be
        # defined in the input script (`f_`).
        elif arg == 'fix':
            prefix = 'f_'
            i += 1
            continue
        elif arg == 'lb' or arg == 'lbound':
            try:
                ctrl_name = '{}{}'.format(prefix, argv[i - 1])
            except IndexError:
                msg = 'the ctrl variable does not exist.'
                raise cvg.CVGError(msg)

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the variable\'s lower bound does not exist.'
                raise cvg.CVGError(msg)

            # lb & ub must be equal-style variable
            try:
                var_lb = lmp.extract_variable(arg, None, 0)
            except:
                msg = 'lb must be followed by an equal-style variable.'
                raise cvg.CVGError(msg)
            var_ub = None

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                ctrl_map[ctrl_name] = tuple([var_lb, var_ub])
                break

            if arg == 'ub' or arg == 'ubound':
                i += 1
                try:
                    arg = argv[i]
                except IndexError:
                    msg = 'the variable\' upper bound does not exist.'
                    raise cvg.CVGError(msg)

                # lb & ub must be equal-style variable
                try:
                    var_ub = lmp.extract_variable(arg, None, 0)
                except:
                    msg = 'ub must be followed by an equal-style variable.'
                    raise cvg.CVGError(msg)
            else:
                i -= 1

            ctrl_map[ctrl_name] = tuple([var_lb, var_ub])
            i += 1
            continue
        elif arg == 'ub' or arg == 'ubound':
            try:
                ctrl_name = '{}{}'.format(prefix, argv[i - 1])
            except IndexError:
                msg = 'the ctrl variable does not exist.'
                raise cvg.CVGError(msg)

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the variable\' upper bound does not exist.'
                raise cvg.CVGError(msg)

            # lb & ub must be equal-style variable
            # being here means that this ctrl variable has no lower bound
            var_lb = None
            try:
                var_ub = lmp.extract_variable(arg, None, 0)
            except:
                msg = 'ub must be followed by an equal-style variable.'
                raise cvg.CVGError(msg)

            ctrl_map[ctrl_name] = tuple([var_lb, var_ub])
            i += 1
            continue

        var_name = '{}{}'.format(prefix, arg)
        msg += var_name + ' '
        arguments_map[n_arguments] = var_name
        n_arguments += 1
        i += 1

    lmp.command(msg)

    if ctrl_map:
        if n_arguments == 1:
            var_name = arguments_map[0]
            msg = 'the variable "{}" is used for '.format(var_name)
            msg += 'controling the stability of the simulation to be '
            msg += 'bounded by lower and/or upper bound. It can not be '
            msg += 'used for the run length control at the same time.'
            raise cvg.CVGError(msg)
        elif n_arguments == len(ctrl_map):
            var_name = arguments_map[0]
            msg = 'the variables "{}", '.format(var_name)
            for i in range(1, n_arguments - 1):
                var_name = arguments_map[i]
                msg = '"{}", '.format(var_name)
            var_name = arguments_map[n_arguments - 1]
            msg = 'and "{}" are used for '.format(var_name)
            msg += 'controling the stability of the simulation to be '
            msg += 'bounded by lower and/or upper bounds. They can not be '
            msg += 'used for the run length control at the same time.'
            raise cvg.CVGError(msg)

    def get_trajectory(step):
        """Get trajectory vector or array of values.

        Arguments:
            step {int} -- number of steps to run the simulation.

        Returns:
            ndarray -- for a single specified value, the values are stored as
                a vector. For multiple specified values, they are stored as
                rows in an array.

        """
        global start, stop
        global nstep, initialstep

        start = stop
        stop += step

        finalstep = stop // nevery * nevery
        if finalstep > stop:
            finalstep -= nevery
        ncountmax = (finalstep - initialstep) // nevery + 1
        initialstep = finalstep + nevery

        # Run the LAMMPS simulation
        msg = 'run {}'.format(step)
        lmp.command(msg)

        if ctrl_map:
            # trajectory array
            _ndim = n_arguments - len(ctrl_map)
            trajectory = np.empty((_ndim, ncountmax), dtype=np.float64)

            # argument index in the trajectory array
            _j = 0
            for j in range(n_arguments):
                var_name = arguments_map[j]
                if var_name in ctrl_map:
                    lb, ub = ctrl_map[var_name]
                    if lb and ub:
                        for _nstep in range(nstep, nstep + ncountmax):
                            val = lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                            if val <= lb or val >= ub:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '{} {}). '.format(lb, ub)
                                msg += 'This run is unstable.'
                                raise cvg.CVGError(msg)
                        continue
                    elif lb:
                        for _nstep in range(nstep, nstep + ncountmax):
                            val = lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                            if val <= lb:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '{} ...). '.format(lb)
                                msg += 'This run is unstable.'
                                raise cvg.CVGError(msg)
                        continue
                    elif ub:
                        for _nstep in range(nstep, nstep + ncountmax):
                            val = lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                            if val >= ub:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '... {}). '.format(ub)
                                msg += 'This run is unstable.'
                                raise cvg.CVGError(msg)
                        continue
                else:
                    for i, _nstep in enumerate(range(nstep, nstep + ncountmax)):
                        trajectory[_j, i] = \
                            lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                    _j += 1
            nstep += ncountmax
            if _ndim == 1:
                return trajectory.squeeze()
            return trajectory
        else:
            if n_arguments == 1:
                trajectory = np.empty((ncountmax), dtype=np.float64)
                for i, _nstep in enumerate(range(nstep, nstep + ncountmax)):
                    trajectory[i] = lmp.extract_fix('cvg_fix', 0, 1, _nstep, 0)
                nstep += ncountmax
                return trajectory
            else:
                trajectory = np.empty((n_arguments, ncountmax),
                                      dtype=np.float64)
                for j in range(n_arguments):
                    for i, _nstep in enumerate(range(nstep, nstep + ncountmax)):
                        trajectory[j, i] = \
                            lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                nstep += ncountmax
                return trajectory

    try:
        msg = cvg.run_length_control(get_trajectory=get_trajectory,
                                     n_variables=n_arguments - len(ctrl_map),
                                     initial_run_length=cvg_initial_run_length,
                                     run_length_factor=cvg_run_length_factor,
                                     maximum_run_length=cvg_maximum_run_length,
                                     maximum_equilibration_step=cvg_maximum_equilibration_step,
                                     sample_size=cvg_sample_size,
                                     relative_accuracy=cvg_relative_accuracy,
                                     confidence_coefficient=cvg_p,
                                     heidel_welch_number_points=cvg_k,
                                     fft=cvg_fft,
                                     test_size=cvg_test_size,
                                     train_size=cvg_train_size,
                                     batch_size=cvg_batch_size,
                                     scale=cvg_scale,
                                     with_centering=cvg_with_centering,
                                     with_scaling=cvg_with_scaling,
                                     ignore_end_batch=cvg_ignore_end_batch,
                                     si=cvg_si,
                                     nskip=cvg_nskip,
                                     minimum_correlation_time=cvg_mct,
                                     ignore_end=cvg_ignore_end,
                                     fp='return')
    except Exception as e:
        msg = '{}'.format(e)
        raise cvg.CVGError(msg)

    cmd = "variable run_var string ''"
    lmp.command(cmd)

    lmp.set_variable('run_var', msg)

    cmd = 'print "${run_var}"'
    lmp.command(cmd)

    cmd = "variable run_var delete"
    lmp.command(cmd)

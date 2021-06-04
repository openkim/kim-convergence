"""Run length control example by LAMMPS."""

import numpy as np
from lammps import lammps
import convergence as cr

# Initial run length
INITIAL_RUN_LENGTH = 1000
# Run length increasing factor
RUN_LENGTH_FACTOR = 1
# The maximum run length represents a cost constraint.
MAX_RUN_LENGTH = 1000 * INITIAL_RUN_LENGTH
# The maximum number of steps as an equilibration hard limit. If the
# algorithm finds equilibration_step greater than this limit it will fail.
# For the default None, the function is using `maximum_run_length // 2` as
# the maximum equilibration step.
MAX_EQUILIBRATION_STEP = None
# Maximum number of independent samples.
MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES = None
# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
RELATIVE_ACCURACY = 0.01
ABSOLUTE_ACCURACY = None
# Probability (or confidence interval) and must be between 0.0 and 1.0, and
# represents the confidence for calculation of relative halfwidths estimation.
CONFIDENCE = 0.95
# Method to use for approximating the upper confidence limit of the mean.
UCL_METHOD = 'uncorrelated_sample'


LAMMPS_ARGUMENTS = (
    'variable',
    'compute',
    'fix',
    'lb',
    'lbound',
    'ub',
    'ubound',
    'population_mean',
    'population_std',
    'population_cdf',
    'population_args',
    'population_loc',
    'population_scale',
)

start = 0
stop = 0
nstep = 0
initialstep = 0


def run_length_control(lmpptr, nevery: int, *argv):
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

    cr.cvg_check(nevery, 'nevery', int, 1)

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
            raise cr.CVGError(msg)

        # The values following the argument `variable` must previously be
        # defined in the input script (`v_`).
        if arg == 'variable':
            prefix = 'v_'
            i += 1
            continue

        # The values following the argument `compute` must previously be
        # defined in the input script (`c_`).
        if arg == 'compute':
            prefix = 'c_'
            i += 1
            continue

        # The values following the argument `fix` must previously be
        # defined in the input script (`f_`).
        if arg == 'fix':
            prefix = 'f_'
            i += 1
            continue

        if arg in ('lb', 'lbound'):
            try:
                ctrl_name = '{}{}'.format(prefix, argv[i - 1])
            except IndexError:
                msg = 'the ctrl variable does not exist.'
                raise cr.CVGError(msg)

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the variable\'s lower bound does not exist.'
                raise cr.CVGError(msg)

            # lb & ub must be equal-style variable
            try:
                var_lb = lmp.extract_variable(arg, None, 0)
            except:
                msg = 'lb must be followed by an equal-style variable.'
                raise cr.CVGError(msg)
            var_ub = None

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                ctrl_map[ctrl_name] = tuple([var_lb, var_ub])
                break

            if arg in ('ub', 'ubound'):
                i += 1
                try:
                    arg = argv[i]
                except IndexError:
                    msg = 'the variable\' upper bound does not exist.'
                    raise cr.CVGError(msg)

                # lb & ub must be equal-style variable
                try:
                    var_ub = lmp.extract_variable(arg, None, 0)
                except:
                    msg = 'ub must be followed by an equal-style variable.'
                    raise cr.CVGError(msg)
            else:
                i -= 1

            ctrl_map[ctrl_name] = tuple([var_lb, var_ub])
            i += 1
            continue

        if arg in ('ub', 'ubound'):
            try:
                ctrl_name = '{}{}'.format(prefix, argv[i - 1])
            except IndexError:
                msg = 'the ctrl variable does not exist.'
                raise cr.CVGError(msg)

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the variable\' upper bound does not exist.'
                raise cr.CVGError(msg)

            # lb & ub must be equal-style variable
            # being here means that this ctrl variable has no lower bound
            var_lb = None
            try:
                var_ub = lmp.extract_variable(arg, None, 0)
            except:
                msg = 'ub must be followed by an equal-style variable.'
                raise cr.CVGError(msg)

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
            raise cr.CVGError(msg)

        if n_arguments == len(ctrl_map):
            var_name = arguments_map[0]
            msg = 'the variables "{}", '.format(var_name)
            for i in range(1, n_arguments - 1):
                var_name = arguments_map[i]
                msg = '"{}", '.format(var_name)
            var_name = arguments_map[-1]
            msg = 'and "{}" are used for '.format(var_name)
            msg += 'controling the stability of the simulation to be '
            msg += 'bounded by lower and/or upper bounds. They can not be '
            msg += 'used for the run length control at the same time.'
            raise cr.CVGError(msg)

    def get_trajectory(step: int) -> np.ndarray:
        """Get trajectory vector or array of values.

        Arguments:
            step (int): number of steps to run the simulation.

        Returns:
            ndarray: trajectory
                for a single specified value, the values are stored as
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
                                raise cr.CVGError(msg)
                        continue
                    elif lb:
                        for _nstep in range(nstep, nstep + ncountmax):
                            val = lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                            if val <= lb:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '{} ...). '.format(lb)
                                msg += 'This run is unstable.'
                                raise cr.CVGError(msg)
                        continue
                    elif ub:
                        for _nstep in range(nstep, nstep + ncountmax):
                            val = lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
                            if val >= ub:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '... {}). '.format(ub)
                                msg += 'This run is unstable.'
                                raise cr.CVGError(msg)
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

        if n_arguments == 1:
            trajectory = np.empty((ncountmax), dtype=np.float64)
            for i, _nstep in enumerate(range(nstep, nstep + ncountmax)):
                trajectory[i] = lmp.extract_fix('cvg_fix', 0, 1, _nstep, 0)
            nstep += ncountmax
            return trajectory

        trajectory = np.empty((n_arguments, ncountmax), dtype=np.float64)
        for j in range(n_arguments):
            for i, _nstep in enumerate(range(nstep, nstep + ncountmax)):
                trajectory[j, i] = \
                    lmp.extract_fix('cvg_fix', 0, 2, _nstep, j)
        nstep += ncountmax
        return trajectory

    try:
        msg = cr.run_length_control(
            get_trajectory=get_trajectory,
            number_of_variables=n_arguments - len(ctrl_map),
            initial_run_length=INITIAL_RUN_LENGTH,
            run_length_factor=RUN_LENGTH_FACTOR,
            maximum_run_length=MAX_RUN_LENGTH,
            maximum_equilibration_step=MAX_EQUILIBRATION_STEP,
            minimum_number_of_independent_samples=MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES,
            relative_accuracy=RELATIVE_ACCURACY,
            absolute_accuracy=ABSOLUTE_ACCURACY,
            population_mean=None,
            population_standard_deviation=None,
            population_cdf=None,
            population_args=None,
            population_loc=None,
            population_scale=None,
            confidence_coefficient=CONFIDENCE,
            confidence_interval_approximation_method=UCL_METHOD,
            heidel_welch_number_points=cr._default._DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            fft=cr._default._DEFAULT_FFT,
            test_size=cr._default._DEFAULT_TEST_SIZE,
            train_size=cr._default._DEFAULT_TRAIN_SIZE,
            batch_size=cr._default._DEFAULT_BATCH_SIZE,
            scale=cr._default._DEFAULT_SCALE_METHOD,
            with_centering=cr._default._DEFAULT_WITH_CENTERING,
            with_scaling=cr._default._DEFAULT_WITH_SCALING,
            ignore_end=cr._default._DEFAULT_IGNORE_END,
            number_of_cores=cr._default._DEFAULT_NUMBER_OF_CORES,
            si=cr._default._DEFAULT_SI,
            nskip=cr._default._DEFAULT_NSKIP,
            minimum_correlation_time=cr._default._DEFAULT_MINIMUM_CORRELATION_TIME,
            dump_trajectory=False,
            dump_trajectory_fp='convergence_trajectory.edn',
            fp='return',
            fp_format='txt')
    except Exception as e:
        msg = '{}'.format(e)
        raise cr.CVGError(msg)

    cmd = "variable run_var string ''"
    lmp.command(cmd)

    lmp.set_variable('run_var', msg)

    cmd = 'print "${run_var}"'
    lmp.command(cmd)

    cmd = "variable run_var delete"
    lmp.command(cmd)

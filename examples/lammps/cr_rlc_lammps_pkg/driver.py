"""Orchestrator: parses argv, builds correct callback, calls kim-convergence."""

from typing import Optional

from lammps import lammps  # type: ignore[import]

import kim_convergence as cr
from kim_convergence._default import (
    _DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
    _DEFAULT_FFT,
    _DEFAULT_TEST_SIZE,
    _DEFAULT_TRAIN_SIZE,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_SCALE_METHOD,
    _DEFAULT_WITH_CENTERING,
    _DEFAULT_WITH_SCALING,
    _DEFAULT_IGNORE_END,
    _DEFAULT_NUMBER_OF_CORES,
    _DEFAULT_SI,
    _DEFAULT_NSKIP,
    _DEFAULT_MINIMUM_CORRELATION_TIME,
)

from ._argument_parser import ArgumentParser
from ._trajectory_callback import build_get_trajectory


__all__ = ["Driver"]


# Initial run length
INITIAL_RUN_LENGTH: int = 1000
# Run length increasing factor
RUN_LENGTH_FACTOR: float = 1.0
# The maximum run length represents a cost constraint.
MAX_RUN_LENGTH: int = 1000 * INITIAL_RUN_LENGTH
# The maximum number of steps as an equilibration hard limit. If the
# algorithm finds equilibration_step greater than this limit it will fail.
# For the default None, the function is using `maximum_run_length // 2` as
# the maximum equilibration step.
MAX_EQUILIBRATION_STEP: Optional[int] = None
# Maximum number of independent samples.
MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES: int = 300
# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
RELATIVE_ACCURACY: float = 0.01
ABSOLUTE_ACCURACY: Optional[float] = None
# Probability (or confidence interval) and must be between 0.0 and 1.0, and
# represents the confidence for calculation of relative halfwidths estimation.
CONFIDENCE: float = 0.95
# Method to use for approximating the upper confidence limit of the mean.
UCL_METHOD: str = "uncorrelated_sample"
# if ``True``, dump the final trajectory data to a file.
DUMP_TRAJECTORY: bool = False


class Driver:
    def __init__(self, lmpptr, nevery: int, argv: tuple) -> None:
        self.lmp = lammps(ptr=lmpptr)
        self.nevery = nevery
        self.config = ArgumentParser(self.lmp, nevery, argv).parse()

    def _pop_list(self, d: dict) -> list | None:
        if not d:
            return None

        return [
            d.get(v, None)
            for v in self.config["var_names"]
            if v not in self.config["ctrl_map"]
        ]

    def go(self) -> None:
        # 1.  build callback
        get_trajectory = build_get_trajectory(
            lmp=self.lmp,
            config=self.config,
            nevery=self.nevery,
        )

        # 2.  args dict that kim-convergence will pass back
        traj_args = {"stop": 0, "nstep": 0, "initialstep": 0}

        # 3.  population lists
        p_mean = self._pop_list(self.config["population_mean"])
        p_std = self._pop_list(self.config["population_std"])
        p_cdf = self._pop_list(self.config["population_cdf"])
        p_args = self._pop_list(self.config["population_args"])
        p_loc = self._pop_list(self.config["population_loc"])
        p_scale = self._pop_list(self.config["population_scale"])

        me = self.lmp.extract_setting("world_rank")
        nprocs = self.lmp.extract_setting("world_size")
        if nprocs > 1:
            try:
                from mpi4py import MPI  # type: ignore[import]
            except ImportError as e:
                raise cr.CRError("MPI run requires mpi4py package.") from e

            comm = MPI.COMM_WORLD

        if me == 0:
            # 4.  run kim-convergence with
            try:
                report = cr.run_length_control(
                    get_trajectory=get_trajectory,
                    get_trajectory_args=traj_args,
                    number_of_variables=self.config["number_of_variables"],
                    initial_run_length=INITIAL_RUN_LENGTH,
                    run_length_factor=RUN_LENGTH_FACTOR,
                    maximum_run_length=MAX_RUN_LENGTH,
                    maximum_equilibration_step=MAX_EQUILIBRATION_STEP,
                    minimum_number_of_independent_samples=MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES,
                    relative_accuracy=RELATIVE_ACCURACY,
                    absolute_accuracy=ABSOLUTE_ACCURACY,
                    population_mean=p_mean,
                    population_standard_deviation=p_std,
                    population_cdf=p_cdf,
                    population_args=p_args,
                    population_loc=p_loc,
                    population_scale=p_scale,
                    confidence_coefficient=CONFIDENCE,
                    confidence_interval_approximation_method=UCL_METHOD,
                    heidel_welch_number_points=_DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
                    fft=_DEFAULT_FFT,
                    test_size=_DEFAULT_TEST_SIZE,
                    train_size=_DEFAULT_TRAIN_SIZE,
                    batch_size=_DEFAULT_BATCH_SIZE,
                    scale=_DEFAULT_SCALE_METHOD,
                    with_centering=_DEFAULT_WITH_CENTERING,
                    with_scaling=_DEFAULT_WITH_SCALING,
                    ignore_end=_DEFAULT_IGNORE_END,
                    number_of_cores=_DEFAULT_NUMBER_OF_CORES,
                    si=_DEFAULT_SI,  # type: ignore[arg-type]
                    nskip=_DEFAULT_NSKIP,
                    minimum_correlation_time=_DEFAULT_MINIMUM_CORRELATION_TIME,
                    dump_trajectory=DUMP_TRAJECTORY,
                    dump_trajectory_fp="kim_convergence_trajectory.edn",
                    fp="return",
                    fp_format="txt",
                )

            finally:
                if nprocs > 1:
                    # Send exit signal
                    for r in range(1, nprocs):
                        comm.send(-1, dest=r, tag=56789)

            # 5.  hand result back to LAMMPS
            self.lmp.command("variable run_var string ''")
            if hasattr(self.lmp, "set_string_variable"):
                self.lmp.set_string_variable("run_var", report)
            else:
                self.lmp.set_variable("run_var", report)
            self.lmp.command('print "${run_var}"')
            self.lmp.command("variable run_var delete")
        elif nprocs > 1:
            while True:
                nstep = comm.recv(source=0, tag=56789)
                if nstep < 0:
                    break
                self.lmp.command(f"run {nstep}")

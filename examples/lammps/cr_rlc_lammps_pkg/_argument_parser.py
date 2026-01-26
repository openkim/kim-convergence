from lammps import lammps  # type: ignore[import]

from kim_convergence import CRError, cr_check


_LAMMPS_ARGUMENTS = frozenset(
    {
        "variable",
        "compute",
        "fix",
        "lb",
        "lbound",
        "ub",
        "ubound",
        "mean",
        "population_mean",
        "std",
        "population_std",
        "cdf",
        "population_cdf",
        "args",
        "population_args",
        "loc",
        "population_loc",
        "scale",
        "population_scale",
    }
)

_PREFIX_TO_TYPE = {
    "v_": "variable",
    "c_": "compute",
    "f_": "fix",
}

_ID_DESCRIPTIONS = {
    "v_": "User-assigned name for the variable",
    "c_": "User-assigned ID for the compute",
    "f_": "User-assigned ID for the fix",
}

_BRACKETS = ("(,)", "{,}", "[,]", "()", "{}", "[]", "(", ")", "[", "]", "{", "}")


class ArgumentParser:
    def __init__(self, lmp: lammps, nevery: int, argv: tuple[str, ...]) -> None:
        self.lmp = lmp
        self.nevery = nevery
        self.argv = list(argv)

    def _next(self, i: int, emsg: str) -> tuple[int, str]:
        """Get next argument with proper index increment and error handling."""
        i += 1
        if i >= len(self.argv):
            raise CRError(emsg)
        return i, self.argv[i]

    def _next_float_or_var(self, i: int, emsg: str, name: str) -> tuple[int, float]:
        i, token = self._next(i, emsg)
        try:
            return i, float(token)
        except ValueError:
            # token can be an equal-style variable
            val = self.lmp.extract_variable(token, None, 0)
            if val is None:
                raise CRError(f"{name} must be followed by an equal-style variable.")
            return i, val

    @staticmethod
    def _missing_param(var_name: str, prefix: str, param: str) -> str:
        return f"the {var_name} {_PREFIX_TO_TYPE[prefix]}'s {param} is not provided."

    def parse(self) -> dict:
        lmp = self.lmp
        nevery = self.nevery
        argv = self.argv

        cr_check(nevery, "nevery", int, 1)

        cmd = [f"fix cr_fix all vector {nevery}"]

        # default prefix
        prefix: str = "v_"
        ctrl_map: dict = {}
        var_name: str | None = None
        var_names: list[str] = []

        # population info
        population_mean = {}
        population_std = {}
        population_cdf = {}
        population_args = {}
        population_loc = {}
        population_scale = {}

        i = 0
        number_of_arguments = len(argv)
        while i < number_of_arguments:
            arg = argv[i]

            if arg not in _LAMMPS_ARGUMENTS:
                avail = ", ".join(f'"{a}"' for a in _LAMMPS_ARGUMENTS)
                raise CRError(
                    f'Input argument "{arg}" is not recognized.\n'
                    f"Valid arguments are: [{avail}]."
                )

            if arg in ("variable", "compute", "fix"):
                # The value following the argument `variable`, `compute`, or
                # `fix` must previously (in the input script) be defined
                # (prefixed) as `v_`, `c_`, or `f_` variable respectively.
                prefix = f"{arg[0]}_"
                i, arg = self._next(
                    i,
                    f"{_ID_DESCRIPTIONS[prefix]} is not provided.",
                )
                var_name = f"{prefix}{arg}"
                var_names.append(var_name)

                cmd.append(f"{var_name}")

                i += 1
                continue

            if var_name is None:
                raise CRError(
                    "A `variable` or a `compute`, or a `fix` must previously be defined."
                )

            if arg in ("lbound", "lb"):
                i, var_lb = self._next_float_or_var(
                    i, self._missing_param(var_name, prefix, "lower bound"), "lb"
                )
                var_ub = None

                i += 1
                if i < number_of_arguments:
                    arg = argv[i]
                else:
                    ctrl_map[var_name] = tuple([var_lb, var_ub])
                    break

                if arg in ("ub", "ubound"):
                    i, var_ub = self._next_float_or_var(
                        i,
                        self._missing_param(var_name, prefix, "upper bound"),
                        "ub",
                    )
                else:
                    i -= 1

                ctrl_map[var_name] = tuple([var_lb, var_ub])

            elif arg in ("ubound", "ub"):
                # being here means that this ctrl variable has no lower bound
                var_lb = None
                i, var_ub = self._next_float_or_var(
                    i, self._missing_param(var_name, prefix, "upper bound"), "ub"
                )
                ctrl_map[var_name] = tuple([var_lb, var_ub])

            elif arg in ("population_mean", "mean"):
                i, population_mean[var_name] = self._next_float_or_var(
                    i,
                    self._missing_param(var_name, prefix, "population_mean"),
                    "population_mean",
                )

            elif arg in ("population_std", "std"):
                i, population_std[var_name] = self._next_float_or_var(
                    i,
                    self._missing_param(var_name, prefix, "population_std"),
                    "population_std",
                )

            elif arg in ("population_cdf", "cdf"):
                i, population_cdf[var_name] = self._next(
                    i, self._missing_param(var_name, prefix, "population_cdf")
                )

            elif arg in ("population_args", "args"):
                population_args[var_name] = []

                i, arg = self._next(
                    i, self._missing_param(var_name, prefix, "population_args")
                )

                arg = arg.replace(" ", "")
                for b in _BRACKETS:
                    arg = arg.replace(b, "")
                if len(arg):
                    arg = arg.split(",")

                for arg_ in arg:
                    try:
                        value = int(arg_)
                    except ValueError:
                        try:
                            value = float(arg_)
                        except ValueError:
                            raise CRError(
                                "population_args must be followed by "
                                "a list or tuple of values(s)."
                            )
                        pass

                    population_args[var_name].append(value)

            elif arg in ("population_loc", "loc"):
                i, population_loc[var_name] = self._next_float_or_var(
                    i,
                    self._missing_param(var_name, prefix, "population_loc"),
                    "population_loc",
                )

            elif arg in ("population_scale", "scale"):
                i, population_scale[var_name] = self._next_float_or_var(
                    i,
                    self._missing_param(var_name, prefix, "population_scale"),
                    "population_scale",
                )

            i += 1

        if ctrl_map:
            n_vars = len(var_names)
            if n_vars == 1:
                raise CRError(
                    f'the variable "{var_names[0]}" is used for '
                    "controling the stability of the simulation to be bounded "
                    "by lower and/or upper bound. It can not be used for the "
                    "run length control at the same time."
                )

            n_ctrl = len(ctrl_map)
            if n_vars == n_ctrl:
                msg = ", ".join(f'"{argm}"' for argm in var_names[:-1])
                raise CRError(
                    f'the variables {msg} and "{var_names[-1]}" are used '
                    "for controling the stability of the simulation to be "
                    "bounded by lower and/or upper bounds. They can not be "
                    "used for the run length control at the same time."
                )

        # Run the LAMMPS script
        lmp.command(" ".join(cmd))

        return {
            "number_of_variables": len(var_names) - len(ctrl_map),
            "var_names": var_names,
            "ctrl_map": ctrl_map,
            "population_mean": population_mean,
            "population_std": population_std,
            "population_cdf": population_cdf,
            "population_args": population_args,
            "population_loc": population_loc,
            "population_scale": population_scale,
        }

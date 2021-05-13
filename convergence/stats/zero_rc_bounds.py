"""Zero finding class.

This code is adapted from SciPy.
It is completely revised and rewritten by Yaser Afshar.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2001-2002 Enthought, Inc.
              2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from copy import deepcopy
from math import fabs

from .zero_rc import ZERO_RC
from convergence import CVGError

__all__ = [
    'ZERO_RC_BOUNDS',
]


class ZERO_RC_BOUNDS():
    """Bound zero finding class by reverse communication."""

    def __init__(self,
                 small: float,
                 big: float,
                 abs_step: float,
                 rel_step: float,
                 step_mul: float,
                 *,
                 abs_tol=1.0e-50,
                 rel_tol=1.0e-8):
        """Initialize parameters.

        Args:
            small (float): The left endpoint of the interval.
            big (float): The right endpoint of the interval.
            abs_step (float): Absolute tolerance to determine the initial
                step size in the search.
            rel_step (float): Relative tolerance to determine the initial
                step size in the search.
            step_mul (float): Step multiplyer, when a step doesn't bound
                the zero.
            abs_tol (float, optional): Absolute error tolerance.
                (default: 1.0e-50)
            rel_tol (float, optional): Relative error tolerance.
                (default: 1.0e-8)

        """
        self.fx_small = 0.0
        self.step = 0.0
        self.xhi = 0.0
        self.xlb = 0.0
        self.xlo = 0.0
        self.xsave = 0.0
        self.xub = 0.0
        self.yy = 0.0
        self.index = 0
        self.qincr = False
        self.small = deepcopy(small)
        self.big = deepcopy(big)
        self.abs_step = deepcopy(abs_step)
        self.rel_step = deepcopy(rel_step)
        self.step_mul = deepcopy(step_mul)
        self.abs_tol = deepcopy(abs_tol)
        self.rel_tol = deepcopy(rel_tol)
        self.z = None

    def zero(self, status: int, x: float, fx: float):
        """Bounds the zero of the function.

        Bounds the zero of the function and finds zero of the function by
        reverse communication.

        f must be a monotone function, otherwise the results are undefined.
        If f is an increasing monotone, then the result is bound by
        ``[f(x-tolerance(x)) f(x+tolerance(x))]``.
        If f is a decreasing monotone, then the result is bound by
        ``[f(x+tolerance(x)) f(x-tolerance(x))]``.
        Where ``tolerance(x) = Maximum(abs_tol, rel_tol * |x|)``.

        Args:
            status (int): Status.
                If status set to 0, the value of other parameters will be
                ignored on the call.
            x (float): The input value at which function f is to be
                evaluated.
            fx (float): Function f evaluated at x.

        Returns:
            int, float: status, x.
                Where, the status = 0 on return means it has finished without
                error. The status = 1 on return, means the function needs to
                be evaluated.

        """
        if status == 0:
            monotone = (self.small <= x) and (x <= self.big)
            if not monotone:
                msg = 'small={}, x={}, big={} '.format(
                    self.small, x, self.big)
                msg += 'are not monotone.'
                raise CVGError(msg)

            self.xsave = deepcopy(x)
            self.index = 1
            return 1, self.small

        if self.index == 1:
            self.fx_small = deepcopy(fx)
            self.index = 2
            return 1, self.big
        elif self.index == 2:
            fx_big = deepcopy(fx)

            self.qincr = fx_big > self.fx_small

            if self.qincr:
                if self.fx_small <= 0.0:
                    if fx_big >= 0.0:
                        self.step = max(self.abs_step,
                                        self.rel_step * fabs(self.xsave))
                        self.index = 3
                        return 1, self.xsave
                    msg = 'Answer x = {}, appears to be higher '.format(x)
                    msg += 'than the highest search bound = '
                    msg += '{}.\n'.format(self.big)
                    msg += 'It means that the stepping search terminated '
                    msg += 'unsuccessfully at the highest search bound.'
                    raise CVGError(msg)
                msg = 'Answer x = {}, appears to be lower than '.format(x)
                msg += 'the lowest search bound = {}.\n'.format(self.small)
                msg += 'It means that the stepping search terminated '
                msg += 'unsuccessfully at the lowest search bound.'
                raise CVGError(msg)

            if self.fx_small >= 0.0:
                if fx_big <= 0.0:
                    self.step = max(self.abs_step,
                                    self.rel_step * fabs(self.xsave))
                    self.index = 3
                    return 1, self.xsave
                msg = 'Answer x = {}, appears to be higher than '.format(x)
                msg += 'the highest search bound = {}.\n'.format(self.big)
                msg += 'It means that the stepping search terminated '
                msg += 'unsuccessfully at the highest search bound.'
                raise CVGError(msg)
            msg = 'Answer x = {}, appears to be lower than the '.format(x)
            msg += 'lowest search bound = {}.\n'.format(self.small)
            msg += 'It means that the stepping search terminated '
            msg += 'unsuccessfully at the lowest search bound.'
            raise CVGError(msg)
        elif self.index == 3:
            self.yy = deepcopy(fx)

            if self.yy == 0.0:
                return 0, x

            qup = (self.qincr and (self.yy < 0.0)) or \
                (not self.qincr and (self.yy > 0.0))

            if qup:
                self.xlb = deepcopy(self.xsave)
                self.xub = min(self.xlb + self.step, self.big)
                self.index = 4
                return 1, self.xub

            self.xub = deepcopy(self.xsave)
            self.xlb = max(self.xub - self.step, self.small)
            self.index = 5
            return 1, self.xlb
        elif self.index == 4:
            self.yy = deepcopy(fx)
            qbdd = (self.qincr and (self.yy >= 0.0)) or \
                (not self.qincr and (self.yy <= 0.0))
            qlim = self.xub >= self.big
            qcond = qbdd or qlim

            if not qcond:
                self.step *= self.step_mul
                self.xlb = deepcopy(self.xub)
                self.xub = min(self.xlb + self.step, self.big)
                self.index = 4
                return 1, self.xub

            if qlim and not qbdd:
                msg = 'Answer x = {}, appears to be higher '.format(self.big)
                msg += 'or equal the highest search bound = '
                msg += '{}.\n'.format(self.big)
                msg += 'It means that the stepping search terminated '
                msg += 'unsuccessfully at the highest search bound.'
                raise CVGError(msg)

            # Create an instance of ZERO_RC
            self.z = ZERO_RC(self.xlb,
                             self.xub,
                             abs_tol=self.abs_tol,
                             rel_tol=self.rel_tol)

            status, x, self.xlo, self.xhi = self.z.zero(0,
                                                        x, fx,
                                                        self.xlo,
                                                        self.xhi)

            if status == 1:
                self.index = 6
                return 1, x

            while True:
                if status == 1:
                    status, x, self.xlo, self.xhi = self.z.zero(status,
                                                                x, fx,
                                                                self.xlo,
                                                                self.xhi)
                    if status == 1:
                        self.index = 6
                        return 1, x
                else:
                    return 0, self.xlo
        elif self.index == 5:
            self.yy = deepcopy(fx)
            qbdd = (self.qincr and (self.yy <= 0.0)) or \
                (not self.qincr and (self.yy >= 0.0))
            qlim = self.xlb <= self.small
            qcond = qbdd or qlim

            if qcond:
                if qlim and not qbdd:
                    msg = 'Answer x = {}, appears to be '.format(self.small)
                    msg += 'lower than or equal to the lowest search bound '
                    msg += '= {}.\n'.format(self.small)
                    msg += 'It means that the stepping search terminated '
                    msg += 'unsuccessfully at the lowest search bound.'
                    raise CVGError(msg)

                # Create an instance of ZERO_RC
                self.z = ZERO_RC(self.xlb,
                                 self.xub,
                                 abs_tol=self.abs_tol,
                                 rel_tol=self.rel_tol)

                status, x, self.xlo, self.xhi = self.z.zero(0,
                                                            x, fx,
                                                            self.xlo,
                                                            self.xhi)

                if status == 1:
                    self.index = 6
                    return 1, x

                while True:
                    if status == 1:
                        status, x, self.xlo, self.xhi = self.z.zero(status,
                                                                    x, fx,
                                                                    self.xlo,
                                                                    self.xhi)
                        if status == 1:
                            self.index = 6
                            return 1, x
                    else:
                        return 0, self.xlo
            else:
                self.step = self.step_mul * self.step
                self.xub = deepcopy(self.xlb)
                self.xlb = max(self.xub - self.step, self.small)

            self.index = 5
            return 1, self.xlb
        elif self.index == 6:
            while True:
                if status == 1:
                    status, x, self.xlo, self.xhi = self.z.zero(status,
                                                                x, fx,
                                                                self.xlo,
                                                                self.xhi)
                    if status == 1:
                        self.index = 6
                        return 1, x
                else:
                    return 0, self.xlo
        else:
            msg = 'Wrong index number = {}.\n'.format(self.index)
            msg += 'This function should be called with zero status for the '
            msg += 'first time.'
            raise CVGError(msg)

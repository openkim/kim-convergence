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
from math import copysign, fabs, nan

from convergence import CVGError

__all__ = [
    'ZERO_RC',
]


class ZERO_RC():
    """Zero finding class by reverse communication."""

    def __init__(self,
                 xlo: float,
                 xhi: float,
                 *,
                 abs_tol: float = 1.0e-50,
                 rel_tol: float = 1.0e-8):
        """Initialize parameters.

        Args:
            xlo (float): Lower inverval bounds.
            xhi (float): Upper inverval bounds.
            abs_tol (float, optional): Absolute error tolerance.
                (default: 1.0e-50)
            rel_tol (float, optional): Relative error tolerance.
                (default: 1.0e-8)

        """
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
        self.fa = 0.0
        self.fb = 0.0
        self.fc = 0.0
        self.fd = 0.0
        self.m = 0.0
        self.mb = 0.0
        self.p = 0.0
        self.q = 0.0
        self.tol = 0.0
        self.w = 0.0
        self.zx = 0.0
        self.ext = 0
        self.index = 0
        self.first = False
        self.xxlo = deepcopy(xlo)
        self.xxhi = deepcopy(xhi)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def zero(self, status: int, x: float, fx: float, xlo: float, xhi: float):
        """Perform the zero finding.

        Args:
            status {int}: Status.
                If status set to 0, the value of other parameters will be
                ignored on the call.
            x (float): The input value at which function f is to be evaluated.
            fx (float): Function f evaluated at x.
            xlo (float): Interval bounds.
            xhi (float): Interval bounds.

        Returns:
            int, float, float, float: status, x, xlo, xhi.
                The status = 0 on return means it has finished without error,
                and ``[xlo, xhi]`` bounds the answer.
                The status = 1 on return, means the function needs to be
                evaluated.
                The status = -1 on return, means an error happened.

        """
        if status <= 0:
            self.b = deepcopy(self.xxlo)
            self.index = 1
            return 1, self.xxlo, self.xxlo, self.xxhi

        if self.index == 1:
            self.a = deepcopy(xhi)
            self.fb = deepcopy(fx)
            self.index = 2
            return 1, xhi, xhi, xhi
        elif self.index == 2:
            if self.fb < 0.0 and fx < 0.0:
                return -1, x, xlo, xhi

            if self.fb > 0.0 and fx > 0.0:
                return -1, x, xlo, xhi

            self.c = deepcopy(self.a)
            self.fa = deepcopy(fx)
            self.fc = deepcopy(self.fa)
            self.first = True
            self.ext = 0
        elif self.index == 3:
            self.fb = deepcopy(fx)

            if self.fc * self.fb > 0.0:
                self.c = deepcopy(self.a)
                self.fc = deepcopy(self.fa)
                self.ext = 0

            if self.w == self.mb:
                self.ext = 0
            else:
                self.ext += 1
        else:
            msg = 'Wrong index number={}.\n'.format(self.index)
            msg += 'This function should be called with zero status for the '
            msg += 'first time.'
            raise CVGError(msg)

        if fabs(self.fc) < fabs(self.fb):
            if self.c != self.a:
                self.d = deepcopy(self.a)
                self.fd = deepcopy(self.fa)
            self.a = deepcopy(self.b)
            self.fa = deepcopy(self.fb)
            xlo = deepcopy(self.c)
            self.b = deepcopy(xlo)
            self.fb = deepcopy(self.fc)
            self.c = deepcopy(self.a)
            self.fc = deepcopy(self.fa)

        # E = max(self.abs_tol, self.rel_tol * abs(x))
        #
        # When abs(x) is larger than abs_tol/rel_tol, then only the relative
        # error tolerance is important, and whether E/abs(x) < rel_tol.
        # Thus E/abs(x) is an approximation of the relative error as
        # abs(x-x0)/abs(x0). It controls the number of significant digits.
        #
        # However, when abs(x) is smaller than abs_tol/rel_tol, only the
        # absolute error tolerance is used, so the error test in that case is
        # E < abs_tol, which is approximately abs(x-x0) < abs_tol.
        #
        # When the true solution is zero the relative error is undefined. In
        # other words, when the solution component gets small, it needs to
        # switch to something besides relative error control.
        self.tol = 0.5 * max(self.abs_tol, self.rel_tol * fabs(xlo))
        self.m = (self.c + self.b) * 0.5
        self.mb = self.m - self.b

        if fabs(self.mb) <= self.tol:
            xhi = deepcopy(self.c)
            qrzero = (self.fc >= 0.0 and self.fb <= 0.0) or \
                (self.fc < 0.0 and self.fb >= 0.0)
            if qrzero:
                return 0, x, xlo, xhi
            else:
                return -1, x, xlo, xhi

        if self.ext > 3:
            self.w = deepcopy(self.mb)
            self.d = deepcopy(self.a)
            self.fd = deepcopy(self.fa)
            self.a = deepcopy(self.b)
            self.fa = deepcopy(self.fb)
            self.b += self.w
            self.index = 3
            return 1, self.b, self.b, xhi

        self.tol = copysign(self.tol, self.mb)
        self.p = (self.b - self.a) * self.fb

        if self.first:
            self.first = False
            self.q = self.fa - self.fb
        else:
            db = self.d - self.b
            fdfb = self.fd - self.fb
            if (db != 0) or (fdfb != 0):
                fda = (self.fd - self.fa) / (self.d - self.a)
                fdb = fdfb / db
                self.p = fda * self.p
                self.q = fdb * self.fa - fda * self.fb
            else:
                self.p = nan
                self.q = nan

        if self.p < 0.0:
            self.p = -self.p
            self.q = -self.q

        if self.ext == 3:
            self.p *= 2.0

        if (self.p == 0.0) or (self.p <= (self.q * self.tol)):
            if self.p < (self.mb * self.q):
                self.w = self.p / self.q
            else:
                self.w = deepcopy(self.mb)
        else:
            self.w = deepcopy(self.tol)

        self.d = deepcopy(self.a)
        self.fd = deepcopy(self.fa)
        self.a = deepcopy(self.b)
        self.fa = deepcopy(self.fb)
        self.b += self.w
        self.index = 3
        return 1, self.b, self.b, xhi

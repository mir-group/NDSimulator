"""
Metadynamics
~~~~~~~~~~~~

This module implements the bias/penalty functions applied
on the potential landscape during metadynamics

It deposit gaussian functions with a constant frequency,
before the first half step of intergration or MC trial
All deposited functions will penalize the system,
with additional forces and energy

Only 2 collective variables are accepted

Only well-tempered metadynamics is implemented.
The width of the gaussian function is fixed while the height
of the gaussian function changes depending on whether a lot
of local gaussian has already been deposited.

"""

import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB
from math import floor


class Gaussian(AllData):
    """The gaussian_bias object implements gaussian shape bias.

    This object does not directly modify atom energy
    and forces. It only computes them and return them in the
    force() function.

    Example:

        fix = Gaussian_Bias()
        ... ...
        pointer = all_data(run=self)
        pointer.atoms = atoms
        pointer.fix = fix
        ... ...
        fix.initialize(pointer)


    """

    def __init__(
        self,
        w=None,
        sigma=None,
        dep_freq=10,
        biasf=1.0,
        adapsig=False,
        adapsig_geo=False,
        run=None,
    ):
        self.w = w
        self.sigma = sigma
        self.dep_freq = dep_freq
        self.biasf = biasf
        self.adapsig = adapsig
        self.adapsig_geo = adapsig_geo
        super(Gaussian, self).__init__(run)

    def initialize(self, pointer):

        AllData.__init__(self, run=pointer)

        #: the dimension of the original space
        self.colvardim = self.colvar.colvardim

        if (self.adapsig) and (self.adapsig_geo is False):
            self.stat.track_everything()

        if isinstance(self.sigma, float):
            if self.colvardim > 1:
                self.sigma = np.array([self.sigma for i in range(self.colvardim)])
        else:
            self.sigma = np.array(self.sigma)
            if self.sigma.shape[0] != self.colvardim:
                raise NameError("wrong input shape of gaussian width sigma")

        #: the initial height of the gaussian bias
        self._winit = self.w

        #: the initial width of the gaussian bias
        #: an numpy array with dimensions of [colvardim, ]
        self._sigma = self.sigma
        self._sigma2 = self._sigma ** 2
        self._invsigma2 = 1.0 / self._sigma2
        self.adapsig = self.adapsig
        self.adapsig_geo = self.adapsig_geo
        self.tao_d = np.min([self.dep_freq * 50, 100])

        if self.adapsig:
            if isinstance(self._invsigma2, list):
                self._invsigma2 = self._invsigma2[0]

        #: the bias factor for the well-tempered metadynamics
        self._biasf = self.biasf
        self._dtemp = (self._biasf - 1) * self.temperature
        self._kBdtemp = kB * self._dtemp

        #: deposition frequency and the number of deposited gaussians
        self._dep_freq = self.dep_freq
        self._dep_count = 0

        #: center of all the gaussian functions
        #: a numpy array with the dimension of [_dep_count, colvardim]
        self._history = None
        self._sigma_history = None

        #: heights of all the gaussian functions
        #: a numpy array with the dimension of [_dep_count, ]
        self._w = None

        #: whether the deposited hill has been added to
        #: the summation map
        #: an array of bool values with the dimension of [_dep_count, ]
        self._dep_projection = None
        self._dep_dump = None

        #: the current energy computed by the
        self.VR = 0

        #: the sum map of hills
        self.projected_map = None

    def update(self, step, time):
        """gaussian function will be deposited
        when the time step is a multiple of the _dep_freq
        """
        count = floor(time / self._dep_freq) + 1
        if count > self._dep_count:
            self._deposite()

            atoms = self.atoms
            atoms.pe, atoms.forces = self.potential.compute()
            atoms.biase = 0
            atoms.fixf = np.zeros(atoms.forces.shape)
            for fix in self.fixes:
                fix_V, fix_f = fix.compute()
                atoms.fixf += fix_f
                atoms.biase += fix_V
            atoms.totale = atoms.ke + atoms.pe + atoms.biase

    def _deposite(self):
        """deposite a gaussian function"""

        col = self.atoms.colv
        x = self.atoms.positions

        #: if it is the first time to deposit a function
        #: allocate the history array
        #: otherwise, append them
        if self._history is None:
            self._history = np.array(col).reshape(1, self.colvardim)
            self._w = np.array([self._winit])
            self._dep_projection = [False]
            self._dep_dump = [False]
            if self.adapsig:
                if self.adapsig_geo:
                    jacob = self.colvar.jacobian(x)
                    self._sigma_history = [(jacob.dot(jacob.T))]
                else:
                    stat = self.stat
                    n = len(stat.colv)
                    tao_d = self.tao_d
                    if n <= tao_d:
                        colv = np.array(stat.colv)
                    else:
                        colv = np.array(stat.colv[-tao_d:-1])
                    n0 = colv.shape[0]
                    avg_col = np.average(colv, axis=0)
                    dcol = colv - avg_col
                    if np.min(np.abs(dcol)) < 1e-10:
                        sigma2 = np.ones([self.colvardim, self.colvardim])
                    else:
                        exp = np.diag(np.exp(-(n0 - np.arange(n0)) / float(n0)))
                        sigma2 = (dcol.T.dot(exp)).dot(dcol)
                        sigma2 = 1 / float(n0) * sigma2
                    self._sigma_history = [np.copy(sigma2)]
        else:
            VR = self.VR
            self._history = np.vstack([self._history, col])
            self._w = np.append(self._w, self._winit * np.exp(-VR / (self._kBdtemp)))
            self._dep_projection.append(False)
            self._dep_dump.append(False)
            if self.adapsig:
                if self.adapsig_geo:
                    jacob = self.colvar.jacobian(x)
                    self._sigma_history += [(jacob.dot(jacob.T)) * self._sigma2]
                else:
                    stat = self.stat
                    n = len(stat.colv)
                    tao_d = self.tao_d
                    if n <= tao_d:
                        colv = np.array(stat.colv)
                    else:
                        colv = np.array(stat.colv[-tao_d:])
                    n0 = colv.shape[0]
                    avg_col = np.average(colv, axis=0)
                    dcol = colv - avg_col
                    print("colv", colv)
                    if np.min(np.abs(dcol)) < 1e-10:
                        sigma2 = np.identity(self.colvardim)
                    else:
                        exp = np.diag(np.exp(-(n0 - np.arange(n0)) / float(n0)))
                        sigma2 = (dcol.T.dot(exp)).dot(dcol)
                        sigma2 = 1 / float(n0) * sigma2
                    self._sigma_history += [np.copy(sigma2)]
        self._dep_count += 1
        # print("deposit a function on ", col, self._w[-1])

    def compute(self, x0=None, col0=None):
        """compute the energy and force from gaussian function
        the energy equation will be called as well
        """
        if x0 is None:
            #: shape (ndim, )
            x = self.atoms.positions
            #: shape (colvardim, )
            col = self.atoms.colv
        else:
            x = x0
            col = col0

        #: jacobian of the collective variables
        #: numpy array with the dimension of [ndim, colvardim]
        jacobian = self.colvar.jacobian(x)

        #: numpy array with the dimension of [_dep_count, colvardim]
        history = self._history

        if (history is not None) and (self.adapsig is False):
            #: get (_dep_count, ) array

            #: shape (_dep_count, ndim)
            dcolv = col - history

            dcolv2 = dcolv ** 2

            dcolv_sigma2 = dcolv * self._invsigma2

            #: shape (_dep_count)
            exponent = np.sum(dcolv2 / 2.0 * self._invsigma2, axis=1)

            #: shape (_dep_count)
            VRarray = self._w * np.exp(-exponent)

            V = np.sum(VRarray)
            #: prefactor for each peak
            #: shape (_dep_count, colvardim)

            #: get (_dep_count, ) array
            f = VRarray.dot(dcolv_sigma2)
            f = f.dot(jacobian)

            if len(f.shape) == 2 and f.shape[0] == 1:
                f = f.reshape([-1])

        elif (history is not None) and (self.adapsig):
            V = 0
            f = np.zeros(self.ndim)
            for idg in range(self._dep_count):

                sigma2 = self._sigma_history[idg]
                invsigma2 = 1.0 / sigma2

                dcolv = (col - history[idg, :]).reshape([1, self.colvardim])
                dcolv2 = dcolv.T.dot(dcolv)

                #: shape (_dep_count)
                exponent = dcolv2 * invsigma2 / 2.0

                #: shape (_dep_count)
                VR = self._w[idg] * np.sum(np.exp(-exponent))

                pref = dcolv.dot(invsigma2)

                V += VR
                #: prefactor for each peak
                #: shape (_dep_count, colvardim)

                #: get (_dep_count, ) array
                f0 = VR * pref
                f += f0.dot(jacobian).flat
        else:
            f = np.zeros(x.shape)
            V = 0.0
        if x0 is None:
            self.VR = V
        return V, f

    def force(self, x0=None, col0=None):
        """compute the force from gaussian function
        the energy equation will be called as well
        """
        if x0 is None:
            x = self.atoms.positions
            col = self.atoms.colv
        else:
            x = x0
            col = col0

        jacobian = self.colvar.jacobian(x)
        history = self._history

        if (history is not None) and (self.adapsig is False):

            dcolv = col - history
            dcolv_sigma2 = dcolv * self._invsigma2
            exponent = np.sum(dcolv ** 2 / 2.0 * self._invsigma2, axis=1)
            VRarray = self._w * np.exp(-exponent)
            f = VRarray.dot(dcolv_sigma2)
            f = f.dot(jacobian)
            return f
        elif (history is not None) and (self.adapsig):
            V = 0
            f = np.zeros(self.ndim)
            for idg in range(self._dep_count):
                sigma2 = self._sigma_history[idg]
                invsigma2 = 1 / sigma2
                dcolv = (col - history[idg, :]).reshape([1, self.colvardim])
                dcolv2 = dcolv.T.dot(dcolv)
                exponent = dcolv2 * invsigma2 / 2.0
                VR = np.sum(self._w[idg] * np.exp(-exponent))
                pref = dcolv.dot(invsigma2) * self._invsigma2
                V += VR
                f0 = VR * pref
                f += f0.dot(jacobian)
        else:
            return np.zeros(x.shape)

    def energy(self, x0=None, col0=None):

        if col0 is None:
            col = self.atoms.colv
        else:
            col = col0

        history = self._history
        if (history is not None) and (self.adapsig is False):
            dcolv = col - history
            exponent = np.sum(dcolv ** 2 / 2.0 * self._invsigma2, axis=1)
            VRarray = self._w * np.exp(-exponent)
            V = np.sum(VRarray)
        elif (history is not None) and (self.adapsig):
            V = 0
            for idg in range(self._dep_count):
                sigma2 = self._sigma_history[idg]
                invsigma2 = 1 / sigma2
                dcolv = (col - history[idg, :]).reshape([1, self.colvardim])
                dcolv2 = dcolv.T.dot(dcolv)
                exponent = dcolv2 * invsigma2 / 2.0
                VR = np.sum(self._w[idg] * np.exp(-exponent))
                V += VR
        else:
            V = 0
        if col0 is None:
            self.VR = V
        return V

    def projection(self, X, Y):
        """sum up the gaussian hills and return a NXN matrix
        for the computed free energy landscape
        input X, Y matrix for each point
        the X and Y are taken along the colvar1 and colvar2 directions
        only add the new hills that weren't added before
        """

        history = self._history  # shape (npeak, ndim)

        if self.projected_map is None:
            self.projected_map = np.zeros(X.shape)

        if self.colvardim == 2:
            if self.adapsig is False:
                if self._dep_projection is not None:
                    for (idx, value) in enumerate(self._dep_projection):
                        # print("idx value", idx, value)
                        if value is False:
                            exponent_x = (
                                -((X - history[idx, 0]) ** 2) / self._sigma2[0] / 2.0
                            )
                            exponent_y = (
                                -((Y - history[idx, 1]) ** 2) / self._sigma2[1] / 2.0
                            )
                            v = self._w[idx] * np.exp(exponent_x + exponent_y)
                            # print("v", np.max(v), np.min(v))
                            self.projected_map += v
                            self._dep_projection[idx] = True
            else:
                if self._dep_projection is not None:
                    for (idg, value) in enumerate(self._dep_projection):
                        if value is False:
                            sigma2 = self._sigma_history[idg]
                            invsigma2 = 1 / sigma2
                            # print(self._sigma_history, sigma2)
                            dcolv_x = X - history[idg, 0]
                            dcolv_y = Y - history[idg, 1]
                            dcolv_x2 = dcolv_x * dcolv_x
                            dcolv_y2 = dcolv_y * dcolv_y
                            dcolv_xy = dcolv_y * dcolv_x
                            exponent_xx = -dcolv_x2 * invsigma2[0, 0] / 2.0
                            exponent_xy = -dcolv_xy * invsigma2[0, 1]
                            exponent_yy = -dcolv_y2 * invsigma2[1, 1] / 2.0
                            exponent = exponent_xx + exponent_xy + exponent_yy
                            v = self._w[idg] * np.exp(exponent)
                            self.projected_map += v
                            self._dep_projection[idg] = True
        elif self.colvardim == 1:
            if self.adapsig is False:
                if self._dep_projection is not None:
                    for (idx, value) in enumerate(self._dep_projection):
                        # print("idx value", idx, value)
                        if value is False:
                            exponent_x = (
                                -((X - history[idx, 0]) ** 2) * self._invsigma2[0] / 2.0
                            )
                            v = self._w[idx] * np.exp(exponent_x)
                            # print("v", np.max(v), np.min(v))
                            self.projected_map += v
                            self._dep_projection[idx] = True
            else:
                if self._dep_projection is not None:
                    for (idg, value) in enumerate(self._dep_projection):
                        sigma2 = self._sigma_history[idg]
                        invsigma2 = 1 / sigma2
                        dcolv = X - history[idg, :]
                        dcolv2 = dcolv * dcolv
                        exponent = dcolv2 * invsigma2 / 2.0
                        v = self._w[idg] * np.exp(-exponent)
                        self.projected_map += v.flat
                        self._dep_projection[idg] = True

        return self.projected_map

    def dump_data(self):
        """sum up the gaussian hills and return a NXN matrix
        for the computed free energy landscape
        input X, Y matrix for each point
        the X and Y are taken along the colvar1 and colvar2 directions
        only add the new hills that weren't added before
        """
        history = self._history  # shape (npeak, ndim)

        line = ""
        if self._dep_dump is not None:
            for (idx, value) in enumerate(self._dep_dump):
                if value is False:
                    line += str(idx) + " "
                    line += " ".join(map(str, history[idx, :])) + " "
                    if self.adapsig:
                        line += " ".join(map(str, self._sigma_history[idx].flat)) + " "
                    else:
                        line += " ".join(map(str, self._sigma.flat)) + " "
                    line += str(self._w[idx]) + "\n"
                    self._dep_dump[idx] = True
        return line

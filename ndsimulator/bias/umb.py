import numpy as np
from ndsimulator.data import AllData
from ndsimulator.constant import kB


class Harmonic(AllData):
    """The umbrella_bias object implements parabola-shape bias."""

    def __init__(self, k, r0, run=None):
        self._K = k
        self._R0 = np.array(r0)
        AllData.__init__(self, run)

    def initialize(self, pointer):
        AllData.__init__(self, run=pointer)
        # internal parameters
        self.VR = 0
        assert self._R0.shape[0] == self.colvar.colvardim

    def update(self, step, time):
        pass

    def compute(self, x0=None, col0=None):
        R = self.atoms.colv if col0 is None else col0
        x = self.atoms.positions if x0 is None else x0
        # should be a [ncolvar, ndim] matrix
        jacobian = self.colvar.jacobian(x)
        # should be a constant TO DO, K can also be a [ncolvar, ncolvar] matrix
        f = -self._K * (R - self._R0)
        f = f.reshape([1, -1])
        f = f.dot(jacobian)
        V = 1 / 2.0 * np.sum(self._K * (R - self._R0) ** 2)
        if col0 is None:
            self.VR = V

        return V, f.reshape(
            [
                -1,
            ]
        )

    def force(self, x0=None, col0=None):
        if col0 is None:
            R = self.atoms.colv
        else:
            R = col0
        if x0 is None:
            x = self.atoms.positions
        else:
            x = x0
        jacobian = self.colv.jacobian(x)
        # TO DO, K can also be a [ncolvar, ncolvar] matrix
        force = -self._K * (R - self._R0)
        force = force.dot(jacobian)
        return force

    def energy(self, x0=None, col0=None):
        if col0 is None:
            R = self.atoms.colv
        else:
            R = col0
        # TO DO, K can also be a [ncolvar, ncolvar] matrix
        dR = R - self._R0
        V = 1 / 2.0 * np.sum(self._K * (R - self._R0) ** 2)
        if col0 is None:
            self.VR = V
        return V

    def projection(self, X, Y=None):
        if Y is None:
            return 1 / 2.0 * self._K * ((X - self._R0) ** 2)
        else:
            return 1 / 2.0 * self._K * ((X - self._R0[0]) ** 2 + (Y - self._R0[1]) ** 2)

    def dump_data(self):
        line = "{}\n".format(self.VR)

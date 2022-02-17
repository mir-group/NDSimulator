import numpy as np
from ndsimulator.potentials.potential import Potential


class Flat2d(Potential):
    ndim = 2

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        return 0, np.zeros(x.shape)

    def projection(self, X, Y):
        return np.zeros(X.shape)


class Flat1d(Potential):
    ndim = 1

    def compute(self, x=None):
        return 0, np.zeros(1)

    def projection(self, X):
        return np.zeros(X)

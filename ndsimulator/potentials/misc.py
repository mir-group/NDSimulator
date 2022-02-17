import numpy as np
from ndsimulator.constant import kcal, escale, lscale
from ndsimulator.data import AllData
from ndsimulator.potentials.potential import Potential


class Gaussian2d(Potential):
    ndim = 2
    A = -200.0 * kcal * escale
    a = -1.0 / (lscale ** 2)
    c = -10.0 / (lscale ** 2)
    x0 = 20.0
    y0 = 20.0

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        f = np.zeros(x.shape)
        tmp = self.A * np.exp(
            self.a * (x[0] - self.x0) ** 2 + self.c * (x[1] - self.y0) ** 2
        )
        V = tmp
        f[0] = -tmp * 2 * self.a * (x[0] - self.x0)
        f[1] = -tmp * 2 * self.c * (x[1] - self.y0)
        return V, f

    def projection(self, X, Y):
        V = self.A * np.exp(self.a * (X - self.x0) ** 2 + self.c * (Y - self.y0) ** 2)
        return V


class Gaussian(Potential):
    ndim = 2

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        sigma2 = 100  # 0.01
        V1 = -np.exp(-((x[0] - 0.5) ** 2 + x[1] ** 2) / sigma2)
        f1 = V1 * 2 * x / sigma2
        V2 = -np.exp(-((x[0] + 0.5) ** 2 + x[1] ** 2) / sigma2)
        f2 = V2 * 2 * x / sigma2
        return V1 + V2, f1 + f2

    def projection(self, X, Y):
        sigma2 = 100  # 0.01
        V1 = -np.exp(-((X - 0.5) ** 2 + Y ** 2) / sigma2)
        V2 = -np.exp(-((X + 0.5) ** 2 + Y ** 2) / sigma2)
        return V1 + V2


class Constant(Potential):
    ndim = None

    def compute(self, x=None):
        return 0, np.zeros(self.ndim)

    def projection(self, X, Y):
        return np.zeros(X.shape)

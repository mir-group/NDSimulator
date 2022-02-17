import numpy as np
from ndsimulator.constant import kcal, escale, lscale
from ndsimulator.potentials.potential import Potential


class DoubleWell(Potential):
    ndim = 2
    x0 = 20.0
    y0 = 10
    y1 = 30
    K1 = 1.0
    K2 = 0.9
    invsigma2 = 0.01

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        f = np.zeros(x.shape)
        exp1 = np.exp(
            -((x[0] - self.x0) ** 2) * self.invsigma2
            - (x[1] - self.y0) ** 2 * self.invsigma2
        )
        exp2 = np.exp(
            -((x[0] - self.x0) ** 2) * self.invsigma2
            - (x[1] - self.y1) ** 2 * self.invsigma2
        )
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        f[0] = -self.K1 * exp1 * (x[0] - self.x0) - self.K2 * exp2 * (x[0] - self.x0)
        f[1] = -self.K1 * exp1 * (x[1] - self.y0) - self.K2 * exp2 * (x[1] - self.y1)
        f *= self.invsigma2
        return V, f

    def projection(self, X, Y):
        exp1 = np.exp(
            -((X - self.x0) ** 2) * self.invsigma2 - (Y - self.y0) ** 2 * self.invsigma2
        )
        exp2 = np.exp(
            -((X - self.x0) ** 2) * self.invsigma2 - (Y - self.y1) ** 2 * self.invsigma2
        )
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        return V


class Doublewell2dto1d(DoubleWell):
    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        xnew = self.true_colvar.compute(x)[0]
        jacobian = self.true_colvar.jacobian(x)
        exp1 = np.exp(-((xnew - self.x0) ** 2) * self.invsigma2)
        exp2 = np.exp(-((xnew - self.x1) ** 2) * self.invsigma2)
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        f0 = -self.K1 * exp1 * (xnew - self.x0) - self.K2 * exp2 * (xnew - self.x0)
        f0 *= self.invsigma2
        f0 = f0 * jacobian
        return V, f0.reshape([-1])

    def projection(self, X):
        exp1 = np.exp(-((X - self.x0) ** 2) * self.invsigma2)
        exp2 = np.exp(-((X - self.x1) ** 2) * self.invsigma2)
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        return V


class RingDoubleWell(DoubleWell):
    ndim = 2

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        xnew = np.sqrt(x[0] ** 2 + x[1] ** 2)
        jacobian = np.array([x[0] / xnew, x[1] / xnew])
        exp1 = np.exp(-((xnew - self.x0) ** 2) * self.invsigma2)
        exp2 = np.exp(-((xnew - self.x1) ** 2) * self.invsigma2)
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        f0 = -self.K1 * exp1 * (xnew - self.x0) - self.K2 * exp2 * (xnew - self.x0)
        f0 *= self.invsigma2
        f0 = f0 * jacobian
        return V, f0

    def projection(self, X, Y):
        newx = np.sqrt(X ** 2 + Y ** 2)
        exp1 = np.exp(-((newx - self.x0) ** 2) * self.invsigma2)
        exp2 = np.exp(-((newx - self.x1) ** 2) * self.invsigma2)
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        return V


class TiltedDoubleWell(DoubleWell):
    ndim = 2
    sint = np.sqrt(2) / 2.0
    cost = sint
    jac = np.array([[cost, -sint], [sint, cost]])

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        f = np.zeros(x.shape)
        newx = [
            x[0] * self.cost - x[1] * self.sint,
            x[0] * self.sint + x[1] * self.cost,
        ]
        exp1 = np.exp(
            -((newx[0] - self.x0) ** 2) * self.invsigma2
            - (newx[1] - self.y0) ** 2 * self.invsigma2 * 5
        )
        exp2 = np.exp(
            -((newx[0] - self.x0) ** 2) * self.invsigma2
            - (newx[1] - self.y1) ** 2 * self.invsigma2 * 5
        )
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        f[0] = (
            -self.K1 * exp1 * (newx[0] - self.x0) * self.invsigma2
            - self.K2 * exp2 * (newx[0] - self.x0) * self.invsigma2 * 5
        )
        f[1] = (
            -self.K1 * exp1 * (newx[1] - self.y0) * self.invsigma2
            - self.K2 * exp2 * (newx[1] - self.y1) * self.invsigma2 * 5
        )
        return V, f.dot(self.jac)

    def projection(self, X, Y):
        newx = [X * self.cost - Y * self.sint, X * self.sint + Y * self.cost]
        exp1 = np.exp(
            -((newx[0] - self.x0) ** 2) * self.invsigma2
            - (newx[1] - self.y0) ** 2 * self.invsigma2 * 5
        )
        exp2 = np.exp(
            -((newx[0] - self.x0) ** 2) * self.invsigma2
            - (newx[1] - self.y1) ** 2 * self.invsigma2 * 5
        )
        V = -0.5 * (self.K1 * exp1 + self.K2 * exp2)
        return V

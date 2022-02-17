from ndsimulator.potentials.potential import Potential
import numpy as np
from ndsimulator.constant import kcal, escale, lscale


class ThreeHole2d(Potential):
    ndim = 2
    require_colvar = True

    es = 2.0
    ls = 4.0
    ref = 2
    A = np.array([3, -3, -5, -5]) / es
    a = np.array([-1, -1, -1, -1]) / (ls ** 2)
    b = np.array([0, 0, 0, 0]) / (ls ** 2)
    c = np.array([-1, -1, -1, -1]) / (ls ** 2)
    x0 = np.array([0, 0, 1, -1]) * ls + 7.5
    y0 = np.array([1 / 3.0, 5 / 3.0, 0, 0]) * ls + 7.5
    D = 0.2 / (ls ** 4) / es
    E = 0.2 / (ls ** 4) / es
    x1 = 0 * ls + 7.5
    y1 = 1 / 3.0 * ls + 7.5

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        xnew = self.true_colvar.compute(
            x
        )  # np.array([np.sqrt(x[0]**2+x[1]**2 + 1e-7*x[4]**2),
        # np.sqrt(x[2]**2+x[3]**2)])
        # TO DO: add exception at original point because the function is not continuous there
        jacobian = self.true_colvar.jacobian(x)

        V = 0
        f0 = np.zeros(2)
        for i in range(4):
            tmp = self.A[i] * np.exp(
                self.a[i] * (xnew[0] - self.x0[i]) ** 2
                + self.b[i] * (xnew[0] - self.x0[i]) * (xnew[1] - self.y0[i])
                + self.c[i] * (xnew[1] - self.y0[i]) ** 2
            )
            V += tmp
            f0[0] -= tmp * (
                2 * self.a[i] * (xnew[0] - self.x0[i])
                + self.b[i] * (xnew[1] - self.y0[i])
            )
            f0[1] -= tmp * (
                2 * self.c[i] * (xnew[1] - self.y0[i])
                + self.b[i] * (xnew[0] - self.x0[i])
            )
        V += self.D * (xnew[0] - self.x1) ** 4
        V += self.E * (xnew[1] - self.y1) ** 4 + self.ref

        f0[0] -= 4.0 * self.D * (xnew[0] - self.x1) ** 3
        f0[1] -= 4.0 * self.E * (xnew[1] - self.y1) ** 3

        return V, f0.dot(jacobian)

    def projection(self, X, Y):
        V = np.zeros(X.shape)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a[i] * (X - self.x0[i]) ** 2
                + self.b[i] * (X - self.x0[i]) * (Y - self.y0[i])
                + self.c[i] * (Y - self.y0[i]) ** 2
            )
        V += self.D * (X - self.x1) ** 4
        V += self.E * (Y - self.y1) ** 4 + self.ref
        return V


class ThreeHole5d(Potential):
    ndim = 5
    require_colvar = True

    es = 2.0
    ls = 4.0
    ref = 2
    A = np.array([3, -3, -5, -5]) / es
    a = np.array([-1, -1, -1, -1]) / (ls ** 2)
    b = np.array([0, 0, 0, 0]) / (ls ** 2)
    c = np.array([-1, -1, -1, -1]) / (ls ** 2)
    x0 = np.array([0, 0, 1, -1]) * ls + 7.5
    y0 = np.array([1 / 3.0, 5 / 3.0, 0, 0]) * ls + 7.5
    D = 0.2 / (ls ** 4) / es
    E = 0.2 / (ls ** 4) / es
    x1 = 0 * ls + 7.5
    y1 = 1 / 3.0 * ls + 7.5

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        xnew = self.true_colvar.compute(
            x
        )  # np.array([np.sqrt(x[0]**2+x[1]**2 + 1e-7*x[4]**2),
        # np.sqrt(x[2]**2+x[3]**2)])
        # TO DO: add exception at original point because the function is not continuous there
        jacobian = self.true_colvar.jacobian(x)

        V = 0
        f0 = np.zeros(2)
        for i in range(4):
            tmp = self.A[i] * np.exp(
                self.a[i] * (xnew[0] - self.x0[i]) ** 2
                + self.b[i] * (xnew[0] - self.x0[i]) * (xnew[1] - self.y0[i])
                + self.c[i] * (xnew[1] - self.y0[i]) ** 2
            )
            V += tmp
            f0[0] -= tmp * (
                2 * self.a[i] * (xnew[0] - self.x0[i])
                + self.b[i] * (xnew[1] - self.y0[i])
            )
            f0[1] -= tmp * (
                2 * self.c[i] * (xnew[1] - self.y0[i])
                + self.b[i] * (xnew[0] - self.x0[i])
            )

        V += self.D * (xnew[0] - self.x1) ** 4
        V += self.E * (xnew[1] - self.y1) ** 4
        V += self.ref

        f0[0] -= 4.0 * self.D * (xnew[0] - self.x1) ** 3
        f0[1] -= 4.0 * self.E * (xnew[1] - self.y1) ** 3

        return V, f0.dot(jacobian)

    def projection(self, X, Y):
        V = np.zeros(X.shape)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a[i] * (X - self.x0[i]) ** 2
                + self.b[i] * (X - self.x0[i]) * (Y - self.y0[i])
                + self.c[i] * (Y - self.y0[i]) ** 2
            )
        V += self.D * (X - self.x1) ** 4
        V += self.E * (Y - self.y1) ** 4
        V += self.ref
        return V

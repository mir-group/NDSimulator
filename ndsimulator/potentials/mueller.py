import numpy as np
from ndsimulator.constant import kcal, escale, lscale
from ndsimulator.potentials.potential import Potential


class Mueller2d(Potential):
    ndim = 2
    A = np.array([-200, -100, -170, 15]) * kcal * escale
    a = np.array([-1, -1, -6.5, 0.7]) / (lscale ** 2)
    b = np.array([0, 0, 11, 0.6]) / (lscale ** 2)
    c = np.array([-10, -10, -6.5, 0.7]) / (lscale ** 2)

    def __init__(
        self,
        x_center=np.array([1, 0, -0.5, -1]) * lscale + 32,
        y_center=np.array([0, 0.5, 1.5, 1]) * lscale + 8,
        dim=2,
        run=None,
    ):
        self.dim = dim
        super(Mueller2d, self).__init__(run)
        self.x_center = x_center
        self.y_center = y_center

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        V = 0
        f = np.zeros(x.shape)
        for i in range(4):
            tmp = self.A[i] * np.exp(
                self.a[i] * (x[0] - self.x_center[i]) ** 2
                + self.b[i] * (x[0] - self.x_center[i]) * (x[1] - self.y_center[i])
                + self.c[i] * (x[1] - self.y_center[i]) ** 2
            )
            V += tmp
            f[0] -= tmp * (
                2 * self.a[i] * (x[0] - self.x_center[i])
                + self.b[i] * (x[1] - self.y_center[i])
            )
            f[1] -= tmp * (
                2 * self.c[i] * (x[1] - self.y_center[i])
                + self.b[i] * (x[0] - self.x_center[i])
            )
        return V, f

    def projection(self, X, Y):
        V = np.zeros(X.shape)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a[i] * (X - self.x_center[i]) ** 2
                + self.b[i] * (X - self.x_center[i]) * (Y - self.y_center[i])
                + self.c[i] * (Y - self.y_center[i]) ** 2
            )
        return V


class MuellerNd(Mueller2d):
    ndim = None

    k0 = 0.5

    def compute(self, x=None):
        if x is None:
            x = self.atoms.positions
        V = 0
        Varray = 1.0e-3 * np.exp(-x[5:] ** 2 / 2.0)
        V += np.sum(Varray)
        f0 = np.zeros(2)
        for i in range(4):
            tmp = self.A[i] * np.exp(
                self.a[i] * (x[0] - self.x_center[i]) ** 2
                + self.b[i] * (x[0] - self.x_center[i]) * (x[1] - self.y_center[i])
                + self.c[i] * (x[1] - self.y_center[i]) ** 2
            )
            V += tmp
            f0[0] -= tmp * (
                2.0 * self.a[i] * (x[0] - self.x_center[i])
                + self.b[i] * (x[1] - self.y_center[i])
            )
            f0[1] -= tmp * (
                2.0 * self.c[i] * (x[1] - self.y_center[i])
                + self.b[i] * (x[0] - self.x_center[i])
            )
        tmp = self.k0 * np.exp(-x[2:] ** 2)
        frest = 2.0 * x[2:] * tmp
        V += np.sum(tmp)

        ndim = self.ndim
        f = np.zeros(ndim)
        f[0] = f0[0]
        f[1] = f0[1]
        f[2:] = frest

        return V, f

    def projection(self, X, Y):
        V = np.zeros(X.shape)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a[i] * (X - self.x_center[i]) ** 2
                + self.b[i] * (X - self.x_center[i]) * (Y - self.y_center[i])
                + self.c[i] * (Y - self.y_center[i]) ** 2
            )
        return V


class Mueller5d(Mueller2d):
    ndim = 5
    require_colvar = True

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
                self.a[i] * (xnew[0] - self.x_center[i]) ** 2
                + self.b[i]
                * (xnew[0] - self.x_center[i])
                * (xnew[1] - self.y_center[i])
                + self.c[i] * (xnew[1] - self.y_center[i]) ** 2
            )
            V += tmp
            f0[0] -= tmp * (
                2 * self.a[i] * (xnew[0] - self.x_center[i])
                + self.b[i] * (xnew[1] - self.y_center[i])
            )
            f0[1] -= tmp * (
                2 * self.c[i] * (xnew[1] - self.y_center[i])
                + self.b[i] * (xnew[0] - self.x_center[i])
            )

        return V, f0.dot(jacobian)

    def projection(self, X, Y):
        V = np.zeros(X.shape)
        for i in range(4):
            V += self.A[i] * np.exp(
                self.a[i] * (X - self.x_center[i]) ** 2
                + self.b[i] * (X - self.x_center[i]) * (Y - self.y_center[i])
                + self.c[i] * (Y - self.y_center[i]) ** 2
            )
        return V


class Mueller5dshl(Mueller5d):
    ndim = 5
    A = np.array([-200, -100, -170, 15]) * kcal * escale / 5

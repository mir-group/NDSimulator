import numpy as np
from .colvar import Colvar


class Proj5dto1dx0(Colvar):
    colvardim = 1
    jacob = np.array([1, 0, 0, 0, 0]).reshape(1, 5)

    def compute(self, x):
        newx = np.array(x[0]).reshape(1, 1)
        return newx

    def jacobian(self, x):
        newx = np.array(x[0]).reshape(1, 1)
        return self.jacob


class Proj5dto1dx2(Colvar):
    colvardim = 1
    jacob = np.array([0, 0, 1.0, 0, 0]).reshape(1, 5)

    def compute(self, x):
        newx = np.array(x[2]).reshape(1, 1)
        return newx

    def jacobian(self, x):
        newx = np.array(x[2]).reshape(1, 1)
        return self.jacob


class Proj5dto1d(Colvar):
    colvardim = 1

    def compute(self, x):
        newx = np.array([np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-7 * x[4] ** 2)]).reshape(
            1, 1
        )
        return newx

    def jacobian(self, x):
        newx = np.array([np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-7 * x[4] ** 2)])
        return np.array(
            [x[0] / newx[0], x[1] / newx[0], 0, 0, 2.0 * 1.0e-7 * x[4] / newx[0]]
        ).reshape(1, 5)


class Proj5dto2d(Colvar):
    colvardim = 2

    def compute(self, x):
        newx = np.array(
            [
                np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-7 * x[4] ** 2),
                np.sqrt(x[2] ** 2 + x[3] ** 2),
            ]
        )
        return newx

    def jacobian(self, x):
        newx = np.array(
            [
                np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-7 * x[4] ** 2),
                np.sqrt(x[2] ** 2 + x[3] ** 2),
            ]
        )
        return np.array(
            [
                [x[0] / newx[0], x[1] / newx[0], 0, 0, 1.0e-7 * x[4] / newx[0]],
                [0, 0, x[2] / newx[1], x[3] / newx[1], 0],
            ]
        )


class Proj5dto2dv2(Colvar):
    colvardim = 2

    def compute(self, x):
        newx = np.array(
            [np.sqrt(x[0] ** 2 + (x[1] - x[4]) ** 2), np.sqrt(x[2] ** 2 + x[3] ** 2)]
        )
        return newx

    def jacobian(self, x):
        newx = np.array(
            [np.sqrt(x[0] ** 2 + (x[1] - x[4]) ** 2), np.sqrt(x[2] ** 2 + x[3] ** 2)]
        )
        return np.array(
            [
                [
                    x[0] / newx[0],
                    (x[1] - x[4]) / newx[0],
                    0,
                    0,
                    -(x[1] - x[4]) / newx[0],
                ],
                [0, 0, x[2] / newx[1], x[3] / newx[1], 0],
            ]
        )

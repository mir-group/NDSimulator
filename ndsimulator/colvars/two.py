import numpy as np

from .colvar import Colvar


class ExtractY(Colvar):
    colvardim = 1
    jacob = np.array([0, 1]).reshape(1, 2)

    def compute(self, x):
        return np.array(x[1]).reshape(1, 1)

    def jacobian(self, x):
        return self.jacob


class ExtractX(Colvar):
    colvardim = 1
    jacob = np.array([1, 0]).reshape(1, 2)

    def compute(self, x):
        return np.array([x[0]]).reshape(1, 1)

    def jacobian(self, x):
        return self.jacob


class XmY(Colvar):
    colvardim = 1
    jacob = np.array([1, -1]).reshape(1, 2)

    def compute(self, x):
        return np.array([x[0] - x[1]]).reshape(1, 1)

    def jacobian(self, x):
        return self.jacob


class XpY(Colvar):
    colvardim = 1
    jacob = np.array([1, 1]).reshape(1, 2)

    def compute(self, x):
        return np.array([x[0] + x[1]]).reshape(1, 1)

    def jacobian(self, x):
        return self.jacob


class Rotate2d(Colvar):
    colvardim = 2
    sqrt2 = np.sqrt(2)
    rotation = np.array([[1.0, 1.0], [1.0, -1.0]]).T / np.sqrt(2)
    jacob = np.copy(rotation)

    def compute(self, x):
        newx = x.dot(self.rotation)
        return newx

    def jacobian(self, x):
        return self.jacob


class Proj2dto1d(Colvar):
    colvardim = 1

    def compute(self, x):
        newx = np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])
        return newx

    def jacobian(self, x):
        newx = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return np.array([[x[0] / newx, x[1] / newx]])

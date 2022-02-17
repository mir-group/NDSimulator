import numpy as np
from .colvar import Colvar


class SVMdf(Colvar):
    """SVM decision function, with only function values"""

    colvardim = None

    def compute(self, x):
        pass

    def jacobian(self, x):
        dy = np.zeros([1, self.ndim])
        for i in range(self.ndim):
            dx = np.zeros(self.ndim)
            dx[i] += 1.0
            dy[0, i] = self.compute(x + dx) - self.compute(x - dx)
        return dy


class Pytorch(Colvar):
    """an empty container for the pytorch based colvar"""

    colvardim = None

    def compute(self, x):
        pass

    def jacobian(self, x):
        pass

"""
Path Variables
==============
This module implements the path variable as introduced by :footcite:t:`doi_10.1063_1.2432340`
"""

import numpy as np
from copy import deepcopy
from .colvar import Colvar
from ndsimulator.data import AllData
from typing import Optional


class Path(Colvar):
    """1 dimensional Path variable. The input file needs to contain a (n_ref, x) array that conform np.loadtxt syntax.

    n_ref is the number of reference points for the path variable
    x is the dimension of the space plus 1

    the first column contains the indices and the remaining columns contain the reference points

    Args:
        file_name (str, optional): the txt file storing the path parameters. Defaults to None.
        sigma (float, optional): width of the path variable. Defaults to 1.0.
        ref (np.ndarray, optional): the reference position for the path. Defaults to None.
        ref_ind (np.ndarray, optional): the reference index for the path. Defaults to None.
    """

    colvardim = 1

    def __init__(
        self,
        file_name: Optional[str] = None,
        sigma: float = 1.0,
        ref: Optional[np.ndarray] = None,
        ref_ind: Optional[np.ndarray] = None,
    ):
        self.file_name = file_name
        self.sigma = sigma
        self.ref = ref
        self.ref_ind = ref_ind
        self.N = 0
        self.sig2_invh = 1 / 2.0
        self.dref_mat = None
        self.current_X = None
        self.nom = None
        self.denom = None

    def initialize(self, pointer=None):

        AllData.__init__(self, run=pointer)

        if self.file_name is not None:
            data = np.loadtxt(self.file_name)
            self.ref = data[:, 1:]
            self.ref_ind = (data[:, 0]).reshape([-1, 1])

        self.path_dim = self.ref.shape[1]
        self.N = int(self.ref_ind[-1])
        self.sig2_invh = 1 / (self.sigma ** 2) / 2.0

        n_points = self.ref.shape[0]
        self.dref_mat = np.zeros([self.path_dim, n_points, n_points])
        for i in range(n_points):
            for j in range(n_points):
                self.dref_mat[:, i, j] = self.ref_ind[i] * (self.ref[i] - self.ref[j])
        self.dref_mat = self.dref_mat / self.sigma ** 2

        self.current_X = None
        self.nom = None
        self.denom = None

    def compute(self, x):

        dx2 = np.sum((x - self.ref) ** 2 * self.sig2_invh, axis=1)
        dxmin = np.min(dx2, axis=0)
        dx2 = dx2 - dxmin
        dx2 = dx2.reshape([-1, 1])

        self.current_X = deepcopy(x)
        self.nom = np.exp(-dx2)

        denom = np.sum(self.nom)
        nom = np.sum(np.multiply(self.ref_ind, self.nom))

        self.denom = denom

        return np.array([nom / denom / self.N])

    def jacobian(self, x):

        if not np.array_equal(x, self.current_X):
            self.compute(x)

        matrix = np.matmul(self.nom, self.nom.reshape([1, -1]))

        self.jacob = np.zeros(self.path_dim)
        for i in range(self.path_dim):
            self.jacob[i] = np.sum(matrix * self.dref_mat[i, :, :])
        self.jacob = self.jacob.reshape([1, -1]) / (self.denom) ** 2 / self.N

        return self.jacob


class Path5dto2d(Path):
    """2 dimensional Path variable for a 5 dimensional space.
    The input file needs to contain a (n_ref, 3) array that conform np.loadtxt syntax.

    n_ref is the number of reference points for the path variable.

    Args:
        file_name (str, optional): the txt file storing the path parameters. Defaults to None.
        sigma (float, optional): width of the path variable. Defaults to 1.0.
        ref (np.ndarray, optional): the reference position for the path. Defaults to None.
        ref_ind (np.ndarray, optional): the reference index for the path. Defaults to None.
    """

    colvardim = 1

    def compute(self, x):

        newx = np.array(
            [
                np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-7 * x[4] ** 2),
                np.sqrt(x[2] ** 2 + x[3] ** 2),
            ]
        )

        dx2 = np.sum((newx - self.ref) ** 2 * self.sig2_invh, axis=1)
        dxmin = np.min(dx2)
        dx2 = dx2 - dxmin
        dx2 = dx2.reshape([-1, 1])

        expdx2 = np.exp(-dx2)
        denom = np.sum(expdx2)
        nom = np.sum(self.ref_ind * expdx2)

        self.expdx2 = expdx2
        self.denom = denom
        self.current_X = deepcopy(x)
        self.current_newX = deepcopy(newx)

        return np.array([nom / denom / self.N])

    def jacobian(self, x):

        if not np.array_equal(x, self.current_X):
            self.compute(x)

        expdx2 = self.expdx2
        denom = self.denom

        newx = self.current_newX

        matrix = np.matmul(expdx2, expdx2.reshape([1, -1]))

        jab1 = np.zeros(2)
        for i in range(2):
            jab1[i] = np.sum(matrix * self.dref_mat[i, :, :])
        jab1 = jab1.reshape([1, -1]) / (denom) ** 2 / self.N

        jab2 = np.array(
            [
                [x[0] / newx[0], x[1] / newx[0], 0, 0, 1.0e-7 * x[4] / newx[0]],
                [0, 0, x[2] / newx[1], x[3] / newx[1], 0],
            ]
        )

        self.jacob = np.matmul(jab1, jab2)

        ngrad = np.zeros(5)
        epsilon = 0.01
        for i in range(5):
            dx = np.zeros(5)
            dx[i] += epsilon
            pp = self.compute(x + dx)
            pm = self.compute(x - dx)
            ngrad[i] = (pp - pm) / 2.0 / epsilon

        return self.jacob

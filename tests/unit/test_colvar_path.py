import numpy as np

from ndsimulator.colvars import Path, Path5dto2d


class TestColvar:
    def test_path1d(self):

        # set up input
        cov = Path(
            ref=np.array([1, 3, 5, 7]).reshape([-1, 1]),
            ref_ind=np.array([0, 1, 2, 3]).reshape([-1, 1]),
            sigma=1,
        )
        cov.initialize()

        # reference
        ref = 0.8287934528751985
        center = np.array([6])
        p = cov.compute(center)
        print(p)
        assert np.abs(ref - p) < 1e-6, "the path variable computed is wrong"

        # numerical difference
        epsilon = 0.001
        pm = cov.compute(center - epsilon)[0]
        pp = cov.compute(center + epsilon)[0]
        ref = (pp - pm) / 2.0 / epsilon
        grad = cov.jacobian(center)[0][0]
        assert np.abs(grad - ref) < 1e-6, "the jacobian computed is wrong"

    def test_path2d(self):

        # set up input
        cov = Path(
            ref=np.array([[1, 3.14], [2.05, 4], [3.9, 5], [4.6, 5.9]]),
            ref_ind=np.array([0, 1, 2, 3]).reshape([-1, 1]),
        )
        cov.initialize()

        center = np.array([2.5, 3.1])
        p = cov.compute(center)
        ref = 0.24646052616257966
        assert np.abs(ref - p) < 1e-6, "the path variable computed is wrong"

        # numerical difference
        epsilon = 0.001
        for i in range(2):
            dx = np.zeros(2)
            dx[i] += epsilon
            pm = cov.compute(center - dx)[0]
            pp = cov.compute(center + dx)[0]
            ref = (pp - pm) / 2.0 / epsilon
            grad = cov.jacobian(center)[0][i]
            assert np.abs(grad - ref) < 1e-6, "the jacobian computed is wrong"

    def test_path2d_2(self):

        # set up input
        cov = Path(file_name="tests/inputs/path_ref")
        cov.initialize()

        center = np.array([17.40447553, 19.68877675])
        p = cov.compute(center)[0]
        ref = 0.50117643
        assert np.abs(ref - p) < 1e-6, "the path variable computed is wrong"

        # numerical difference
        epsilon = 0.001
        for i in range(2):
            dx = np.zeros(2)
            dx[i] += epsilon
            pm = cov.compute(center - dx)[0]
            pp = cov.compute(center + dx)[0]
            ref = (pp - pm) / 2.0 / epsilon
            grad = cov.jacobian(center)[0][i]
            assert np.abs(grad - ref) < 1e-6, "the jacobian computed is wrong"

    def test_path_5dto2d(self):

        # set up input
        cov = Path5dto2d(file_name="tests/inputs/path_ref")
        cov.initialize()

        cov_ref = Path(ref=cov.ref, ref_ind=cov.ref_ind)
        cov_ref.initialize()

        tx = 17.40447553
        ty = 19.68877675
        x0 = 2.52061352
        x4 = 8.59285082
        x3 = 0.42873581
        x1 = np.sqrt(tx**2 - x0**2 - 1e-7 * x4**2)
        x2 = np.sqrt(ty**2 - x3**2)
        center = np.array([x0, x1, x2, x3, x4])
        tcenter = np.array([tx, ty])
        p = cov.compute(center)
        ref = cov_ref.compute(tcenter)
        assert np.abs(ref - p) < 1e-6, "the path variable computed is wrong"

        # compare numerical difference with the jacobian
        epsilon = 0.001
        for idir in range(5):
            dx = np.zeros(5)
            dx[idir] = epsilon
            pm = cov.compute(center - dx)[0]
            pp = cov.compute(center + dx)[0]
            ref = (pp - pm) / 2.0 / epsilon
            grad = cov.jacobian(center)[0][idir]
            assert np.abs(grad - ref) < 1e-6, "the jacobian computed is wrong"

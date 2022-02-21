"""
Committor Analysis

Class to run relaxation/minimization/one-way-shooting until it falls into a basin

At the moment, only square boxes is implemented for committing criteria.
Meaning, for each basin, one can define a box of [[x1_min, x1_max], [x2_min, x2_max], ..., [xn_min, xn_max]]
once the simulation enter one of the basin boxes, the simulation will stop.

"""

import numpy as np

from pyfile_utils import instantiate

from ndsimulator.data import AllData
from .md import MD
from .minimize import Minimize


class Committor(AllData):
    """Committor simulation. MD engine that stops at basin or when it hits maximum steps.

    Args:
        n_basins (int): Number of basins
        criteria (list): Commit criteria. [[[x1_min, x1_max], [x2_min, x2_max], ..., [xn_min, xn_max]], [2nd basin], ...]
        diffusion_time (int, optional): . Defaults to 0.
        run (AllData, optional): Pointers for AllData. Defaults to None.
    """

    def __init__(
        self,
        n_basins,
        criteria,
        engine_method,
        engine_method_kwargs={},
        #                   [[30, 35], [13, 18]],
        # =[[[20, 25], [30, 35]], [[38, 50], [2, 10]]],
        diffusion_time=0,
        emin: float = -1.2,
        run: AllData = None,
    ):
        super(Committor, self).__init__(run)

        self.emin = emin
        self.n_basins = n_basins
        self.criteria = np.array(criteria)
        self.diffusion_time = diffusion_time

        self.engine_method, self.engine_method_kwargs = instantiate(
            engine_method,
            # MD if self.kBT != 0 else Minimize,
            # prefix="md" if self.kBT != 0 else "minimize",
            optional_args=engine_method_kwargs,
        )

        assert len(self.criteria) == self.n_basins

    def initialize(self, run):
        """
        initialization. For finite temperature, set up MD engine.
        For 0 K, set up minimization
        """

        AllData.__init__(self, run)
        self.stat.track_everything()

        self.engine_method.initialize(run)
        self.commit_basin = -1

        self.colvardim = self.colvar.colvardim
        for ibasin, critirion in enumerate(self.criteria):
            assert len(critirion) == self.colvardim

    def begin(self):
        """
        trigger initialization
        """

        self.engine_method.begin()

    def update(self, step, time):
        """
        normal integration step or minimization step
        but condition the stopping criteria with committing criteria

        Args:
            step (int): current time step
            time (float): current time
        """

        if self.kBT != 0:
            self.engine_method.update(step, time)
            nostop = True
        else:
            self.engine_method.update()
            nostop = not self.engine_method.stop
        x = self.atoms.positions

        if step < self.diffusion_time:
            return nostop

        nostop, ib = self.commit(
            nostop,
            self.kBT,
            self.atoms.colv,
            self.criteria,
            self.atoms.pe,
            self.emin,
        )
        self.commit_basin = ib

        if self.screen and (not nostop):
            print("find basin", ib, self.atoms.colv)

        return nostop

    def commit(self, nostop, kBT, colv, criteria, pe, emin):
        """
        determine whether the current state atoms.colv meet the commit
        criteria. This function can be reloaded for more complex senario.

        Args:
            nostop (boolean): True for not committed
        """
        ib = -1
        for (idx, c) in enumerate(criteria):
            inbasin = True
            for icolv in range(self.colvardim):
                if (colv[icolv] < c[icolv][0]) or (colv[icolv] > c[icolv][1]):
                    inbasin = False

            if kBT != 0:
                cond3 = True
            else:
                cond3 = pe < self.emin

            if inbasin and cond3:
                ib = idx
                nostop = False

        return nostop, ib

    @property
    def current_dt(self):
        return self.engine_method.current_dt

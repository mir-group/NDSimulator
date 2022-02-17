"""
Class to run relaxation/minimization/one-way-shooting until it falls into a basin

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""

import numpy as np

from pyfile_utils import instantiate

from ndsimulator.data import AllData
from .md import MD
from .minimize import Minimize


class Committor(AllData):
    def __init__(
        self,
        emin=-1.2,
        basins=2,
        criteria=[[[20, 25], [30, 35]], [[38, 50], [2, 10]]],
        #                   [[30, 35], [13, 18]],
        diffusion_time=0,
        run=None,
    ):
        self.emin = emin
        self.basins = basins
        self.criteria = np.array(criteria)
        self.diffusion_time = diffusion_time
        super(Committor, self).__init__(run)

    def initialize(self, run):
        """
        initialization. For finite temperature, set up MD engine.
        For 0 K, set up minimization
        """

        AllData.__init__(self, run)
        self.stat.track_pvf = True
        self.engine, _ = instantiate(
            MD if self.kBT != 0 else Minimize,
            prefix="md" if self.kBT != 0 else "minimize",
            optional_args=run.as_dict(),
        )
        self.engine.initialize(run)
        self.commit_basin = -1

    def begin(self):
        """
        trigger initialization
        """

        self.engine.begin()

    def update(self, step, time):
        """
        normal integration step or minimization step
        but condition the stopping criteria with committing criteria

        Args:
            step (int): current time step
            time (float): current time
        """

        if self.kBT != 0:
            self.engine.update(step, time)
            nostop = True
        else:
            self.engine.update()
            nostop = not self.engine.stop
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
        colvardim = len(colv)
        ib = -1
        for (idx, c) in enumerate(criteria):
            inbasin = True
            for icolv in range(colvardim):
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
        return self.engine.current_dt

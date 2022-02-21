"""
Dumping
~~~~~~~

After a simulation, the code will generate data files like below.

.. highlight:: bash
.. code-block:: bash

    ls reference/md_gamma_0.1/
    $ colvar.dat  force.dat  log  oneplot.png  pos.dat  thermo.dat  velocity.dat
   
All the dat files are managed by the ``Dump`` class. 
Each dat file includes the positions (pos.dat), forces (force.dat), velocities (velocity.dat), temperature (thermo.dat)

"""

import logging

from ndsimulator.data import AllData
from ndsimulator.constant import *


class Dump(AllData):
    """Dumping class

    Args:
        freq (int, optional): the frequency to save dump data. Defaults to 1.
        run (AllData, optional): the AllData pointer. Defaults to None.
    """

    dump_buffer = 1

    def __init__(
        self,
        freq: int = 1,
        run: AllData = None,
    ):
        self.freq = freq
        super(Dump, self).__init__(run)

    def initialize(self, run):
        AllData.__init__(self, run=run)

    def begin(self):
        self.open_files()
        self.write()

    def logger(self, str_name):
        return logging.getLogger(self.log_files[str_name])

    def open_files(self):
        self.log_files = dict(
            thermo=self.output.open_logfile(
                "thermo.dat", screen=False, propagate=False
            ),
            pos=self.output.open_logfile("pos.dat", screen=False, propagate=False),
            force=self.output.open_logfile("force.dat", screen=False, propagate=False),
            velocity=self.output.open_logfile(
                "velocity.dat", screen=False, propagate=False
            ),
            colvar=self.output.open_logfile(
                "colvar.dat", screen=False, propagate=False
            ),
        )
        for (i, fix) in enumerate(self.fixes):
            self.log_files[f"fix{i}"] = self.output.open_logfile(
                f"fix{i}.dat", screen=False, propagate=False
            )

        string = "#time dt T pe ke biase totale"
        for i, fix in enumerate(self.fixes):
            string += f" fix_VR_{i}"
        self.logger("thermo").info(string)

    def write(self, last=False):

        self.logger("pos").info(" ".join([f"{x}" for x in self.atoms.positions]))
        self.logger("force").info(" ".join([f"{f}" for f in self.atoms.forces]))
        self.logger("velocity").info(" ".join([f"{v}" for v in self.atoms.velocities]))
        self.logger("colvar").info(" ".join([f"{c}" for c in self.atoms.colv]))

        stat = self.stat
        string = " ".join(
            [
                f"{val}"
                for val in [
                    stat.time[-1],
                    stat.dt[-1],
                    stat.T[-1],
                    stat.pe[-1],
                    stat.ke[-1],
                    stat.biase[-1],
                    stat.totale[-1],
                ]
            ]
        )
        string += " ".join([f" {fix.VR}" for fix in self.fixes])
        self.logger("thermo").info(string)

        for (i, fix) in enumerate(self.fixes):
            self.logger(f"fix{i}").info(fix.dump_data())

    def close_files(self):
        pass

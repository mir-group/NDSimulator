import numpy as np
from ndsimulator.data import AllData


class ReadDump(AllData):
    def __init__(
        self,
        root: str,
        run_name: str,
        run=None,
    ):
        self.root = root
        self.run_name = run_name
        super(ReadDump, self).__init__(run)

    def initialize(self, run):
        AllData.__init__(self, run=run)
        self.current_dt = self.dt

    def begin(self):
        self.load_data()
        self.update(0, 0)

    def load_data(self):
        self._energy = np.loadtxt("{self.root}/{self.run_name}/energy.dat")
        self._positions = np.loadtxt("{self.root}/{self.run_name}/pos.dat")
        self._colv = np.loadtxt("{self.root}/{self.run_name}/colvar.dat")
        if self._energy.shape[1] > 7:
            self._bias = True
        else:
            self._bias = False

    def update(self, step, time):

        atoms = self.atoms
        atoms.positions = self._positions[step, :]
        atoms.colv = self._colv[step, :]
        (
            self.current_dt,
            atoms.T,
            atoms.pe,
            atoms.ke,
            atoms.biase,
            atoms.totale,
        ) = self._energy[step, 1:7]
        if self._bias:
            pass
        return True

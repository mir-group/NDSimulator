"""
Python interface
================

The simulation can be runned through a python interface as followed.

.. highlight:: python
.. code-block:: python

   from ndsimulator.ndrun import NDRun

   params = dict(
       root = "./",
       run_name = "instance",
       ndim = 2,
       x0 = [0.0, 0.0],
       potential = "square",
       seed = 11111,
       method = "md",
       steps = 100,
       dt = 0.5,
       integrate = "langevin",
       temperature = 300.0,
       md_gamma = 0.005,
       light_plot = True,
   )
   simulation = NDRun(**params)
   simulation.begin()
   simulation.run()
   simulation.end()

"""

import inspect
import logging
import numpy as np

from copy import deepcopy
from typing import Optional, Union

from pyfile_utils import Output, instantiate

from ndsimulator.data import Atom
from .bias import bias_from_config
from .colvars import colvar_from_config
from .constant import kB
from .engines import engine_from_config, Modify
from .io import Dump, Plot, LightPlot, Stat
from .potentials import potential_from_config


class NDRun:
    """

    Args:
        ndim (int): The dimensionality of the energy landscape. Defaults to 2.
        temperature(float, optional): Defaults to 300.0.
        mass (float, optional): Defaults to 5.
        steps (int, optional): Defaults to 100.
        atoms (optional): Defaults to None.
        colvar (optional): Defaults to None.
        true_colvar (optional): Defaults to None.
        method (str, optional): The method for sampling. Defaults to "md".
        random (np.random.RandomState, optional): RandomState. Defaults to None.
        seed (int, optional): random number seed. Defaults to None.
        x0 (list, str, optional): initial positions. Defaults to [0.0, 0.0].
        biases (list, optional): list of biases. Defaults to [].
        plot (bool, optional): If True, the sampling process is plotted. Defaults to False.
        light_plot (bool, optional): If True, only simple post analysis is plotted. Defaults to False.
        screen (bool, optional): If True, screen output is dumped. Defaults to True.
        dump (bool, optional): If True, the position history is logged. Defaults to True.

    Raises:
        NameError: _description_
    """

    object_keys = ["atoms", "colvar", "true_colvar", "random"]

    def __init__(
        self,
        ndim: int,
        temperature: float = 300.0,
        mass: float = 5,
        steps: int = 100,
        atoms=None,
        colvar=None,
        true_colvar=None,
        method: str = "md",
        random=None,
        seed: Optional[int] = None,
        x0: Union[list, str] = [0.0, 0.0],
        biases: Optional[list] = [],
        # trials=100,
        plot=False,
        light_plot=False,
        screen=True,
        dump=True,
        # fileresult=True,
        **kwargs,
    ):

        _local_kwargs = {}
        all_kwargs = {k: v for k, v in kwargs.items()}
        for key in self.init_keys:
            setattr(self, key, locals()[key])
            _local_kwargs[key] = locals()[key]
            all_kwargs[key] = locals()[key]
        output = Output.get_output(all_kwargs)
        self.output = output
        self.logfile = output.open_logfile("log", screen=screen, propagate=False)

        if atoms is None:
            self.atoms, _ = instantiate(Atom, prefix="atom", optional_args=kwargs)
        else:
            self.atoms = deepcopy(atoms)

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(seed)

        self.colvar = (
            colvar_from_config(kwargs, prefix="colvar") if colvar is None else colvar
        )
        self.true_colvar = (
            colvar_from_config(kwargs, prefix="true_colvar")
            if true_colvar is None
            else true_colvar
        )

        self.stat, _ = instantiate(Stat, prefix="stat", optional_args=kwargs)

        self.potential = potential_from_config(kwargs)

        self.engine = engine_from_config(method, all_kwargs)

        self.fixes = []
        for bias in biases:
            self.fixes += bias_from_config(kwargs, bias)

        if light_plot:
            self.plot, _ = instantiate(LightPlot, prefix="plot", optional_args=kwargs)
        elif plot:
            self.plot, _ = instantiate(Plot, prefix="plot", optional_args=kwargs)

        if self.method == "read_dump" or not dump:
            self.dump = None
        else:
            self.dump, _ = instantiate(Dump, prefix="dump", optional_args=kwargs)
        self.modify, _ = instantiate(Modify, prefix="modify", optional_args=kwargs)

        self.atoms.initialize(self)
        self.engine.initialize(self)
        self.potential.initialize(self)
        for fix in self.fixes:
            fix.initialize(self)
        self.colvar.initialize(self)
        self.true_colvar.initialize(self)
        if self.dump:
            self.dump.initialize(self)
        if self.plot:
            self.plot.initialize(self)
        self.modify.initialize(self)
        self.stat.initialize(self)

        if method != "read_dump":
            if not isinstance(self.x0, str):
                self.x0 = np.array(self.x0)
                if self.x0.shape[0] != self.ndim:
                    raise NameError("wrong x0 dimension")
            else:
                self.logger.info("the initial position is random")
            self.atoms.colv = self.colvar.compute(self.atoms.positions)

    def set_seed(self, seed):
        if self.random is not None:
            del self.random
        self.random = np.random.RandomState(seed)

    @property
    def logger(self):
        return logging.getLogger(self.logfile)

    def begin(self):

        atoms = self.atoms

        self.engine.begin()

        if self.method != "read_dump":
            np.copyto(atoms.prev_positions, atoms.positions)

            if atoms.velocities is None:
                self.modify.set_velocity(T=self.temperature)

            atoms.ke = np.sum(atoms.velocities ** 2) * atoms.amass / 2.0
            atoms.T = atoms.ke / self.ndim * 2.0 / kB
            atoms.totale = atoms.ke + atoms.pe
            atoms.prev_colv = np.copy(atoms.colv)

        self.step = 0
        self.time = 0
        self.stat.append(self.time, self.engine.current_dt)

        if self.plot:
            self.plot.begin()
        if self.dump:
            self.dump.begin()

    def end(self):
        if self.plot:
            self.plot.end()
        if self.dump:
            self.dump.close_files()

    def run(self):
        atoms = self.atoms

        steps = self.steps
        nostop = True

        while self.step < steps and nostop:

            if self.screen and self.dump:
                if self.step % self.dump.freq == 0:
                    self.logger.info(f"{self.step} temperature: {self.stat.T[-1]}")
                    # process = psutil.Process(os.getpid())
                    # self.logger.info(process.memory_info().rss)
            for fix in self.fixes:
                fix.update(self.step, self.time)

            nostop = self.engine.update(self.step, self.time)

            self.step += 1

            dump = False
            if "mc" in self.method:
                if self.engine.accept:
                    self.time += self.engine.current_dt
                    dump = True
            else:
                self.time += self.engine.current_dt
                dump = True

            if dump:
                if self.plot:
                    if self.step % self.plot.freq == 0:
                        self.plot.update()
                if self.dump:
                    if self.step % self.dump.freq == 0:
                        self.dump.write()
                if self.step % self.stat.freq == 0:
                    self.stat.append(self.time, self.engine.current_dt)

            if atoms.T > 10000:
                self.logger.debug(
                    f"Warning temperature exceed 10000 {self.engine.current_dt} {self.step}"
                )
            # nostop = False

        self.logger.info(f"end condition {self.step} {steps} {nostop}")

        if self.method == "committor":
            if self.screen:
                self.logger.info(f"!! Commit to basin {self.engine.commit_basin}")
            self.commit_basin = self.engine.commit_basin
        if self.screen and ("mc" in self.method):
            self.logger.info(
                f"!! MC acceptance ratio: {self.engine.accept_rate / steps}"
            )

    @classmethod
    def from_dict(cls, dictionary):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = deepcopy(dictionary)
        ndrun = cls(**dictionary)
        return ndrun

    @property
    def init_keys(self):
        return [
            key
            for key in list(inspect.signature(NDRun.__init__).parameters.keys())
            if key not in (["self", "kwargs"] + NDRun.object_keys)
        ]

    def as_dict(self, kwargs: bool = True):
        dictionary = {}
        for key in self.init_keys:
            dictionary[key] = getattr(self, key, None)

        if kwargs:
            dictionary.update(getattr(self, "kwargs", {}))
        return dictionary

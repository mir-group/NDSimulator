"""
Class to run committor/two-way-shooting (forward and backward MD simulation for given v and x)

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""

from copy import deepcopy
from ndsimulator.ndrun import NDRun
import numpy as np


class OnewayShooting:
    """
    Class to run committor/two-way-shooting

    - **parameters**, **types**, **return** and **return types**::

        :param random: random number sequence
        :param x:      initial position
        :param v:      initial velocity
        :param dv:     amplitude of velocity perturbation
        :param dx:     amplitude of position perturbation
        :type random:  None or numpy.random.RandomState object
        :type x:       None or numpy array or list
        :type v:       None or numpy array or list
        :type dv:      None or float
        :type dx:      None or float

        :ivar time     time
        :ivar pos      positions
        :ivar vel      velocities
        :ivar pe       potential energy
        :ivar n        end basin

    - Example

      :one_commit = OnewayShooting(random=random, **kwargs)
      :forward, backward, na, nb = one_commit.run():


    - forward run with the initialized velocity or position
    in the initialization

      * If kBT > 0, the backward run starts with the same point but
        opposite velocity.
      * If kBT == 0, the backward run starts from the position that is
        perturbed to the opposite direction

    - the forward and backward runs are joint to form the
    whole trajectory.

      * the backward run is flip before joining.
      * time always starts from negative value to 0,
        then to positive value.

    """

    def __init__(
        self, random=None, x=None, v=None, dv=None, dx=None, direction=None, **kwargs
    ):
        """create forward and backward instance

        :param random: random number sequence
        :param x:      initial position
        :param v:      initial velocity
        :param dv:     amplitude of velocity perturbation
        :param dx:     amplitude of position perturbation
        :type random:  numpy.random.RandomState object
        :type x:       numpy array or list
        :type v:       numpy array or list
        :type dv:      float
        :type dx:      float
        """

        kwargs = deepcopy(kwargs)
        run_name = deepcopy(kwargs["run_name"])

        kwargs["method"] = "committor"
        kwargs["track_pvf"] = True
        kwargs["run_name"] = f"{run_name}_ins"
        self.instance = NDRun(random=random, **kwargs)
        instance = self.instance

        if direction is None:
            self.direction = "foward"
        else:
            self.direction = direction

        if x is not None:
            instance.modify.set_positions(x)
        if v is not None:
            instance.modify.set_velocity(v=v)
        if dv is not None:
            instance.modify.perturb_velocities(dv)

        self.dx = 0.1
        if dx is not None:
            self.dx = dx
        if instance.kBT == 0:
            alldx = random.normal(0, self.dx, instance.ndim)
            instance.modify.perturb_atom(alldx=alldx)

    def run(self):
        """run the forward and backward simulation

        :return:  the run instance and basin id of the forward and backward move

        """

        instance = self.instance

        instance.begin()
        instance_v = np.copy(instance.atoms.velocities)
        instance.run()
        instance.end()
        self.n = instance.commit_basin

        self.data = {}
        if self.direction == "forward":
            self.data["T"] = self.instance.stat.T
            self.data["time"] = self.instance.stat.time
            self.data["pe"] = self.instance.stat.pe
            self.data["pos"] = self.instance.stat.positions
            self.data["vel"] = self.instance.stat.velocities
            self.data["force"] = self.instance.stat.forces
            self.data["col"] = self.instance.stat.colv
            self.data["n"] = self.n
        if self.direction == "backward":
            self.data["T"] = np.flip(self.instance.stat.T, axis=0)
            self.data["time"] = -np.flip(self.instance.stat.time, axis=0)
            self.data["pe"] = np.flip(self.instance.stat.pe, axis=0)
            self.data["pos"] = np.flip(self.instance.stat.positions, axis=0)
            self.data["vel"] = -np.flip(self.instance.stat.velocities, axis=0)
            self.data["force"] = np.flip(self.instance.stat.forces, axis=0)
            self.data["col"] = np.flip(self.instance.stat.colv, axis=0)
            self.data["n"] = self.n

        return self.data

    def join(self, anotherhalf, sp):

        backward = {}
        forward = {}
        if self.direction == "forward":
            forward = self.data

            backward["T"] = anotherhalf["T"][:sp]
            backward["time"] = anotherhalf["time"][:sp]
            backward["pe"] = anotherhalf["pe"][:sp]
            backward["pos"] = anotherhalf["pos"][:sp]
            backward["vel"] = anotherhalf["vel"][:sp]
            backward["col"] = anotherhalf["col"][:sp]
            backward["force"] = anotherhalf["force"][:sp]
            backward["n"] = anotherhalf["na"]
            e = backward["time"][-1]
            backward["time"] = np.array(backward["time"]) - e

            self.data = forward
        if self.direction == "backward":

            backward = self.data

            forward["T"] = anotherhalf["T"][sp:]
            forward["time"] = anotherhalf["time"][sp:]
            forward["pe"] = anotherhalf["pe"][sp:]
            forward["pos"] = anotherhalf["pos"][sp:]
            forward["vel"] = anotherhalf["vel"][sp:]
            forward["col"] = anotherhalf["col"][sp:]
            forward["force"] = anotherhalf["force"][sp:]
            s = forward["time"][0]
            forward["time"] = np.array(forward["time"]) - s
            forward["n"] = anotherhalf["nb"]

            self.data = backward

        # join the two trjs
        joint = {}
        joint["T"] = np.hstack([backward["T"], forward["T"]])
        joint["time"] = np.hstack([backward["time"], forward["time"]])
        joint["pe"] = np.hstack([backward["pe"], forward["pe"]])
        joint["pos"] = np.vstack([backward["pos"], forward["pos"]])
        joint["vel"] = np.vstack([backward["vel"], forward["vel"]])
        joint["col"] = np.vstack([backward["col"], forward["col"]])
        joint["force"] = np.vstack([backward["force"], forward["force"]])
        joint["na"] = backward["n"]
        joint["nb"] = forward["n"]
        joint["sp"] = len(backward["time"])
        return joint

    def dump_data(self, filename=None):
        """dump data to local npz file and return the trajectories

        :param filename: name for the npz file to store info
        :type filename:  string
        :return:  the numpy arrays of time, positions, velocities, colvar, potential energy

        """
        if filename is not None:
            np.savez(filename, **self.data)

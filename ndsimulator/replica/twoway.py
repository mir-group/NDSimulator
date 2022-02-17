"""
Class to run committor/two-way-shooting (forward and backward MD simulation for given v and x)

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""

from copy import deepcopy
from ndsimulator.ndrun import NDRun
import numpy as np


class TwowayShooting:
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
        :ivar na       end basin of backward run
        :ivar nb       end basin of forward run

    - Example

      :one_commit = CommittorReplica(random=random, **kwargs):
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

    def __init__(self, random=None, x=None, v=None, dv=None, dx=None, **kwargs):
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

        kwargs["run_name"] = f"{run_name}_forward"
        self.forward = NDRun(random=random, **kwargs)

        kwargs["run_name"] = f"{run_name}_backward"
        self.backward = NDRun(random=self.forward.random, **kwargs)

        forward = self.forward
        backward = self.backward

        if x is not None:
            forward.modify.set_positions(x)
        if v is not None:
            forward.modify.set_velocity(v=v)
        if dv is not None:
            forward.modify.perturb_velocities(dv)

        self.dx = 0.1
        if dx is not None:
            self.dx = dx
        if forward.kBT == 0:
            alldx = random.normal(0, self.dx, forward.ndim)
            forward.modify.perturb_atom(alldx=alldx)

        x0 = np.copy(forward.atoms.positions)

        backward.modify.set_positions(x0)

    def run(self):
        """run the forward and backward simulation

        :return:  the run instance and basin id of the forward and backward move

        """

        forward = self.forward

        forward.begin()
        forward_v = np.copy(forward.atoms.velocities)
        forward.run()
        forward.end()
        nb = forward.commit_basin

        backward = self.backward

        backward.begin()
        backward.modify.set_velocity(v=-forward_v)
        backward.run()
        backward.end()
        na = backward.commit_basin

        # join the two trjs
        a1 = np.flip(np.array(backward.stat.time), 0)
        a2 = np.array(forward.stat.time)
        self.time = np.hstack([-a1, a2])
        del a1, a2
        a1 = np.flip(np.array(backward.stat.pe), 0)
        a2 = np.array(forward.stat.pe)
        self.pe = np.hstack([a1, a2])
        del a1, a2
        a1 = np.flip(np.array(backward.stat.positions), 0)
        a2 = np.array(forward.stat.positions)
        self.pos = np.vstack([a1, a2])
        del a1, a2
        a1 = np.flip(np.array(backward.stat.velocities), 0)
        a2 = np.array(forward.stat.velocities)
        self.vel = np.vstack([-a1, a2])
        del a1, a2
        a1 = np.flip(np.array(backward.stat.colv), 0)
        a2 = np.array(forward.stat.colv)
        self.col = np.vstack([a1, a2])
        del a1, a2
        a1 = np.flip(np.array(backward.stat.forces), 0)
        a2 = np.array(forward.stat.forces)
        self.forces = np.vstack([a1, a2])
        del a1, a2
        self.na = na
        self.nb = nb
        return forward, backward, na, nb

    def dump_data(self, filename=None):
        """dump data to local npz file and return the trajectories

        :param filename: name for the npz file to store info
        :type filename:  string
        :return:  the numpy arrays of time, positions, velocities, colvar, potential energy

        """
        if filename is not None:
            np.savez(
                filename,
                pos=self.pos,
                vel=self.vel,
                force=self.forces,
                col=self.col,
                pe=self.pe,
                na=self.na,
                nb=self.nb,
                time=self.time,
            )
        return self.time, self.pos, self.vel, self.col, self.pe

    def end(self):
        del self.forward
        del self.backward

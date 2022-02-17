"""
Class to run first umb sampling

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""

from ndsimulator.ndrun import NDRun
from ndsimulator.colvars.pytorch import ColvarSVMdf
from ndsimulator.colvars.pytorch import ColvarPytorch
from ndsimulator.replica.twoway import TwowayShooting
import numpy as np
from joblib import load
from copy import deepcopy


class UmbCommit:
    def __init__(
        self, clf=None, umb_kwargs={}, committor_kwargs={}, random=None, **kwargs
    ):

        self.umb_kwargs = deepcopy(kwargs)
        self.umb_kwargs.update(umb_kwargs)
        self.umb_kwargs["biases"] = ["umb"]

        self.committor_kwargs = deepcopy(kwargs)
        self.committor_kwargs.update(committor_kwargs)
        self.committor_kwargs["method"] = "committor"
        self.committor_kwargs["integrate"] = "rescale"
        self.committor_kwargs["umb_n"] = None
        self.committor_kwargs["colvar"] = umb_kwargs.true_colvarfunc

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(umb_kwargs.get("seed", 0))

        self.x0 = np.array(umb_kwargs.x0)
        self.x0 = self.x0.reshape(1, len(self.x0))

        # define the collective variables
        ndim = self.umb_kwargs.ndim
        if umb_kwargs.colvar == "svm_df":
            if clf is None:
                self.clf = load(umb_kwargs.classifier)
            else:
                self.clf = clf
            if hasattr(clf, "predict_proba"):

                def compute(self, x):
                    return clf.predict_proba(x.reshape([1, ndim]))[:, 0]

                def func(x):
                    return clf.predict_proba(x.reshape([1, ndim]))[:, 0]

            elif hasattr(clf, "decision_function"):

                def compute(self, x):
                    return clf.decision_function(x.reshape([1, ndim]))  # [0]

                def func(x):
                    return clf.decision_function(x.reshape([1, ndim]))  # [0]

            else:

                def compute(self, x):
                    return clf.predict(x.reshape([1, ndim]))

                def func(x):
                    return clf.predict(x.reshape([1, ndim]))

                ColvarSVMdf.compute = compute
            self.func = func
        elif umb_kwargs.colvar == "pytorch":
            if umb_kwargs.classifier == "label":
                ColvarPytorch.compute = clf.predict_label
                ColvarPytorch.jacobian = clf.jacobian_label
                self.func = clf.predict_label
            elif umb_kwargs.classifier == "latent":
                ColvarPytorch.compute = clf.encode
                ColvarPytorch.jacobian = clf.jacobian_latent
                self.func = clf.predict_label
                self.func = None
            else:
                raise AssertionError(
                    "colvar pytorch with classifier {} is not supported yet".format(
                        umb_kwargs.classifier
                    )
                )
        else:
            raise AssertionError(
                "umb sampling with colvar {} is not supported yet".format(
                    umb_kwargs.colvar
                )
            )

        umb_kwargs.dump = False
        umb_kwargs.plot = False
        committor_kwargs.dump = False
        committor_kwargs.plot = False

    def run(self):

        originalkBT = self.umb_kwargs.kBT
        ndim = self.umb_kwargs.ndim

        # first umb sampling

        instance = NDRun(random=self.random, **self.umb_kwargs)
        instance.begin()
        if self.func is not None:
            print(
                "start um x0, R",
                instance.true_colvar.compute(self.umb_kwargs.x0),
                self.func(self.umb_kwargs.x0.reshape([1, ndim])),
            )
            print("pe, colv", instance.atoms.pe, instance.atoms.colv)
        instance.run()
        instance.end()

        self.umb_start = self.umb_kwargs.x0
        self.pos1 = np.array(instance.stat.positions)
        self.umb_kwargs.kBT /= originalkBT

        # committor run

        # start from average position
        x = np.average(instance.stat.positions[-100:], axis=0)

        # # start from minimum energy position
        # idp = np.argmin(instance.stat.totale)
        # x = instance.stat.positions[idp]

        # start from last position
        # x = instance.stat.positions[-1]

        if self.func is not None:
            self.umb_end = self.func(x.reshape([1, ndim]))
            print(
                "start committor",
                instance.true_colvar.compute(x),
                self.umb_end,
                instance.fixes[0]._R0,
            )
        self.commit_start = np.copy(x)

        one_commit = TwowayShooting(
            random=self.random, x=instance.atoms.positions, **self.committor_kwargs
        )
        forward, backward, na, nb = one_commit.run()

        self.pos3 = np.array(one_commit.pos)
        self.col3 = np.array(one_commit.col)
        self.step3 = np.array(one_commit.time)
        self.pe3 = np.array(one_commit.pe)

        self.na = na
        self.nb = nb
        print("end committor", na, nb, len(self.pos3))

        del instance
        del one_commit

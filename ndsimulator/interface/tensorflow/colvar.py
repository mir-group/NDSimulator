"""
class to load tensorflow model as collective variables

This version only works with tensorflow 1.14

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""
import tensorflow as tf
import numpy as np
from ndsimulator.data import AllData


class Colvar_tfmodel(AllData):
    def __init__(self):
        self.initialized = False

    def initialize(self, pointer, dim):

        AllData.__init__(self, run=pointer)
        cont = self.cont
        self.ndim = cont.ndim

        if self.initialized:
            return

        modelfolder = cont.tf_folder
        inputname = cont.tf_inputname
        outputname = cont.tf_outputname
        gradname = cont.tf_gradname

        self.sess = tf.Session()

        tf.saved_model.loader.load(self.sess, ["serve"], modelfolder)
        graph = tf.get_default_graph()

        fout = open("read_graph.info", "w+")
        for op in graph.get_operations():
            print(op, file=fout)
        fout.close()

        self.phX = graph.get_tensor_by_name(inputname)
        self.phz = graph.get_tensor_by_name(outputname)
        self.phjab_z = graph.get_tensor_by_name(gradname)

        self.colvardim = self.phz.shape[1]
        if self.colvardim != dim:
            raise NameError(
                "dimension of collective variable {} does not match with the function {}".format(
                    dim, self.colvardim
                )
            )

    def compute(self, x):
        z = self.sess.run(self.phz, feed_dict={self.phX: x.reshape([1, -1])})
        z = z.reshape([-1])
        return z

    def jacobian(self, x):
        jab_z = self.sess.run(self.phjab_z, feed_dict={self.phX: x.reshape([1, -1])})
        return jab_z

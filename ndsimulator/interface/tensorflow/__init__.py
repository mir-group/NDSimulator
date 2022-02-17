import os

try:
    if "tf" in os.environ:
        import tensorflow as tf

        import ndsimulator.colvars

        ndsimulator.colvars.cv_modules += ["ndsimulator.interface.tensorflow.colvar"]

        additional_attr = {}
        additional_attr["tf_folder"] = "models/hello_world"
        additional_attr["tf_inputname"] = "X:0"
        additional_attr["tf_outputname"] = "Z:0"
        additional_attr["tf_gradname"] = "jab_z:0"
except ImportError:
    pass

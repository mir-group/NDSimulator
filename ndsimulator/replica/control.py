"""
Inject extra attributes for replica
Include a couple more variables for multi-replica simulation
command line and file support is the same as the parent class

Lixin Sun, Harvard University, nw13mi0faso@gmail.com
"""
from ndsimulator.control import Control

# general parameters
Control.additional_attr["tracking_mode "] = False
Control.additional_attr["replica "] = 10
Control.additional_attr["grid "] = 10
Control.additional_attr["run_name "] = "PylotRun"
Control.additional_attr["classifier "] = None
Control.additional_attr["initial_data "] = None

# transition path sampling
Control.additional_attr["na0 "] = None
Control.additional_attr["nb0 "] = None
Control.additional_attr["tpe_dv "] = 0.1
Control.additional_attr["commitorn "] = 1
Control.additional_attr["commitor_t "] = 0.0
Control.additional_attr["tpe_initial "] = None

# umbrella sampling
Control.additional_attr["umb_vlist "] = None
Control.additional_attr["umb_vmin "] = 0.0
Control.additional_attr["umb_vmax "] = 1.0

# umbrella sampling k
Control.additional_attr["umb_adaptivek "] = False
Control.additional_attr["umb_rtolerance "] = 0.25

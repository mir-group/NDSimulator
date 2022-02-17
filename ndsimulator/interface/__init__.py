from os import listdir
from os.path import dirname, basename, isfile, join, isdir
import glob
from importlib import import_module

gpath = dirname(__file__)
modules = glob.glob(join(gpath, "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]

modules = listdir(gpath)
__all__ += [basename(f) for f in modules if isdir(join(gpath, f)) and not ("__" in f)]

additional_attr = {}

intf_modules = ["ndsimulator.interface." + f for f in __all__]
for pmname in intf_modules:
    pm = import_module(pmname)
    if "additional_attr" in dir(pm):
        additional_attr.update(pm.additional_attr)

# # in experiment section
# import ndsimulator.interface.pytorch
# import ndsimulator.interface.sklearn

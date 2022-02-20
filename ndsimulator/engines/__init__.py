from pyfile_utils import instantiate
from .md import MD
from .mc import MC
from .minimize import Minimize
from .wang_landau import WangLandau
from .committor import Committor
from .read_dump import ReadDump
from .modify import Modify


def engine_from_config(method, config):

    if method == "md":
        engine = MD
    elif method == "mc":
        engine = MC
    elif method == "wl-mc":
        engine = WangLandau
    elif method == "minimize":
        engine = Minimize
    elif method == "committor":
        engine = Committor
        config["engine_method"] = MD if config["temperature"] != 0 else Minimize
        # prefix="md" if self.kBT != 0 else "minimize",
    elif method == "read_dump":
        engine = ReadDump
    else:
        raise NameError(f"method type ``{method}'' undefined")

    instance, _ = instantiate(engine, prefix=[method, "engine"], optional_args=config)
    return instance

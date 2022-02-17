"""
bias
~~~~
This module implements the bias/penalty functions applied
on the potential landscape
"""
from typing import Union
from pyfile_utils import instantiate, Config

from .gaus import Gaussian
from .pe import PEBias
from .umb import Harmonic


def bias_from_config(config: Union[dict, Config], bias_name: str):
    """instantiate a bias object base on a config dictionary.
    it read the bias_name

    Args:
        config (Union[dict, Config]): config to grab
        bias_name (str): name of bias

    Returns:
        biases: a list of biases
    """

    if bias_name == "mtd":
        instance, _ = instantiate(Gaussian, prefix=bias_name, optional_args=config)
    elif bias_name == "pe":
        instance, _ = instantiate(PEBias, prefix=bias_name, optional_args=config)
    elif bias_name == "umb":
        instances = []
        umb_n = config.get("umb_n", 1)
        for i in range(umb_n):
            instance, _ = instantiate(
                Harmonic, prefix=[f"umb_{i}", "umb"], optional_args=config
            )
            instances += [instance]
        return instances
    return [instance]

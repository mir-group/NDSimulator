from pyfile_utils import instantiate, load_callable


def potential_from_config(config):
    """
    initialize potential instance
    extract all parameters related to this potential class with or without the prefix "potential"
    """

    func_name = config.get("potential", "Gaussian")

    try:
        func_class = load_callable(func_name)
    except:
        if isinstance(func_name, str):
            func_class = load_callable("ndsimulator.potentials." + func_name)
        else:
            raise ValueError("cannot load the potential {func_name}")

    instance, _ = instantiate(func_class, prefix="potential", optional_args=config)
    return instance

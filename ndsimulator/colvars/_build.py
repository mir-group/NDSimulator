from pyfile_utils import instantiate, load_callable


def colvar_from_config(config, prefix: str = "colvar"):
    """
    initialize potential instance
    extract all parameters related to this potential class with or without the prefix "potential"
    """

    func_name = config.get(prefix + "_name", "Original")

    try:
        func_class = load_callable(func_name)
    except:
        if isinstance(func_name, str):
            func_class = load_callable("ndsimulator.colvars." + func_name)
        else:
            raise ValueError("cannot load the potential {func_name}")

    instance, _ = instantiate(func_class, prefix=prefix, optional_args=config)
    return instance

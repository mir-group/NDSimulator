"""
class that handle all paramter used in the normal unbiased MD simulation and
enhanced MD simulations

New parameters can be added before instanization

    >>> Parameters.param['key'] = default_value

The class will take all items in the static member param, add them
to the argument parser. The input type will be consistent with the
default value.

If default value is a bool type, the input argument will be set as
a flag, instead of an value input. If the default, declaring
the name of the value in input argument means seting the value to
False and vice versa.

If default value is set to None, but the type of the
value is not "str". One should also define a type

    >>> Parameters.param['_key_type'] = default_value
    ... Parameters.help_str['key'] = 'help_string'


To parse command line arguments:
    >>> param = Parameters()
    ... param.parse_commandline_args()

To read from dictionary directly
    >>> inputs = {}
    >>> param = Parameters()
    >>> param.copy_args(inputs)
    ... param.check_instance()

To read from a plain-text input file

    >>> param = Parameters()
    ... param.parse_file(filename)

"""

import argparse
import json
import logging
import numpy as np
import random

from copy import deepcopy
from os.path import isfile


class Parameters:
    def __init__(self):
        """
        the initial default value and keys are store in static member param (dictionary)
        once initialized, the values get copy as attributes
        """

        self.param_list = []
        self.add_param(self.param)

    def update_random(self, seed, np_random_state):

        np.random.seed(seed + 1)
        random.seed(seed + 2)
        if np_random_state is not None:
            self.np_random_state = np_random_state
        else:
            self.np_random_state = np.random.RandomState(seed + 3)

    def add_param(self, input_dict):

        for k, value in input_dict.items():

            # if the key is not started with "_"
            # check its typehint
            if k[0] != "_":
                setattr(self, k, deepcopy(value))

                self.param_list += [k]

                # use the type of default value for typehint unless specified
                if value is not None:
                    typehint = type(value)
                else:
                    typehint = self.param.get(f"_{k}_type", None)
                setattr(self, f"_{k}_type", typehint)

    def parse_commandline_args(self, args=None):
        """
        function to call parse_arg to change parameters
        from command line input
        """
        parser = self.define_parser()
        parsed_args = parser.parse_args(args=args)

        # if input_file argument exist
        if parsed_args.input_file is not None:
            inputs = self.read_file(parsed_args.input_file)
            self.copy_args(inputs)
        else:
            for key in parsed_args.__dict__:
                value = parsed_args.__dict__[key]
                setattr(self, key, value)
            self.check_instance()

    def read_file(self, input_file):
        inputs = {}
        if isfile(input_file):
            with open(input_file) as json_file:
                inputs = json.load(json_file)
        else:
            # raise RuntimeError(
            print(f"failed to open input file {self.inputs}")
        return inputs

    def define_parser(self, o_parser=None):
        """add helper information for argparse

        :return args: Namespace object, parsed arguments.
        """
        if o_parser is None:
            parser = argparse.ArgumentParser(
                description="Run MD Tensor-Field Molecular Dynamics."
            )
        else:
            parser = o_parser

        # go through all the default variables
        for k in self.param_list:

            # get the default value and its type
            value = getattr(self, k)
            typehint = getattr(self, f"_{k}_type", str)

            # combine the help string with default values
            help_str = self.help_str.get(k, "")
            if len(help_str) > 0:
                help_str += "; "
            help_str += f"default: {value}"

            # add argument
            if type(value) is bool:
                not_default = repr(not value).lower()
                parser.add_argument(
                    f"--{k}", action=f"store_{not_default}", help=help_str
                )
            else:
                parser.add_argument(
                    f"--{k}", type=typehint, default=value, help=help_str
                )

        return parser

    def copy_args(self, inputs: dict):
        """
        copy parameter set up from a
        dictionary input
        """

        for key in inputs:
            value = inputs[key]

            if key in self.param_list:
                ref_value = getattr(self, key)
                typehint = getattr(self, f"_{key}_type", None)

                if typehint is None:
                    setattr(self, key, value)
                elif isinstance(value, typehint):
                    setattr(self, key, value)
                elif isinstance(value, int) and isinstance(ref_value, float):
                    setattr(self, key, float(value))
                elif ref_value is None:
                    setattr(self, key, value)
                elif isinstance(ref_value, np.ndarray):
                    setattr(self, key, np.array(value))
                else:
                    help_str = self.help_str.get(key, "Not defined")
                    raise NameError(
                        f'The input value of "{key}"'
                        f" = {repr(value):10.10}... is {type(value)}. "
                        f" It should be {type(ref_value)}. "
                        f" Help {key}: {help_str}"
                    )
            else:
                print(f'the input key "{key}" does not exist, skipped')

        self.check_instance()

    def check_instance(self):
        """
        sanity check for parameters
        """
        raise NotImplementedError(
            "this function needs to be implemented in child" "class"
        )

    def parse_file(self, filename: str):
        """Recover Params object from training file.

        :param filename: str, file to read restart info from.
        """

        with open(filename) as rf:
            lines = rf.readlines()

            for line_idx, line in enumerate(lines):

                # system
                splt = line.split()
                if len(splt) <= 1:
                    continue

                nvalue = len(splt) - 1
                key = splt[0][:-1]

                if key in self.param_list:
                    ref_value = getattr(self, key)
                    typehint = getattr(self, f"_{key}_type", str)

                    islist = True
                    if nvalue == 1:
                        islist = False
                        value = splt[1]
                        if value.lower() == "none":
                            value = None
                        elif value == "[]":
                            value = []
                        elif value[0] == "[":
                            islist = True
                        else:
                            try:
                                value = typehint(value)
                            except Exception as e:
                                print(
                                    "type of input does not match with default setting"
                                )
                                raise (e)

                    if islist:
                        start = line.find(":")
                        value_string = line[start + 1 :].strip()
                        value = value_string[1:-1].split(",")
                        try:
                            value = list(map(int, value))
                        except:
                            try:
                                value = list(map(float, value))
                            except:
                                pass
                        if typehint is np.ndarray:
                            value = typehint(value)
                    setattr(self, key, value)
                    print("reference", key, ref_value, value)

                else:
                    logging.info(f"skip line {line}")
                    # raise NotImplementedError(f"{key} is not a key in this parameter set")

    def print_info(self, logger):
        """Print parsed arguments to file."""

        logging.info("\n" + "=" * 50 + "\nPassed Arguments" + "\n" + "=" * 50)

        for k in self.param_list:
            logger.info(f"{k}: {getattr(self, k)}")

        logger.info("=" * 50 + "\n\n")
        for handle in logger.handlers:
            handle.flush()

    param = {}
    param["input_file"] = None

    help_str = {
        "input_file": "json file input for more parameters. This will "
        "overwrite the setting in command line"
    }

# NDSimulator

[![Documentation Status](https://readthedocs.org/projects/nequip/badge/?version=latest)](https://mir-group.github.io/NDSimulator/)

A molecular dynamics tool for toy model systems.

## Features

* model a particle on a N-dimensional energy landscape 
* Molecular dynamics, metadynamics, umbrella sampling are implemented.
* Allow users to define collective variables
* Langevin (1st and 2nd order), velocity rescale and NVE ensemble are available for MD

## Installation

1. from GitHub

```bash
git clone git@github.com:mir-group/NDSimulator.git
pip install -e ./
```

2. from pip

```
pip install ndsimulator
```

### Prerequisits

* Python3.8
* matplotlib
* NumPy
* pyfile-utils

### Testing

```bash
pytest tests
```

## Commandline interface

The inputs specifying the system, method, parameters, and visualization options can be received via an inputs file 

```bash
ndsimulator examples/2d-md.yaml
```

![Result](https://github.com/mir-group/NDSimulator_archive/raw/main/examples/reference/instance_oneplot.png)

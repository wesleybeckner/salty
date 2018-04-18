
## SALTY
[![Build Status](https://travis-ci.org/wesleybeckner/salty.svg?branch=master)](https://travis-ci.org/wesleybeckner/salty)
[![Coverage Status](https://coveralls.io/repos/github/wesleybeckner/salty/badge.svg?branch=master)](https://coveralls.io/github/wesleybeckner/salty?branch=master)
[![PyPI version](https://badge.fury.io/py/salty-ilthermo.svg)](https://badge.fury.io/py/salty-ilthermo)
========
Salty is an interactive data exploration tool for ionic liquid data from ILThermo (NIST)

## Installation

### Dependencies

Salty requires:

* python (>= 3.6)
* scikit-learn (>= 0.19.1)
* rdkit (>= 2017.09.1)

To take full advantage of rdkit you will also need Matplotlib >= 1.3.1.

#### User installation

You will first need to install [rdkit](http://www.rdkit.org/docs/GettingStartedInPython.html):
```
conda create -n py36 python=3.6 anaconda
# activate the new virtual environment, e.g. on OSX/Linux
source activate py36
# on Windows
# activate py36
conda install -c rdkit rdkit

```
Salty can then be installed with:
```
pip install salty-ilthermo
```
##### Alternative versions of rdkit:
Should rdkit fail to install properly, refer to the [Anaconda rdkit package entry](https://anaconda.org/rdkit/rdkit)
and try:
```
#Recommended
conda install -c rdkit/label/beta rdkit 
```
```
conda install -c rdkit/label/attic rdkit 
```
```
conda install -c rdkit/label/testing rdkit
```

## Development

Salty is currently underdevelopment by researchers at the University of Washington. Our  research page can be found [here](http://www.prg.washington.edu).

### Testing

After installation, you can launch the test suite from outside the source directory (you will need to have the pytest package installed):
```
pytest salty
```

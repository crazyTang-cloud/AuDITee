*********************************************************************************************
## Dependencies
`AuDITe` work with Python 3.9+ **only**.

`AuDITee` require [numpy](www.numpy.org) to be already installed in your system. 
There are multiple ways to install `numpy`, and the easiest one is
using [pip](https://pip.pypa.io/en/stable/#):
```bash
$ pip install -U numpy
```

[Cython](https://cython.org/) is also required. 
Cython can be installed using pip:
```bash
$ pip install -U Cython
```

Another required package is [skmultiflow](https://scikit-multiflow.readthedocs.io/en/stable/installation.html). The most convenient way to install `skmultiflow` is from [conda-forge](https://anaconda.org/conda-forge/scikit-multiflow) using the following command:
```bash
$  conda install -c conda-forge scikit-multiflow
```
Note: this will install the latest (stable) release. 

Our code was developed with Python 3.9 and relied on the below packages.
- numpy version 1.21.5
- sklearn version 1.0.2
- scikit-multiflow version 0.5.3

But any python environment that satisfies the above requirements should also be workable.

We have already implement AuDITee in JGroups, Broadleaf (soon), and Tomcat (soon).

All testing resources are in dataset

We keep the dependency packages and scripts that are necessary for testing in dataset/dependencies and dataset/scripts

## run preparation
When you want to run AuDITee or test such project (i.e. JGroups), you need:

- firstly, using git pulls JGroups project from github in dataset/data/jgroups: ```git pull <url>```
- secondly, copy dependencies from dataset/dependencies and scripts from dataset/scripts/jgroups to dataset/data/jgroups.

When you want to implement AuDITee in other project:
- firstly, using git pulls <project> project from github in dataset/data/<project>.
- secondly, copy dependencies from dataset/dependencies to dataset/data/<project>.
- thirdly, write your own scripts similar to scripts in three projects above, make sure you can test it.


## run
you can easily use script```python test_main.py``` to run AuDITee quickly.



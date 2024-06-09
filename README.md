*********************************************************************************************
## Dependencies
`HumLa and ECo-HumLa` work with Python 3.5+ **only**.

`HumLa and ECo-HumLa` require [numpy](www.numpy.org) to be already installed in your system. 
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

## Structure
After setting up the Python environment, the code would be ready to run. 

There are two main folders in the replication package at first, namely `codes/` and `datasets/`:
- The folder `codes/` contains the code scripts implementing the submitted manuscript; 
- The folder `dataset/` contains the 3 datasets produced based on the GitHub open source projects as explained in Section 4 of the submitted paper.

The running the code will then create a third folder `results/` to store ongoing results.


### Core Scripts
The `code/` folder, as it is named, contains the core scripts that implement our RQs and the proposed AuDITee. The important two scripts are explained as follows.

- `run_RQs.py` implements RQ1 (1.1-1.2) and RQ2 (2.1-2.2). 
  - `RQ1()` implements RQ1.1 - RQ1.2;
  - `RQ2()` implements RQ2.1 - RQ2.2.

- `test_main.py` contains several important methods implementing AuDITee.


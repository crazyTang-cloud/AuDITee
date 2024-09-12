*********************************************************************************************
## Dependencies
`AuDITee` work with Python 3.9+, Java 8, and Maven 3.9.6.

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
``
Our code was developed with Python 3.9 and relied on the below packages.
- numpy version 1.21.5
- sklearn version 1.0.2
- scikit-multiflow version 0.5.3

But any python environment that satisfies the above requirements should also be workable.

We have already implement AuDITee in JGroups, Broadleaf, and Tomcat.

All testing resources are in dataset

We keep the dependency packages and scripts that are necessary for testing in ```dataset/dependencies``` and ```dataset/scripts```

## run preparation
When you want to run AuDITee or test such project (i.e. JGroups), you need:

- firstly, using git pulls JGroups project from github in dataset/data/jgroups: ```git pull <url>```
- secondly, copy dependencies from ```dataset/dependencies``` and scripts from ```dataset/scripts/jgroups``` to ```dataset/data/jgroups```.

When you want to implement AuDITee in other project:
- firstly, using git pulls ```<project>``` project from github in ```dataset/data/<project>```.
- secondly, copy dependencies from ```dataset/dependencies``` to ```dataset/data/<project>```.
- thirdly, write your own scripts similar to scripts in three projects above, make sure you can test it.

Make sur you have java environment can compile and run such project, i.e. jdk8.
If you run a maven project make sure you have maven.

## run
### run framework
The main combination functions are in ```codes/```.
First you need```cd codes/```.
Then you can easily use script ```python AuDITee_framework.py``` to run AuDITee.

The main function is ```run_AuDITee()```, with required parameters project id, we define it in real_data_stream.data_id_2name

Note:

1 Make sure your network status when running AuDITee in maven projects.
The first time running AuDITee in maven projects may need download dependencies from remote repositories.
It will take a lot of time.

2 When running AuDITee in Tomcat:
first ```cd dataset/data/tomcat``` ```mkdir classes/```,
second ```tar -zxf apache-tomcat-8.0.1-deployer.tar.gz```, apache-tomcat-8.0.1-deployer.tar.gz has the dependencies tomcat requires.

### run framework with our testing results
Testing is costly and time-consuming. If you only want to verify whether AuDITee works. 
We provide the results of our tests with three different seeds.
All testing results are in ```dataset/data/<project>_testing_results```
First you need```cd codes/```.
Then you can easily use script ```python AuDITee_exp.py``` to run AuDITee quickly.

The final results are in ```dataset/data/results/<project>/AuDITee_<versions>```

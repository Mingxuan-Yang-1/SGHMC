# SGHMC

This is the final project of STA 663: Statistical Computation and Programming in 2020 Spring Semester at Duke University, finished by Jiawei Chen and Mingxuan Yang. 

This repository contains a package implementing Stochastic Gradient Hamiltonian Monte Carlo algorithm proposed by Chen et. al.

Reference:

Tianqi Chen, Emily B. Fox, Carlos Guestrin "Stochastic Gradient Hamiltonian Monte Carlo" ICML 2014

## Introduction

Here we give a brief introduction of the folders in this repo:

### SGHMC

This folder contains the source code of our package. The major part of the code is in the folder *sghmc_pkg*.

In folder *Simulation*, the basic functions used in the **3.1** section of the report are provided, which contain the algorithms of standard HMC, naive SGHMC, SGHMC and SGLD under different conditions (unidimensional or multidimensional case, with or without Metropolis Hastings, resample or not).

In folder *Data*, the basic functions used in the **3.2** section of the report are provided, which contain the algorithms of SGHMC, SGLD and SGD with momentum under different conditions (with or without Metropolis Hastings, resample or not). It needs some attention that for functions in this folder, we assume the input data set should contain at least two predictors.

Based on the functions in *Simulation* and *Data* folders, *algorithms_sim.py* and *algorithms_data.py* attach method theta and r to each of the algorithms and make the package easier to use.

### Tests

This folder contains the code used to conduct unit tests for our package.

### Examples

This folder contains the code used to produce the plots in our report.

### Report

This folder contains the project report.

## Installing

This package can be viewed on [TestPyPI](https://test.pypi.org/project/sghmc-pkg-663-2.0.0/2.0.0). The preferred installation is through `pip`:
```
pip install -i https://test.pypi.org/simple/ sghmc-pkg-663-2.0.0==2.0.0
```

## Authors

- Jiawei Chen
- Mingxuan Yang

## License

This project is licensed under the MIT License - see the [LICENSE](SGHMC/LICENSE) file for details.

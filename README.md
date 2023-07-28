# TDAOpt

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sidv23.github.io/TDAOpt.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sidv23.github.io/TDAOpt.jl/dev/)
[![Build Status](https://github.com/sidv23/TDAOpt.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sidv23/TDAOpt.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sidv23/TDAOpt.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sidv23/TDAOpt.jl)


`TDAOpt.jl` is a Julia package to optimize statistical and topological loss functions defined on point-clouds and functions defined on fixed grids. 



---

## §2 $-$ Overview

There are 3 (and a haf) main functions exported by the current API.

* `dist()` : computes a distance between two point-clouds/measures
* `val()` : evaluates a loss functional on a source point-cloud/measure
* `backprop()` : performs minimization of loss functional starting from a source point-cloud/measure
    * `∇w` : computes the Wasserstein gradient of a loss functional (used in `backprop` when specified)

These functions belong to the following parts of the current API:

#### §2.1 $-$ Discrepancies

A discrepancy is a `struct` which configures the parameters for measuring a distance between two input matrices (`x` and `y`), or two input measures (`μ` and `ν`). The ones currently implemented are:

* Statistical (MMD, Sinkhorn)
* Topological (Wasserstein Matching)

Every discrepancy should dispatch on (i.e. extend) the function `dist`


#### §2.2 $-$ Losses

Losses are `structs` which define methods of computing loss functionals. The abstract supertype is `AbstractLossFunction`. Every loss is expected to extend the function `val` which takes in a `source` (i.e. matrix `x` or measure `μ`) and an `AbstractLossFunction`, and evaluates the loss functional $\ell(\texttt{source})$ to be minimized.



Currently, the implemented `AbstractLossFunctions` fall into the following categories:

* `StatLoss`: Fixes a reference discrepancy `d` and `target` (matrix or measure)
    
    > `val(source, Loss) = dist(Loss.d, source, Loss.target)`


* `TopLoss`: Fixes a reference discrepancy `d`, a persistence diagram constructor `dgmFun` and `target` (persistence diagram)

    > `val(source, Loss) = dist(Loss.d, Loss.dgmFun(source), Loss.target)`

* `BarycenterStatLoss`: Fixes a reference discrepancy `d` along with `targets` $\{\nu_1, \nu_2 \dots \nu_M\}$ and the `weights` $\{\lambda_1, \lambda_2, \dots, \lambda_n\}$

    > `val(source, Loss) = ` $\sum\limits_{i=1}^{M}$ `weights[i] * dist(Loss.d, source, Loss.target)^2`


* `BarycenterTopLoss`: Fixes a reference discrepancy `d` along with `dgmFun`, the precomputed persistence diagram `targets` $\{D_1, D_2 \dots D_M\}$ and the `weights` $\{\lambda_1, \lambda_2, \dots, \lambda_n\}$

    > `val(source, Loss) = ` $\sum\limits_{i=1}^{M}$ `weights[i] * dist(Loss.d, Loss.dgmFun(source), Loss.target)^2`

#### §2.2 $-$ Abstract Backprop

An `AbstractBackprop` object configures the parameters for performing backpropagation for a specified `AbstractLossFunction`. It dispatches different instances of the function `backprop`.


The main difference between an `AbstractBackprop` and `AbstractGradflow` is how the gradients are computed. `AbstractBackprop` methods compute the usual gradients while `AbstractGradflow` computes the Wasserstein gradient (or using the JKO method).
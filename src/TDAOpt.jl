# TDAOpt.jl

module TDAOpt

using Distances
using Flux
using KernelFunctions
using LinearAlgebra
using NNlib
using Pipe
using Random
using Ripserer
using Plots
using ProgressMeter
using Setfield
using Statistics
using StatsBase
using Zygote

import Base: eltype, length, @kwdef

######################################################################
# Abstract Types

export AbstractLossFunction, AbstractDiscrepancy, AbstractBackprop, AbstractGradflow, StatDiscrepancy, TDADiscrepancy
include("structs.jl")




######################################################################
# Main functions

export dist, val, backprop, ∇w



######################################################################
# Helper Functions

# Plotting functions
export plotpath, make_gif, make_gif2

# Type conversion functions
export m2t, m2a, t2m, t2a, a2t, a2m, p2m, p2nm, points_to_measure, pers,points_to_norm_measure

include("utils.jl")



######################################################################
# Shapes

export randSphere, randSpiral, randCross, randLemniscate, halfMoon, gaussMix, sphereMix

include("shapes.jl")



######################################################################
# Measures
export Measure, mass
include("measures.jl")



######################################################################
# TDA Functions

# weighted Distance Functions
export wDist_0, wDist_1, wDist_2, wDist_inf

# Differentiable Rips complex
export get_indices, rips_indices, persistences, dgm_0, dgm_1, dgm_inf
include("tda.jl")

# Differentiable Cubical complex
export dgm_cubical, cubical_indices, supporting_indices
include("cubical.jl")



######################################################################
# Persistence Dgm Metrics
export Matching, FastMatching, dist, W1, W2, Winf, W
include("dgm-metrics.jl")
include("dgm-fast-metrics.jl")


######################################################################
# Statistical Metrics

######################
# MMD

export MMD, mmd, ∇₂K, ∇W_mmd
include("mmd.jl")


######################
# OT Metrics

export Sinkhorn, SinkhornDiv, costMatrix, sinkhorn, sinkhorn_cost, sinkhorn_div, sinkhorn_loss, sinkhorn_div_loss, sink_cost, sink_div_cost, sink_loss, sink_div_loss
include("sinkhorn.jl")



######################################################################
# Loss Functionals

export StatLoss, TDALoss, BarycenterStatLoss, BarycenterTDALoss, val
include("losses.jl")



######################################################################
# Backprop Functions

export AbstractBackprop, Backprop, TiltedBackprop, AlternatingBackprop, ScheduledBackprop, backprop

include("backprop.jl")


######################################################################
# Wasserstein Gradient Flow

export barycenters, baryplot, weightmatrix

include("barycenter.jl")


######################################################################
# Wasserstein Gradient Flow

# export ∇₂K, mmd_gradflow, ∂W_forwardEuler!, ∂W_backwardEuler!, gradflow
export FwGradflow, BwGradflow, FwBwGradflow
include("gradflow.jl")



# #####################################################################
# # Test Metrics
# export get_pairings, W
# include("distances.jl")


end
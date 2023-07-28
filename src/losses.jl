######################################################################

@kwdef struct StatLoss <: AbstractLossFunction
    d::StatDiscrepancy
    target::Union{Matrix{Float64},Measure}
    scale::T where {T<:Real} = 1.0
end

val(source, l::StatLoss) = l.scale * dist(l.d, source, l.target)
∇w(source, l::StatLoss) = l.scale * ∇w(l.d, source, l.target)


######################################################################


@kwdef struct TDALoss <: AbstractLossFunction
    d::TDADiscrepancy
    dgmFun::Function
    target::Union{Vector{Matrix{Float64}},Matrix{Float64}}
end

val(source, l::TDALoss) = dist(l.d, l.dgmFun(source), l.target)



######################################################################


@kwdef struct BarycenterStatLoss <: AbstractLossFunction
    d::StatDiscrepancy
    targets::Union{Vector{Matrix{Float64}},Vector{Measure}}
    weights::Vector{T} where {T<:Real}
end

val(source, l::BarycenterStatLoss) = l.weights' * [dist(l.d, source, target)^2 for target in l.targets]

∇w(source, l::BarycenterStatLoss) = sum([∇w(l.d, source, l.targets[i]) .* l.weights[i] for i in eachindex(l.targets)])



######################################################################


@kwdef struct BarycenterTDALoss <: AbstractLossFunction
    d::TDADiscrepancy
    dgmFun::Function
    targets::Vector{M} where {M<:Union{Vector{Matrix{Float64}},Matrix{Float64}}}
    weights::Vector{T} where {T<:Real}
end

val(source, l::BarycenterTDALoss) = l.weights' * [dist(l.d, l.dgmFun(source), target)^2 for target in l.targets]
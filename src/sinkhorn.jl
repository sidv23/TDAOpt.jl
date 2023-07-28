using Distances, NNlib

######################################################################

# Sinkhorn cost

@kwdef struct Sinkhorn <: StatDiscrepancy
    ϵ::T where {T<:Real} = 1e-2
    iters::Integer = 20
end


# Sinkhorn Divergence cost

@kwdef struct SinkhornDiv <: StatDiscrepancy
    ϵ::T where {T<:Real} = 1e-2
    iters::Integer = 20
end



######################################################################
# Sinkhorn Implementation


function costMatrix(d::T, μ::M, ν::M) where {M<:Measure,T<:PreMetric}
    return pairwise(d, μ.support, ν.support, dims=1)
end



function sinkhorn(C::Matrix{Float64}, μ::M, ν::M; ϵ::Float64=1e-1, iters::T=5)::Matrix{Float64} where {M<:Measure,T<:Integer}
    log_P = -C ./ ϵ
    log_μ = μ.log_density
    log_ν = ν.log_density'

    for _ in 1:iters
        log_P = log_P .- (NNlib.logsumexp(log_P, dims=1) - log_ν)
        log_P = log_P .- (NNlib.logsumexp(log_P, dims=2) - log_μ)
    end
    P = exp.(log_P)
    return P
end


function sinkhorn_cost(C::Matrix{Float64}, μ::M, ν::M; ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,T<:Integer}
    P = sinkhorn(C, μ, ν; ϵ=ϵ, iters=iters)
    return dot(P, C)
end



function sinkhorn_div(d::S, μ::M, ν::M; ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,S<:PreMetric,T<:Integer}
    C11 = costMatrix(d, μ, μ)
    P11 = sinkhorn(C11, μ, μ; ϵ=ϵ, iters=iters)

    C22 = costMatrix(d, ν, ν)
    P22 = sinkhorn(C22, ν, ν; ϵ=ϵ, iters=iters)

    C12 = costMatrix(d, μ, ν)
    P12 = sinkhorn(C12, μ, ν; ϵ=ϵ, iters=iters)

    div = ϵ * (dot(P12, C12) - 0.5 * dot(P11, C11) - 0.5 * dot(P22, C22))
    return div
end



function sinkhorn_loss(μ::M, ν::M; ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,T<:Integer}
    C = costMatrix(SqEuclidean(), μ, ν)
    return sinkhorn_cost(C, μ, ν; ϵ=ϵ, iters=iters)
end



function sinkhorn_div_cost(μ::M, ν::M; d::S=Euclidean(), ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,S<:PreMetric,T<:Integer}
    d = Euclidean()

    C11 = costMatrix(d, μ, μ)
    P11 = sinkhorn(C11, μ, μ; ϵ=ϵ, iters=iters)

    C22 = costMatrix(d, ν, ν)
    P22 = sinkhorn(C22, ν, ν; ϵ=ϵ, iters=iters)

    C12 = costMatrix(d, μ, ν)
    P12 = sinkhorn(C12, μ, ν; ϵ=ϵ, iters=iters)

    div = ϵ * (dot(P12, C12) - 0.5 * dot(P11, C11) - 0.5 * dot(P22, C22))
    return div
end


####################################################################################
# Fast implementation with stop gradients


# Fast with stop-gradient
function sink_loss(μ::M, ν::M; ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,T<:Integer}
    C = costMatrix(SqEuclidean(), μ, ν)
    P = Zygote.@ignore sinkhorn(C, μ, ν; ϵ=ϵ, iters=iters)
    return dot(P, C)
end


# Stop-gradient + Normalized Measures
function sink_cost(x::Matrix{F}, y::Matrix{F}; ϵ::Float64=1e-1, iters::T=5)::Float64 where {F<:Real,T<:Integer}
    μ, ν = points_to_norm_measure.((x, y))
    C = costMatrix(SqEuclidean(), μ, ν)
    P = Zygote.@ignore sinkhorn(C, μ, ν; ϵ=ϵ, iters=iters)
    return dot(P, C)
end


# Fast with stop-gradient
function sink_div_cost(μ::M, ν::M; d::S=Euclidean(), ϵ::Float64=1e-1, iters::T=5)::Float64 where {M<:Measure,S<:PreMetric,T<:Integer}
    d = SqEuclidean()

    C11 = costMatrix(d, μ, μ)
    P11 = Zygote.@ignore sinkhorn(C11, μ, μ; ϵ=ϵ, iters=iters)

    C22 = costMatrix(d, ν, ν)
    P22 = Zygote.@ignore sinkhorn(C22, ν, ν; ϵ=ϵ, iters=iters)

    C12 = costMatrix(d, μ, ν)
    P12 = Zygote.@ignore sinkhorn(C12, μ, ν; ϵ=ϵ, iters=iters)

    div = ϵ * (dot(P12, C12) - 0.5 * dot(P11, C11) - 0.5 * dot(P22, C22))
    return div
end



# Stop-gradient + Normalized Measures
function sink_div_cost(x::Matrix{F}, y::Matrix{F}; d::S=Euclidean(), ϵ::Float64=1e-1, iters::T=5)::Float64 where {F<:Real,S<:PreMetric,T<:Integer}
    d = SqEuclidean()

    μ, ν = points_to_norm_measure.((x, y))

    C11 = costMatrix(d, μ, μ)
    P11 = Zygote.@ignore sinkhorn(C11, μ, μ; ϵ=ϵ, iters=iters)

    C22 = costMatrix(d, ν, ν)
    P22 = Zygote.@ignore sinkhorn(C22, ν, ν; ϵ=ϵ, iters=iters)

    C12 = costMatrix(d, μ, ν)
    P12 = Zygote.@ignore sinkhorn(C12, μ, ν; ϵ=ϵ, iters=iters)

    div = ϵ * (dot(P12, C12) - 0.5 * dot(P11, C11) - 0.5 * dot(P22, C22))
    return div
end




####################################################################################
# Dist definitions for Sinkhorn & Sinkhorn Divergence


function dist(d::Sinkhorn, x::Matrix{T}, y::Matrix{T}) where {T<:Real}
    sink_cost(x, y; ϵ=d.ϵ, iters=d.iters)
end


function dist(d::SinkhornDiv, x::Matrix{T}, y::Matrix{T}) where {T<:Real}
    sink_div_cost(x, y; ϵ=d.ϵ, iters=d.iters)
end

using Base: @kwdef

##########################################
# TDA Matching Distance

@kwdef struct Matching <: TDADiscrepancy
    ϵ::Real = 1e-3
    iters::Real = 1000
    weights::Vector{Float64} = [0.5, 0.5]
end


###################################
##### Diagram Distance Functions

function diag_dist(x)
    return map(x_ -> x_ .- mean(x_), eachrow(x)) .|> norm
end

function unbalanced_cost(x, y)
    c = pairwise(SqEuclidean(), x, y, dims=1)
    C = [c diag_dist(x); diag_dist(y)' 0]
    return C
end

function pd_snk(C::M; ϵ::Real=1.0, iters::Int=100) where {
    M<:AbstractMatrix
}
    log_P = -C ./ ϵ
    m, n = size(C) .- 1 
    log_a = log.([fill(1, m); n] ./ (m + n))
    log_b = log.([fill(1, n); m] ./ (m + n))'

    for _ in 1:iters
        # log_P .-= (StatsFuns.logsumexp(log_P, dims=1) .- log_b)
        # log_P .-= (StatsFuns.logsumexp(log_P, dims=2) .- log_a)
        log_P .-= (NNlib.logsumexp(log_P, dims=1) .- log_b)
        log_P .-= (NNlib.logsumexp(log_P, dims=2) .- log_a)
    end

    P = exp.(log_P)
    # cost = sum(P .* C)
    # return cost
    return P
end

function W_ot(Dgmx, Dgmy; ϵ::Real=1e-1, iters::Real=100, w=nothing)

    N = Zygote.@ignore minimum(length.((Dgmx, Dgmy)))
    w = isnothing(w) ? fill(1 / N, N) : w
    result = 0.0

    for i in 1:N
        dgmx, dgmy = Dgmx[i], Dgmy[i]
        C = unbalanced_cost(dgmx, dgmy)
        P = Zygote.@ignore pd_snk(C; ϵ=ϵ, iters=iters)
        result = result + sum(C .* P)
    end
    return result
end

####################################################################################

function dist(d::Matching, dgmx::U, dgmy::U) where {U<:Union{Vector{Matrix{Float64}},Matrix{Float64}}}
    # W(dgmx, dgmy; p=d.p, w=d.weights)
    W_ot(dgmx, dgmy; ϵ=d.ϵ, iters=d.iters, w=d.weights)
end


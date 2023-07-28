
using Base: @kwdef

##########################################
# Fast TDA Matching Distance

@kwdef struct FastMatching <: TDADiscrepancy
    p::Real = 2
    weights::Vector{Float64} = [0.5, 0.5]
end

###################################
##### Diagram Distance Functions

function W_fast(Dgmx, Dgmy; p::Real=2, w=nothing)

    N = Zygote.@ignore minimum(length.((Dgmx, Dgmy)))
    w = isnothing(w) ? fill(1 / N, N) : w
    result = 0.0

    for i in 1:N
        dgmx, dgmy = Dgmx[i], Dgmy[i]
        nx, ny = size.((dgmx, dgmy), 1)

        if ny < nx
            padding = @pipe dgmx[1:(nx-ny), :] |> 
                mean.(eachrow(_)) |> [_ _]
            # padding = dgmx[1:(nx-ny), 1] .+ zeros(nx - ny, 2)
            dgmy = [padding; dgmy]
        elseif nx < ny
            dgmy = dgmy[end-nx+1:end, :]
        end

        res = maximum.(eachrow(abs.(dgmx .- dgmy)))
        if !isempty(res)
            result = result + w[i] * norm(res, p)
        end
    end

    return result
end

W1(Dgmx, Dgmy; w=nothing) = W_fast(Dgmx, Dgmy; p=1, w=w)
W2(Dgmx, Dgmy; w=nothing) = W_fast(Dgmx, Dgmy; p=2, w=w)
Winf(Dgmx, Dgmy; w=nothing) = W_fast(Dgmx, Dgmy; p=Inf, w=w)


####################################################################################

function dist(d::FastMatching, dgmx::U, dgmy::U) where {U<:Union{Vector{Matrix{Float64}},Matrix{Float64}}}
    W_fast(dgmx, dgmy; p=d.p, w=d.weights)
end
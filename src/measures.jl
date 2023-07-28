# These structures are essentially identical to https://github.com/ericphanson/UnbalancedOptimalTransport.jl

struct Measure{P,S}
    density::P
    log_density::P
    support::S
end

function Measure(density::P, support::S) where {P,S}
    T = eltype(density)
    n = length(density)
    log_density = log.(density)
    Measure{P,S}(density, log_density, support)
end

mass(μ::Measure) = sum(μ.density)
Base.length(μ::Measure) = size(μ.density, 1)
Base.eltype(::Measure{P,S}) where {P,S} = S

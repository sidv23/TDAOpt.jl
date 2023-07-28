using TDAOpt
using Test
using Random
using Ripserer
using Distances

@testset "TDAOpt.jl" begin
    x = randn(100, 2)
    dgm(x) = dgm_0(x; order=1)
    @test typeof(dgm(x)) <: Vector{Matrix{Float64}}
    # More to come later
end

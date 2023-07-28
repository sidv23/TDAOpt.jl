function randSphere(n; dim=1, s=0.0)
    X = randn(n, dim + 1)
    X = X ./ norm.(eachrow(X))
    return X .+ (s .* randn(n, dim + 1))
end


function randSpiral(n; k=5, a=1.0, b=1.0, s=0.0)
    θ = range(0, 2 * k * π, n)
    r = a .+ b .* θ
    return hcat(r .* cos.(θ), r .* sin.(θ)) .+ (s .* randn(n, 2))
end


function randCross(n; s=0.1, min=-3.0, max=3.0, seed=2022)
    Random.seed!(seed)
    x = range(min, max, n)
    return vcat(hcat(x, -x .+ s .* randn(n)), hcat(x, x .+ s .* randn(n)))
end


function randLemniscate(n; a=1.0, s=0.1, seed=2022)
    Random.seed!(seed)
    θ = range(0, 2π, n)
    x = a .* sin.(θ)
    y = a .* sin.(θ) .* cos.(θ)
    return hcat(x, y) .+ s .* randn(n, 2)
end


function halfMoon(;n, m, s=0.1)
    angle1 = range(0, π; length=n)
    angle2 = range(0, π; length=m)
    X1 = [cos.(angle1) sin.(angle1)] .+ s .* randn.()
    X2 = [1 .- cos.(angle2) 1 .- sin.(angle2) .- 0.5] .+ s .* randn.()
    return [X1; X2]
end


function gaussMix(n; m, s)
    return vcat([x[2] .* randn(n, 2) .+ [x[1]...]' for x in zip(m, s)]...)
end


function sphereMix(n; m, s)
    return vcat([x[2] .* randSphere(n; dim=1) .+ [x[1]...]' for x in zip(m, s)]...)
end

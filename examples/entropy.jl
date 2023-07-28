using DrWatson
@quickactivate "tdaopt"

begin
    using TDAOpt
    using Plots
    using LinearAlgebra
    using Zygote
    using ProgressMeter
    using Optimisers
    using Flux
    using Pipe
end

function foo(X, ℓ, N, opt; α::Real=0.9)
    optim = Flux.setup(opt, X)
    xs = fill(copy(X), N+1)

    @showprogress for i in 1:N
        if (i > 1) && (i % 10 == 0)
            Optimisers.adjust!(optim, α * optim.rule.eta)
        end
        g = Zygote.gradient(ℓ, X)[1]
        Flux.Optimise.update!(optim, X, g)
        xs[i+1] = copy(X)
    end
    return xs
end

dgm = x -> dgm_0(x; m=0.0)
lifetime(D) = map(x -> diff(x)[1], eachrow(D))

function persistent_entropy(Dx; order::Int=0)
    dx = Dx[1+order]
    pers_x = lifetime(dx)
    npers_x = pers_x ./ sum(pers_x)
    return -sum(npers_x .* log.(npers_x))
end

function norm_pers(Dx; order::Int=0)
    dx = Dx[1+order]
    pers_x = lifetime(dx)
    return norm(pers_x, 2)
end

begin
    function ℓ(x)
        dx = dgm(x)
        res = persistent_entropy(dx; order=1)
        return -res
    end

    X = 3 .* rand(200, 2) .- 1.5
    path = foo(X, ℓ, 100, NADAM(0.05), α=0.99)
    @gif for i in eachindex(path)
        scatter(Tuple.(eachrow(path[i])), label="", lim=(-3, 3), title="epoch: $i", size=(450, 450))
    end
end



begin
    function ℓ(x)
        dx = dgm(x)
        res0 = persistent_entropy(dx; order=0) - norm_pers(dx; order=0)
        res1 = persistent_entropy(dx; order=1) + norm_pers(dx; order=1)
        return -(0.01 * res0 + res1)
    end

    X = 3 .* rand(200, 2) .- 1.5
    path = foo(X, ℓ, 300, NADAM(0.05))
    @gif for i in eachindex(path)
        scatter(Tuple.(eachrow(path[i])), label="", lim=(-3, 3), title="epoch: $i", size=(450, 450))
    end
end
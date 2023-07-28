using Plots
using LinearAlgebra
using Zygote
using ProgressMeter
using Optimisers
using Flux
using Pipe

X = rand(50, 50)

function foo(X, ℓ, N, opt; α::Real=0.9)
    optim = Flux.setup(opt, X)
    xs = fill(copy(X), N + 1)

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


dgm = x -> dgm_cubical(x; order=1)
dgmnorm(Dx; order, p) = @pipe Dx[1+order] |> 
                                eachrow .|> 
                                pers |> 
                                norm(_, p)

function persistent_entropy(Dx; order::Int=0)
    dx = Dx[1+order]
    pers_x = pers.(eachrow(dx))
    npers_x = pers_x ./ sum(pers_x)
    return -sum(npers_x .* log.(npers_x))
end

function ℓ(x)
    dx = dgm(x)
    # return -dgmnorm(dx; order=1, p=25) + 1e-2 * dgmnorm(dx; order=0, p=2)
    return persistent_entropy(dx; order=1) - persistent_entropy(dx; order=0) * 1e-3
end

N = 100
opt = NADAM(0.1)
path = foo(copy(X), ℓ, N, opt, α=2.5)
@gif for i in eachindex(path)
    plot(Gray.(path[i]), 
        label="", title="epoch: $i",
        grid=false, axis=false,
    )
end
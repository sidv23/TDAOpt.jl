using DrWatson
@quickactivate "tdaopt"

begin
    using Pipe, Plots, ProgressMeter, Random
    using Flux, Zygote
    using Images, HTTP
    using TDAOpt

    ProgressMeter.ijulia_behavior(:clear)
    default(msw=0.5)
end

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

dgm = x -> dgm_cubical(x)

function persistent_entropy(Dx; order::Int=0)
    dx = Dx[1+order]
    pers_x = pers.(eachrow(dx))
    npers_x = pers_x ./ sum(pers_x)
    return -sum(npers_x .* log.(npers_x))
end

function ℓ_tda(x)
    dx = dgm(x)
    loss = persistent_entropy(dx; order=1) - 1e-3 * persistent_entropy(dx; order=0)
    return loss
end

img = load(datadir("img_128x128.jpg"))
X = convert.(Float64, img)

path = foo(copy(X), ℓ_tda, 50, NADAM(0.05), α=0.75)


@gif for i in eachindex(path)
    plot(Gray.(path[i]),
        label="", title="epoch: $i",
        grid=false, axis=false,
    )
end

p1 = plot(Gray.(path[1]), grid=false, axis=false, title="Original")
p2 = plot(Gray.(path[end]), grid=false, axis=false, title="TDA Adjusted")

function segment_image(x; threshold=0.0)
    
    dg = ripserer(Cubical(x), reps=true, alg=:homology)
    per = persistence.(dg[2])
    scores = (per .- median(per))  / mad(per)
    
    indx = findall(x -> x .> threshold, scores)
    p = plot(zeros(size(x)) .|> Gray, grid=false)
    for i in indx
        p = plot!(dg[2][i], zeros(size(x)), label="", c=:white)
    end
    return plot(p, grid=false, axis=false)
end

p3 = segment_image(path[1  ])
p4 = segment_image(path[end])

plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800))













# Load Image
img = load(datadir("img_128x128.jpg"))

# Convert from Grayscale to Array
X = convert.(Float64, img)

# Loss function
function ℓ_tda(x)
    dx = dgm(x)
    loss = persistent_entropy(dx; order=1) - 
            1e-3 * persistent_entropy(dx; order=0)
    return loss
end

# Median Segmentation
function segment_image(x; threshold=5.0)
    # Construct the diagram with representative cycles
    dg = ripserer(Cubical(x), reps=true, alg=:homology)
    
    # Filter points which are are greater than the robust score
    per = persistence.(dg[2])
    scores = (per .- median(per)) / mad(per)
    indx = findall(x -> x .> threshold, scores)
    
    # Plot the cycle
    p = plot(zeros(size(x)) .|> Gray, grid=false)
    for i in indx
        p = plot!(dg[2][i], zeros(size(x)), label="", c=:white)
    end
    return plot(p, grid=false, axis=false)
end

# TDAOpt
path = foo(copy(X), ℓ_tda, 20, NADAM(0.05), α=0.75)







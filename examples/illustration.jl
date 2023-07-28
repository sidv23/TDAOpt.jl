using DrWatson
@quickactivate "tdaopt"

begin
    using Distances, LinearAlgebra, Pipe, Plots, ProgressMeter, Random
    using CSV, DataFrames
    using Flux, Zygote
    using KernelFunctions
    using TDAOpt

    ProgressMeter.ijulia_behavior(:clear)
    default(msw=0.5)
end

begin
    function ℓ_sinkhorn(x, y; ϵ=2e-2)
        μ, ν = points_to_norm_measure.((x, y))
        # return sinkhorn_div_loss(μ, ν; d=SqEuclidean(), ϵ=ϵ, iters=10)
        return sinkhorn_loss(μ, ν; ϵ=ϵ, iters=50)
    end


    function ℓ_PD(x, Dgm; w=[0.5, 0.5])
        return W1(x, Dgm, w=w)
    end


    function generate_data(; m=200, n=150, seed=2023)
        Random.seed!(seed)
        Y = randLemniscate(m; s=0.05)
        X = 2.1 .* rand(n, 2) .- 2.5
        return (X, Y)
    end
end

begin
    X, Y = generate_data()
    baseplot(; lim=(-3, 1.5)) = scatter(Y |> m2t, ratio=1, label="target", lim=lim, ma=0.7, msw=0.0)
    scatter(baseplot(), X |> m2t, ratio=1, label="source", legend=:bottomright, ma=0.7, msw=0.0, size=(400, 400))
end

function pplot(path; plt=plot(), cls=nothing, mid=nothing)
    if isnothing(cls)
        # cls = cgrad(:viridis, range(0.5, 1, length=length(path)), scale=:log10) |> reverse
        cls = cgrad(:viridis, range(0.5, 1, length=length(path))) |> reverse
    end
    if isnothing(mid)
        mid = round(Int, length(path) / 2)
    end
    plt0 = scatter(plt, path[1] |> m2t, c=cls[1], label="", legend=:bottomright)
    plt1 = plt0
    for i in eachindex(path)
        plt = scatter(plt, path[i] |> m2t, c=cls[i], ms=3, ma=0.5, label=false, msw=0)
        if i == mid
            plt1 = scatter(plt, path[i] |> m2t, ms=3, c=cls[i], ma=1.0, label="", legend=:bottomright)
        end
    end
    plt2 = scatter(plt, path[end] |> m2t, ms=3, c=cls[end], ma=1.0, label="", legend=:bottomright)
    return plt0, plt1, plt2
end

begin
    # kernel = SqExponentialKernel() ∘ ScaleTransform(1 / 2)
    kernel = SqExponentialKernel() ∘ ScaleTransform(5)
    dgm = x -> dgm_0(x; m=0.05)

    d_mmd = MMD(k=kernel, unbiased=false)
    d_sink = Sinkhorn(ϵ=1e-2, iters=50)
    d_tda = FastMatching(p=1, weights=[0.2, 0.8])
    # d_tda = FastMatching(ϵ=1e-3, iters=1000, weights=[0.2, 0.75])
    # d_tda = FastMatching(ϵ=1e-3, iters=500, weights=[0.5, 0.5])

    loss_mmd = StatLoss(d=d_mmd, target=Y)
    loss_sink = StatLoss(d=d_sink, target=Y)
    loss_tda = TDALoss(d=d_tda, dgmFun=dgm, target=dgm(Y))
end

begin
    method_sink = Backprop(
        loss=loss_sink,
        opt=NADAM(0.1),
        epochs=100
    )

    method_mmd = Backprop(
        loss=loss_mmd,
        opt=NADAM(0.1),
        epochs=100
    )

    method_sink_tda = AlternatingBackprop(
        loss1=loss_sink,
        opt1=NADAM(0.1),
        loss2=loss_tda,
        opt2=NADAM(0.1),
        epochs=100
    )

    method_mmd_tda = AlternatingBackprop(
        loss1=loss_mmd,
        opt1=NADAM(0.07),
        loss2=loss_tda,
        opt2=NADAM(0.1),
        epochs=200
    )
    method_tda1 = Backprop(
        loss=TDALoss(d=FastMatching(p=2, weights=[0.7, 0.3]), dgmFun=dgm, target=dgm(Y)),
        # loss=TDALoss(d=FastMatching(ϵ=1e-2, iters=100, weights=[0.7, 0.3]), dgmFun=dgm, target=dgm(Y)),
        opt=NADAM(0.02),
        epochs=50
    )
    method_tda2 = Backprop(
        loss=TDALoss(d=FastMatching(p=2, weights=[0.3, 0.7]), dgmFun=dgm, target=dgm(Y)),
        # loss=TDALoss(d=FastMatching(ϵ=1e-2, iters=500, weights=[0.5, 0.5]), dgmFun=dgm, target=dgm(Y)),
        opt=NADAM(0.02),
        epochs=200
    )
end

# _, path1 = backprop(deepcopy(X), method_tda2);
# _, path2 = backprop(deepcopy(path1[end]), method_tda1);
# path = [path1; path2];
_, path = backprop(deepcopy(X), method_mmd);
_, path = backprop(deepcopy(X), method_mmd_tda);
# _, path = backprop(deepcopy(X), method_tda2);
# pathalt = copy(path); path = pathalt[1:200]
plotpath(path,
    plt=baseplot(lim=(-6.5, 2.5)),
    title="Epoch: ", legend=:bottomright
)


# plot(
#     pplot(path, plt=baseplot(), mid=40)...,
#     title=["Initial" "Intermediate" "Final"],
#     size=(1200, 300), layout=(1, 3)
# )

function weightmatrix(dim)
    M = Matrix{Vector{Float64}}(undef, dim, dim)
    for j in 1:dim, i in 1:dim
        M[i, j] = [(dim - i) * (dim - j), (j - 1) * (dim - i), (i - 1) * (dim - j), (i - 1) * (j - 1)]
    end
    return M
end

function barycenters(n::Integer, B::AbstractBackprop; plotsdim::Integer=2)

    dim = plotsdim
    M = weightmatrix(dim)

    # if typeof(B) <: Union{Backprop,FwGradflow}
    #     res = repeat([deepcopy(B.loss.targets[1])], dim, dim)
    # elseif typeof(B) <: Union{TiltedBackprop,AlternatingBackprop,ScheduledBackprop}
    #     res = repeat([deepcopy(B.loss1.targets[1])], dim, dim)
    # elseif typeof(B) <: Union{BwGradflow}
    #     res = repeat([deepcopy(B.back.loss2.targets[1])], dim, dim)
    # elseif typeof(B) <: Union{FwBwGradflow}
    #     res = repeat([deepcopy(B.backward.back.loss2.targets[1])], dim, dim)
    # end

    res = repeat([randn(n, 2)], dim, dim)

    p = Progress(dim^2)

    for j in 1:dim, i in 1:dim

        next!(p)


        Random.seed!(2022)
        X = 2 .* rand(n, 2) .- 1

        if typeof(B) <: Union{VanillaBackprop,FwGradflow}

            B.loss = BarycenterStatLoss(B.loss.d, B.loss.targets, M[i, j])

        elseif typeof(B) <: Union{TiltedBackprop,AlternatingBackprop,ScheduledBackprop}

            B.loss1 = BarycenterStatLoss(B.loss1.d, B.loss1.targets, M[i, j])
            B.loss2 = BarycenterTDALoss(B.loss2.d, B.loss2.dgmFun, B.loss2.targets, M[i, j])

        elseif typeof(B) <: Union{BwGradflow}

            B.back.loss2 = BarycenterStatLoss(B.loss1.d, B.loss1.targets, M[i, j])

        elseif typeof(B) <: Union{FwBwGradflow}

            # B.forward.loss = BarycenterStatLoss(B.backward.back.loss1.d, B.backward.back.loss1.targets, M[i, j])
            # B.backward.back.loss2 = BarycenterTDALoss(B.loss2.d, B.loss2.dgmFun, B.loss2.targets, M[i, j])
            @set! B.forward.loss.weights = M[i, j]
            @set! B.backward.back.loss2.weights = M[i, j]

        end

        result, _ = backprop(X, B; path=false, prog=false)
        res[i, j] = deepcopy(result)
    end

    return res, M
end


function baryplot(res, M; size=(900, 900))

    n = round(Int, M |> length |> sqrt)

    resplt = [scatter(
        res[i, j] |> m2t,
        ratio=1,
        label=nothing,
        msw=0.0,
        lim=(-1.5, 1.5),
        title="λ=$(round.(Int, M[i, j]))"
    ) for i in 1:n, j in 1:n]

    display(plot(resplt..., size=size))

    return resplt
end
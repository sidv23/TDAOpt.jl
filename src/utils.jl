###################################
##### Helper Functions
begin
    m2t = x -> tuple.(eachcol(x)...)                           # Convert from Matrix -> Tuples
    m2a = x -> collect(eachrow(x))                             # Convert from Matrix -> Arrays
    t2m = T -> hcat(collect.(T)...)'                           # Convert from Tuples -> Matrix
    t2a = T -> [[a...] for a in T]                             # Convert from Tuples -> Arrays
    a2t = V -> Tuple.(V)                                       # Convert from Arrays -> Tuples
    a2m = V -> reduce(hcat, V)'                                # Convert from Arrays -> Matrix
    points_to_measure = X -> Measure(ones(size(X, 1)), X)                    # Convert from Matrix -> Unnormalized Measure
    points_to_norm_measure = X -> Measure(ones(size(X, 1)) / size(X, 1), X)       # Convert from Matrix -> Normalized Measure
end


# Convenient shorthand
begin
    p2m  = X ->  points_to_measure(X)           # Convert from Matrix -> Unnormalized Measure
    p2nm = X -> points_to_norm_measure(X)       # Convert from Matrix -> Normalized Measure
end


pers(x) = diff(x, dims=1)[1]


function plotpath(path; loss=nothing, plt=nothing, anim=true, fps=20, title="", cls=true, ylabel="Loss", filename="./plots/tmp", kwargs...)

    n = size(path, 1)

    if isnothing(plt)
        refplt = plot(0, 0)
    else
        refplt = plt
    end

    plts = repeat([refplt], n)

    if !isnothing(loss)
        error = loss.(path)
        lims = extrema(error)
    end

    if cls
        zvals = norm.(eachrow(path[1])) |> normalize
    else
        zvals = nothing
    end
        
    if anim a = Plots.Animation() end

    for i in eachindex(path)

        plts[i] = scatter(refplt, path[i] |> m2t, label=nothing, title=title * "$i", marker_z=zvals, c = cgrad(:spring, 5, categorical = true), colorbar=false; kwargs...)

        if !isnothing(loss)
            plts[i] = plot(plts[i], plot(1:i, error[1:i], ylim=lims, xlim=(1, n), label=ylabel))
        end

        if anim frame(a, plts[i]) end
    end

    if anim
        # a = @animate for i in eachindex(plts)
        #     plot(plts[i])
        # end
        gif(a, filename * ".gif", fps=fps)
    else
        return plts
    end
end


function make_gif(X, loss; opt=nothing, epochs=50, fps=20, box=1.2, plt=nothing, filename="./pts-tmp.gif")
    a = Plots.Animation()
    i = 0
    if isnothing(plt)
        p = scatter(X |> m2t, label=nothing, title="Epoch: 0", ratio=1, lim=box)
    else
        p = scatter(plt, X |> m2t, label=nothing, title="Epoch: 0", ratio=1)
    end
    frame(a, p)

    for i in 1:epochs
        if i == 1 || i % 10 == 0
            println("Epoch: $i. Loss = $(loss(X))")
        end
        g = Zygote.gradient(x -> loss(x), X)[1]
        Flux.Optimise.update!(opt, X, g)

        if isnothing(plt)
            p = scatter(X |> m2t, label=nothing, title="Epoch: $i", ratio=1, lim=box)
        else
            p = scatter(plt, X |> m2t, label=nothing, title="Epoch: $i", ratio=1)
        end
        frame(a, p)
    end
    gif(a, filename, fps=fps)
end



function make_gif2(X, Y, loss; filename="./pts-tmp.gif", opt=nothing, w=nothing, epochs=50, fps=20, box=(-1, 1))

    Ay = pairwise(Euclidean(), Y')
    indy = [rips_index(Ay, order=i) for i in [0, 1]]
    dgmy = [persistences(Ay, i) for i in indy]

    a = Plots.Animation()
    i = 0
    plt = scatter(X |> m2t, label=nothing, title="Epoch: 0", ratio=1, lim=box)
    frame(a, plt)
    for i in 1:epochs
        if i % 10 == 0
            println("Epoch: $i. Loss = $(loss(X, dgmy, w=w))")
        end
        g = Zygote.gradient(x -> loss(x, dgmy, w=w), X)[1]
        Flux.Optimise.update!(opt, X, g)
        plt = scatter(Y |> m2t, label="target", title="Epoch: $i", ratio=1, lim=box)
        plt = scatter(plt, X |> m2t, label="source", title="Epoch: $i", ratio=1, lim=box, legend=:bottomright)
        frame(a, plt)
    end
    gif(a, filename, fps=fps)
end
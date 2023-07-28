using NNlib
using Base: @kwdef

###################################################################
# Vanilla Backprop


@kwdef mutable struct Backprop <: AbstractBackprop
    loss::AbstractLossFunction
    opt::Flux.Optimise.AbstractOptimiser = NADAM(0.1)
    epochs::Integer = 100
end


function backprop(source, b::Backprop; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    optim = Flux.Optimiser(b.opt)

    if path
        trajectory = repeat([deepcopy(source)], 1 + b.epochs)
    end

    if prog
        p = Progress(b.epochs)
        generate_showvalues(x) = () -> [(typeof(b.loss), x)]
        print_every = print_every > 1 ? print_every : round(Int, b.epochs / 10)
        loss_val = val(source, b.loss)
    end


    for i in 1:b.epochs

        g = Zygote.gradient(x -> val(x, b.loss), source)[1]
        Flux.Optimise.update!(optim, source, g)

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, b.loss)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end

    return path ? (source, trajectory) : (source, nothing)
end



###################################################################
# Tilted Backprop

@kwdef mutable struct TiltedBackprop <: AbstractBackprop
    loss1::AbstractLossFunction
    loss2::AbstractLossFunction
    tilt::Union{T,Vector{T},Tuple{T,T}} where {T<:Real} = 1.0
    opt::Flux.Optimise.AbstractOptimiser = NADAM(0.1)
    epochs::Integer = 100
end


function backprop(source, b::TiltedBackprop; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    # tilted_loss(x) = NNlib.logsumexp(b.tilt .* [val(x, b.loss1), val(x, b.loss2)] .- log(2), dims=1)[1] ./ b.tilt
    # g = Zygote.gradient(tilted_loss, source)[1]

    optim = Flux.Optimiser(b.opt)

    if path
        trajectory = repeat([deepcopy(source)], 1 + b.epochs)
    end

    if prog
        p = Progress(b.epochs)
        generate_showvalues(x) = () -> [(typeof(b.loss1), x)]
        print_every = print_every > 1 ? print_every : round(Int, b.epochs / 10)
        loss_val = val(source, b.loss1)
    end

    lims = length(b.tilt) > 1 ? [b.tilt...] : [b.tilt, b.tilt]

    tilt = range(lims..., b.epochs)


    for i in 1:b.epochs
        L = softmax(tilt[i] .* [val(source, b.loss1), val(source, b.loss2)])

        g1 = Zygote.gradient(x -> val(x, b.loss1), source)[1]
        g2 = Zygote.gradient(x -> val(x, b.loss2), source)[1]

        g = (L[1] .* g1) .+ (L[2] .* g2)

        Flux.Optimise.update!(optim, source, g)

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, b.loss1)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end
    return path ? (source, trajectory) : (source, nothing)
end



###################################################################
# Alternating Backprop

@kwdef mutable struct AlternatingBackprop <: AbstractBackprop
    loss1::AbstractLossFunction
    opt1::Flux.Optimise.AbstractOptimiser = NADAM(0.1)

    loss2::AbstractLossFunction
    opt2::Flux.Optimise.AbstractOptimiser = NADAM(0.1)

    epochs::Integer = 100
end


function backprop(source, b::AlternatingBackprop; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    optim1 = Flux.Optimiser(b.opt1)
    optim2 = Flux.Optimiser(b.opt2)

    if path
        trajectory = repeat([deepcopy(source)], 1 + b.epochs)
    end

    if prog
        p = Progress(b.epochs)
        generate_showvalues(x) = () -> [(typeof(b.loss1), x)]
        print_every = print_every > 1 ? print_every : round(Int, b.epochs / 10)
        loss_val = val(source, b.loss1)
    end


    for i in 1:b.epochs

        g1 = Zygote.gradient(x -> val(x, b.loss1), source)[1]
        Flux.Optimise.update!(optim1, source, g1)

        g2 = Zygote.gradient(x -> val(x, b.loss2), source)[1]
        Flux.Optimise.update!(optim2, source, g2)

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, b.loss1)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end
    return path ? (source, trajectory) : (source, nothing)
end




###################################################################
# Backprop with a schedule


@kwdef mutable struct ScheduledBackprop <: AbstractBackprop

    loss1::AbstractLossFunction
    opt1::Flux.Optimise.AbstractOptimiser = NADAM(0.1)

    loss2::AbstractLossFunction
    opt2::Flux.Optimise.AbstractOptimiser = NADAM(0.1)

    schedule::Function = t -> 0.5 * (1 + sin(t / 10))
    epochs::Integer = 100
end


function backprop(source, b::ScheduledBackprop; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    optim1 = Flux.Optimiser(b.opt1)
    optim2 = Flux.Optimiser(b.opt2)

    if path
        trajectory = repeat([deepcopy(source)], 1 + b.epochs)
    end

    if prog
        p = Progress(b.epochs)
        generate_showvalues(x) = () -> [(typeof(b.loss1), x)]
        print_every = print_every > 1 ? print_every : round(Int, b.epochs / 10)
        loss_val = val(source, b.loss1)
    end


    for i in 1:b.epochs

        g1 = Zygote.gradient(x -> val(x, b.loss1), source)[1]
        g2 = Zygote.gradient(x -> val(x, b.loss2), source)[1]

        Flux.Optimise.update!(optim1, source, (b.schedule(i)) .* g1)
        Flux.Optimise.update!(optim2, source, (1 - b.schedule(i)) .* g2)

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, b.loss1)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end

    return path ? (source, trajectory) : (source, nothing)
end




###################################################################
# Hierarchies


const CompositeBackprop = Union{TiltedBackprop,AlternatingBackprop,ScheduledBackprop}
const VanillaBackprop = Union{Backprop}
const EuclideanBackprop = Union{VanillaBackprop,CompositeBackprop}
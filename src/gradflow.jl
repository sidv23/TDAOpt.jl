using Base: @kwdef

###################################################################
# Forward Euler Method

@kwdef mutable struct FwGradflow <: AbstractGradflow
    loss::AbstractLossFunction
    opt::Flux.Optimise.AbstractOptimiser = NADAM(0.1)
    epochs::Integer = 100
    noise::Function = t -> 0
end


function backprop(source, Fw::FwGradflow; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    optim = Flux.Optimiser(Fw.opt)

    if path
        trajectory = repeat([deepcopy(source)], 1 + Fw.epochs)
    end

    if prog
        p = Progress(Fw.epochs)
        generate_showvalues(x) = () -> [(typeof(Fw.loss), x)]
        print_every = print_every > 1 ? print_every : round(Int, Fw.epochs / 10)
        loss_val = val(source, Fw.loss)
    end


    for i in 1:Fw.epochs
        g = ∇w(source .+ (Fw.noise(i) .* randn(size(source))), Fw.loss)
        Flux.Optimise.update!(optim, source, g)

        if i % print_every == 0
            loss_val = val(source, Fw.loss)
        end

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end

    return path ? (source, trajectory) : (source, nothing)
end



###################################################################
# Backward Euler Step

@kwdef struct BwGradflow <: AbstractGradflow
    back::CompositeBackprop
    epochs::Integer = 100
    step::Function = t -> 1e-1
    noise::Function = t -> 0
end


function backward_euler_step(source, method::CompositeBackprop; t=1.0, s=0.0)
    @set! method.loss1.scale = 1 / (2 * t)
    @set! method.loss1.target = source .+ (s .* randn(size(source)))
    source, _ = backprop(source, method; prog=false, path=false)
    return source
end


function backprop(source, Bw::BwGradflow; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    if path
        trajectory = repeat([deepcopy(source)], 1 + Bw.epochs)
    end

    if prog
        p = Progress(Bw.epochs)
        generate_showvalues(x) = () -> [(typeof(Bw.back.loss2), x)]
        print_every = print_every > 1 ? print_every : round(Int, Bw.epochs / 10)
        loss_val = val(source, Bw.back.loss2)
    end

    for i in 1:Bw.epochs
        source = backward_euler_step(source, Bw.back; t=Bw.step(i), s=Bw.noise(i))

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, Bw.back.loss2)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end

    return path ? (source, trajectory) : (source, nothing)
end



###################################################################
# Forward/Backward Gradient Flow

@kwdef struct FwBwGradflow <: AbstractGradflow
    forward::FwGradflow
    backward::BwGradflow
end


function backprop(source, FwBw::FwBwGradflow; print_every::Integer=1, prog::Bool=true, path::Bool=true)

    optim = Flux.Optimiser(FwBw.forward.opt)

    if path
        trajectory = repeat([deepcopy(source)], 1 + FwBw.forward.epochs)
    end

    if prog
        p = Progress(FwBw.forward.epochs)
        generate_showvalues(x) = () -> [(typeof(FwBw.forward.loss), x)]
        print_every = print_every > 1 ? print_every : round(Int, FwBw.forward.epochs / 10)
        loss_val = val(source, FwBw.forward.loss)
    end


    for i in 1:FwBw.forward.epochs

        g = ∇w(source .+ (FwBw.forward.noise(i) .* randn(size(source))), FwBw.forward.loss)
        Flux.Optimise.update!(optim, source, g)

        source = backward_euler_step(source, FwBw.backward.back; t=FwBw.backward.step(i), s=FwBw.backward.noise(i))

        if prog
            next!(p, showvalues=generate_showvalues(loss_val))

            if i % print_every == 0
                loss_val = val(source, FwBw.forward.loss)
            end
        end

        if path
            trajectory[i+1] = deepcopy(source)
        end
    end

    return path ? (source, trajectory) : (source, nothing)
end

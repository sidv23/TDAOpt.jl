using TDAOpt
using Plots
using LinearAlgebra
using Zygote 
using ProgressMeter
using Optimisers
using Flux
using Pipe
using LaTeXStrings

begin
    dgm = x -> TDAOpt.dgm_cubical(x; order=1)

    function im_animate(paths; name="")
        prefix = length(name) == 0 ? fill("Epoch: ", 3) : name .* ". Epoch: "

        @gif for i in eachindex(paths[1])
            plot(
                plot(
                    Gray.(paths[1][i]),
                    label="", title=prefix[1] * "$i",
                    grid=false, axis=false,
                ),
                plot(
                    Gray.(paths[2][i]),
                    label="", title=prefix[2] * "$i",
                    grid=false, axis=false,
                ),
                plot(
                    Gray.(paths[3][i]),
                    label="", title=prefix[3] * "$i",
                    grid=false, axis=false,
                ),
                layout=(1, 3), size=(600, 200),
            )
        end
    end

    function img_foo(b, ℓ, N, opt; α::Real=0.9)
        bs = fill(copy(b), N + 1)
        losses = fill(ℓ(b), N + 1)
        optim = Flux.setup(opt, b)

        @showprogress for i in 1:N
            if (i > 1) && (i % 10 == 0)
                Optimisers.adjust!(optim, α * optim.rule.eta)
            end
            g = Zygote.gradient(ℓ, b)[1]
            Flux.Optimise.update!(optim, b, g)
            bs[i+1] = copy(b)
            losses[i+1] = ℓ(b)
        end
        return bs
    end

    function persistence_entropy(Dx; order::Int=0)
        dx = Dx[1+order]
        pers_x = pers.(eachrow(dx))
        npers_x = pers_x ./ sum(pers_x)
        return -sum(npers_x .* log.(npers_x))
    end

    dgmnorm(Dx; order, p) = @pipe Dx[1+order] |>
                                    eachrow .|>
                                    pers |>
                                    norm(_, p)
end

# Generate Data
begin
    n_samples, (n, m) = 200, fill(20, 2)
    Xs = randn(n_samples, n, m)
    β = [(n/3.5)^2 <= (n/2 - i)^2 + (n/2 - j)^2 < (n/2.5)^2 ? 1.0 : 0.0 for i in 1:m, j in 1:n]
    y = [3.0 * randn() + sum(β .* x) for x in eachslice(Xs, dims=1)]
    Gray.(β)
end




f(b, x_, y_) = sum(abs2, y_ .- [sum(b .* x) for x in eachslice(x_, dims=1)])
ℓ(b; λ) = f(b, Xs, y) + λ * norm(b, 1)

function ℓ_tda(x)
    dx = dgm(x)
    loss0 = dgmnorm(dx; order=0, p=1)
    loss1 = dgmnorm(dx; order=1, p=1) - 2 * dgmnorm(dx; order=1, p=Inf)
    return loss0 + loss1
end





begin
    b = randn(size(Xs)[2:end])
    N, opt, a = 400, NADAM(0.5), 0.9
    reg_path = img_foo(copy(b), x -> ℓ(x; λ=0.0), N, opt; α=a);
    lasso_path = img_foo(copy(b), x -> ℓ(x; λ=5.0), N, opt; α=a);
    tda_path = img_foo(copy(b), x -> ℓ(x; λ=5.0) +  10.0 * ℓ_tda(x), N, opt; α=a);

    reg_loss = [sum(abs2, x .- β) for x in reg_path]
    lasso_loss = [sum(abs2, x .- β) for x in lasso_path]
    tda_loss = [sum(abs2, x .- β) for x in tda_path]

    plot(0, ylabel=L"\||\hat{\beta} - \beta^*\||", xlabel="# Iterations")
    plot!(reg_loss, label="Reg", yscale=:log10, lw=2)
    plot!(lasso_loss, label="Reg + Lasso", yscale=:log10, lw=2)
    plot!(tda_loss, label="Reg + Lasso + TDA", yscale=:log10, lw=2, legend=:right)
end
im_animate([reg_path, lasso_path, tda_path], name=["Reg", "Lasso", "TDA"])





# Signal lies on a circular region
function condition(i, j, n)
    dist_from_center = (i - 0.5 * n)^2 + (j - 0.5 * n)^2
    return (n / 3.5)^2 <= dist_from_center < (n / 2.5)^2
end

n_samples, (n, m) = 200, fill(20, 2)

# X: Data
Xs = randn(n_samples, n, m)

# β: The ground truth regression parameter
β = [condition(i, j, n) for i in 1:m, j in 1:n]

# y: Response Variable
y = [sum(β .* x) + 3.0 * randn() for x in eachslice(Xs, dims=1)]

plot(Gray.(β), axis=false, grid=false, size=(400, 400))
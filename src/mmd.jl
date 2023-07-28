######################################################################
# MMD: Implementation

function mean_triu(A::Matrix)
    n = size(A, 1)
    return (sum(A) - sum(diag(A))) / (n * (n - 1))
end

function sum_triu(A::Matrix)
    n = size(A, 1)
    return (sum(A) - sum(diag(A))) * n / (n - 1)
end

function mmd(x::Matrix{T}, y::Matrix{T}; kernel::Kernel, unbiased::Bool=false)::T where {T<:Real}

    Kxx = kernelmatrix(kernel, x, obsdim=1)
    Kyy = kernelmatrix(kernel, y, obsdim=1)
    Kxy = kernelmatrix(kernel, x, y, obsdim=1)

    if unbiased
        return sqrt(mean_triu(Kxx) + mean_triu(Kyy) - 2 * mean(Kxy))
    else
        return sqrt(mean(Kxx) + mean(Kyy) - 2 * mean(Kxy))
    end

end


function mmd(μ::Measure, ν::Measure; kernel::Kernel, unbiased::Bool=false)
    Kxx = kernelmatrix(kernel, μ.support, obsdim=1) .* (μ.density * μ.density')
    Kyy = kernelmatrix(kernel, ν.support, obsdim=1) .* (ν.density * ν.density')
    Kxy = kernelmatrix(kernel, μ.support, ν.support, obsdim=1) .* (μ.density * ν.density')

    if unbiased
        return sqrt(sum_triu(Kxx) + sum_triu(Kyy) - 2 * sum(Kxy))
    else
        return sqrt(sum(Kxx) + sum(Kyy) - 2 * sum(Kxy))
    end
end


######################################################################
# Wasserstein Gradient of MMD

# When inputs are matrices
function ∇₂K(x::Matrix{T}, y::Matrix{T}; kernel::Kernel) where {T<:Real}
    return gradient(z -> sum(mean(kernelmatrix(kernel, x', z'), dims=1)), y)[1]
end


function ∇W_mmd(x::Matrix{T}, y::Matrix{T}; kernel::Kernel) where {T<:Real}
    return ∇₂K(x, x; kernel=kernel) .- ∇₂K(y, x; kernel=kernel)
end


# When inputs are Measures
function ∇₂K(μ::M, ν::M; kernel::Kernel) where {M<:Measure}
    return gradient(z -> sum(mean(kernelmatrix(kernel, μ.support', z'), dims=1)), ν.support)[1]
end


function ∇W_mmd(μ::M, ν::M; kernel::Kernel) where {M<:Measure}
    return ∇₂K(μ, μ; kernel=kernel) .- ∇₂K(ν, μ; kernel=kernel)
end


######################################################################
# MMD

@kwdef struct MMD <: StatDiscrepancy
    k::Kernel
    unbiased::Bool = false
end

######################################################################
# Dist definition for MMD

function dist(d::MMD, x::Matrix{T}, y::Matrix{T}) where {T<:Real}
    return mmd(x, y; kernel=d.k, unbiased=d.unbiased)
end


function ∇w(d::MMD, x::Matrix{T}, y::Matrix{T}) where {T<:Real}
    return ∇W_mmd(x, y; kernel=d.k)
end
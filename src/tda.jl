###################################
##### TDA Functions: Point Cloud
using Ripserer: birth_simplex, death_simplex, vertices, Wasserstein, Bottleneck

###################################
##### Distance Matrix Functions

function wDist_0(X; kwargs...)
    D = pairwise(Euclidean(), X, dims=1)
    return D
end



function wDist_1(X; m=0.1)
    n = size(X, 1)
    D = pairwise(Euclidean(), X, dims=1)
    
    # Compute Distance to measure
    indx = Zygote.@ignore hcat(map(x -> x .< quantile(x, m), eachrow(D))...)'
    w = sqrt.(sum(D .* indx, dims=2) ./ (n * m))
    # w = sum(D .* indx, dims=2) ./ (n * m)
    
    flag = Zygote.@ignore D .≤ abs.(w .- w')
    result = (flag .* max.(w, w')) .+ ((1 .- flag) .* 0.5 .* (w .+ w' .+ D))
    return result
end




function wDist_2(X; m=0.1)
    n = size(X, 1)
    D = pairwise(SqEuclidean(), X, dims=1)
    
    # Compute Distance to measure
    indx = Zygote.@ignore hcat(map(x -> x .< quantile(x, m), eachrow(D))...)'
    w = sqrt.(sum(D .* indx, dims=2) ./ (n * m))
    
    flag = Zygote.@ignore D .≤ abs.(w .^ 2 .- w' .^ 2)
    result = (flag .* max.(w, w')) .+ ((1 .- flag) .* abs.(((w .+ w') .^ 2 .+ D) .* ((w .- w') .^ 2 .+ D)))
    return result
end




function wDist_inf(X; m=0.1)
    n = size(X, 1)
    D = pairwise(Euclidean(), X, dims=1)
    
    # Compute Distance to measure
    indx = Zygote.@ignore hcat(map(x -> x .< quantile(x, m), eachrow(D))...)'
    w = sqrt.(sum(D .* indx, dims=2) ./ (n * m))
    
    return max.(w, w', D .* 0.5)
end


###################################
##### Persistence Diagram Functions


# Indices for edges supporting the birth/death simplices for a Dgm
function get_indices(A, D)
    v = x -> isnothing(x) ? collect(1:size(A, 1)) : Ripserer.vertices(x)
    bs = collect.(v.(Ripserer.birth_simplex.(D)))
    ds = collect.(v.(Ripserer.death_simplex.(D)))
    indices = Any[]
    for i in 1:length(D)
        push!(indices, bs[i][[Tuple(argmax(A[[bs[i]...], [bs[i]...]]))...]])
        push!(indices, ds[i][[Tuple(argmax(A[[ds[i]...], [ds[i]...]]))...]])
    end
    return indices
end


# Get the indices for the Rips complex
function rips_indices(A; order=1)
    D = Zygote.@ignore ripserer(A, reps=true, alg=:involuted, dim_max=order)
    # D[1] = D[1][1:end-1]
    indices = [get_indices(A, D[i]) for i in 1:order+1]
    return indices
end


# Retrun the birth/death values using the distance matrix
function persistences(A, indices)
    # return map(x -> A[x...], reshape(indices, 2, :)')
    return Matrix(reshape([A[x...] for x in indices], 2, :)')
end




###################################
##### Persistence Diagrams


function dgm_0(x; order::I=1, kwargs...) where {I<:Int}
    Ax = pairwise(Euclidean(), x, dims=1)
    indx = Zygote.@ignore rips_indices(Ax; order=order)
    dgmx = [persistences(Ax, i) for i in indx]
    return dgmx
    # return map(x -> Transpose(reshape(x, 2, :)), [[Ax[i...] for i in ind] for ind in indx])
end


function dgm_1(x; order::I=1, kwargs...) where {I<:Int}
    Ax = wDist_1(x; kwargs...)
    indx = Zygote.@ignore rips_indices(Ax; order=order)
    dgmx = [persistences(Ax, i) for i in indx]
    return dgmx
end


function dgm_inf(x; order::I=1, kwargs...) where {I<:Int}
    Ax = wDist_inf(x; kwargs...)
    indx = Zygote.@ignore rips_indices(Ax; order=order)
    dgmx = [persistences(Ax, i) for i in indx]
    return dgmx
end


# Get persistence diagram from point cloud
# function makeDgm(x; order::I, weighted::Bool=false, p=1)
#     Ax = pairwise(Euclidean(), x')
#     indx = Zygote.@ignore rips_indices(Ax; order=1)
#     dgmx = [persistences(Ax, i) for i in indx]
#     return dgmx, indx, Ax
# end


# function makeDgm(x; order::I, weighted::Bool=false, p::T=Inf, kwargs...) where {I<:Int,T<:Real}
#     if weighted
#         if p == 1
#             Ax = wDist_1(x; kwargs...)
#         else
#             Ax = wDist_inf(x; kwargs...)
#         end
#     else
#         Ax = pairwise(Euclidean(), x, dims=1)
#     end
#     indx = Zygote.@ignore rips_indices(Ax; order=order)
#     dgmx = [persistences(Ax, i) for i in indx]
#     return dgmx
# end
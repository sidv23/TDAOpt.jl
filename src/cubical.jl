###################################
##### TDA Functions: Cubical

function dgm_cubical(x, order::I=1; kwargs...) where {I<:Int}
    indx = Zygote.@ignore cubical_indices(x; order=order)
    dgmx = [[x[k...] for k in dgm_indx] for dgm_indx in indx]
    return dgmx
end

function cubical_indices(x; order=1, sublevel=true)
    if !sublevel
        x .= -x
    end
    Cube = Cubical(x)
    C = Cube.cubemap
    Dgm = ripserer(Cube; reps=true, alg=:involuted, dim_max=order)
    indices = [supporting_indices(C, Dgm[i]) for i in 1:order+1]
    return indices
end

function supporting_indices(C, D)
    return [map(x -> birth_vertices(C, x), D) map(x -> death_vertices(C, x), D)]
end

function birth_vertices(C, interval)
    b = birth_simplex(interval).root
    return supporting_index(C, b...)
end

function death_vertices(C, interval)
    if isinf(death(interval))
        d = findall(C .== maximum(C))[1] |> Tuple
    else
        d = death_simplex(interval).root
    end
    return supporting_index(C, d...)
end

function supporting_index(C, i, j)
    search = [CartesianIndex(i, j)]
    if i % 2 == 0 && j % 2 != 0
        search = [
            CartesianIndex(i - 1, j),
            CartesianIndex(i + 1, j)
        ]
    elseif i % 2 != 0 && j % 2 == 0
        search = [
            CartesianIndex(i, j - 1),
            CartesianIndex(i, j + 1)
        ]
    elseif i % 2 == 0 && j % 2 == 0
        search = [
            CartesianIndex(i - 1, j - 1),
            CartesianIndex(i - 1, j + 1),
            CartesianIndex(i + 1, j - 1),
            CartesianIndex(i + 1, j + 1),
        ]
    end
    vertex_map = Int.((Tuple(search[C[search].==C[i, j]][1]) .+ 1) ./ 2)
    return vertex_map
end
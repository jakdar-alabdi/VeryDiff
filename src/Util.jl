import Base: size

function size(Z::Zonotope, d::Integer)
    @assert d<=2
    return size(Z.G,d)
end

function truncate_network(T::Type,N::Network)
    layers = []
    for l in N.layers
        if l isa ReLU
            push!(layers, l)
        elseif l isa Dense
            push!(layers,Dense(convert.(Float64,convert.(T,deepcopy(l.W))),convert.(Float64,convert.(T,deepcopy(l.b)))))
        else
            throw("Unknown layer type")
        end
    end
    return Network(layers)
end

function to_diff_zono(task :: VerificationTask)
    input_dim = size(task.middle,1)
    if NEW_HEURISTIC
        Z1 = Zonotope(Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance', task.middle, Matrix(1.0I, size(task.distance_indices,1), size(task.distance_indices,1)))
    else
        Z1 = Zonotope(Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance', task.middle, nothing)
    end
    if task.∂num_approx > 0
        z1_bounds = zono_bounds(Z1)
        ∂z_bounds = zono_bounds(task.∂Z)
        lower_bounds = max.(z1_bounds[:,1] .+ ∂z_bounds[:,1], task.lower_bounds)
        upper_bounds = min.(z1_bounds[:,2] .+ ∂z_bounds[:,2], task.upper_bounds)
        mid = (lower_bounds .+ upper_bounds) ./ 2
        distance = (upper_bounds .- lower_bounds) ./ 2
        if NEW_HEURISTIC
            influence2 = zeros(Float64, input_dim, input_dim)
        else
            influence2 = nothing
        end
        Z2 = Zonotope(Matrix(I, input_dim, input_dim) .* distance', mid, influence2)
        num_approx2 = input_dim - length(task.distance_indices)
    else
        Z2 = deepcopy(Z1)
        num_approx2 = 0
    end
    return DiffZonotope(Z1, Z2, task.∂Z, 0, num_approx2, task.∂num_approx)
end
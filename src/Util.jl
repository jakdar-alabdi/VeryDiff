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
    if !isnothing(task.distance1_secondary)
        generator1 = hcat(
            Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance',
            Matrix(I, input_dim, input_dim) .* task.distance1_secondary',
            Matrix(0.0*I, input_dim, input_dim)
        )
        middle1 = task.middle .+ task.middle1_secondary
    else
        generator1 = Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance'
        middle1 = task.middle
    end
    if !isnothing(task.distance2_secondary)
        generator2 = hcat(
            Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance',
            Matrix(0.0*I, input_dim, input_dim),
            Matrix(I, input_dim, input_dim) .* task.distance2_secondary'
        )
        middle2 = task.middle .+ task.middle2_secondary
    else
        generator2 = Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance'
        middle2 = task.middle
    end

    generator_diff = zeros(Float64, input_dim, size(generator1,2))
    middle_diff = zeros(Float64, input_dim)
    if !isnothing(task.distance1_secondary)
        generator_diff[:, (size(task.distance,1)+1):(size(task.distance,1)+input_dim)] .= generator1[:, (size(task.distance,1)+1):(size(task.distance,1)+input_dim)]
        middle_diff .+= task.middle1_secondary
    end
    if !isnothing(task.distance2_secondary)
        generator_diff[:, (size(task.distance,1)+input_dim+1):end] .-= generator2[:, (size(task.distance,1)+input_dim+1):end]
        middle_diff .-= task.middle2_secondary
    end

    if NEW_HEURISTIC
        # TODO (steuber): If we keep discarding the the secondary dimensions, we can optimize this with a smaller matrix
        influence1 = Matrix(1.0I, size(generator1,2), size(generator1,2))
        influence2 = Matrix(1.0I, size(generator2,2), size(generator2,2))
        influence_diff = nothing
    else
        influence1 = nothing
        influence2 = nothing
        influence_diff = nothing
    end
    Z1 = Zonotope(generator1, middle1, influence1)
    Z2 = Zonotope(generator2, middle2, influence2)
    ∂Z = Zonotope(generator_diff, middle_diff, influence_diff)
    return DiffZonotope(Z1, Z2, ∂Z, 0, 0, 0)
end
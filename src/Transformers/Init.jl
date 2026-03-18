function init_default_zono(Z :: CachedZonotope)
    Z.zonotope = DiffZonotope(
        Zonotope(
            Vector{Matrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.Z₁.Gs]),
            Z.zonotope_proto.Z₁.c,
            (isnothing(Z.zonotope_proto.Z₁.influence) ? nothing : Vector{Matrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.Z₁.influence])),
            Z.zonotope_proto.Z₁.generator_ids,
            Z.zonotope_proto.Z₁.owned_generators
        ),
        Zonotope(
            Vector{Matrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.Z₂.Gs]),
            Z.zonotope_proto.Z₂.c,
            (isnothing(Z.zonotope_proto.Z₂.influence) ? nothing : Vector{Matrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.Z₂.influence])),
            Z.zonotope_proto.Z₂.generator_ids,
            Z.zonotope_proto.Z₂.owned_generators
        ),
        Zonotope(
            Vector{Matrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.∂Z.Gs]),
            Z.zonotope_proto.∂Z.c,
            (isnothing(Z.zonotope_proto.∂Z.influence) ? nothing : Vector{Matrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.∂Z.influence])),
            Z.zonotope_proto.∂Z.generator_ids,
            Z.zonotope_proto.∂Z.owned_generators
        )
    )
end

function init_zono_from_dims(output_dim :: Int, generator_ids :: SortedVector{Int64}, generator_dims :: Dict{Int64, Int64}, owned_generator_id :: Union{Int64, Nothing}; influence_matrix=false, influence_dim=0)
    if !isnothing(owned_generator_id)
        owned_generators = find_index_position(generator_ids, owned_generator_id)
    else
        owned_generators = nothing
    end
    generators = Vector{Matrix{Float64}}()
    for gid in generator_ids
        new_g = zeros(Float64, output_dim, generator_dims[gid])
        push!(generators, new_g)
    end
    c = zeros(Float64, output_dim)
    if !influence_matrix
        influence = nothing
    else
        influence = Matrix{Float64}[]
        for g in generators
            push!(influence, zeros(Float64, influence_dim, size(g,2)))
        end
    end
    return Zonotope(generators, c, influence, generator_ids, owned_generators)
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXAdd{S1}, ONNXAdd{S2}, ONNXAdd{S3}}, input_zonos :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
    @assert length(input_zonos) >= 2 "Add DiffLayer should have at least two inputs"
    @assert length(output_positions) == 1 "Add DiffLayer should have exactly one output"

    # Assert output shapes match
    for i in 2:length(input_zonos)
        @assert size(input_zonos[i].zonotope.Z₁.c) == size(input_zonos[1].zonotope.Z₁.c) "Input dimension mismatch for Add layer"
        @assert size(input_zonos[i].zonotope.Z₂.c) == size(input_zonos[1].zonotope.Z₂.c) "Input dimension mismatch for Add layer"
        @assert size(input_zonos[i].zonotope.∂Z.c) == size(input_zonos[1].zonotope.∂Z.c) "Input dimension mismatch for Add layer"
    end

    # Get generator ids for all inputs and create union of them
    generator_ids1 = SortedVector{Int64}()
    generator_ids2 = SortedVector{Int64}()
    generator_idsd = SortedVector{Int64}()
    # Get generator dimensions for all mentioned generator ids
    generator_dims = Dict{Int64, Int64}()
    for input_zono in input_zonos
        generator_ids1 = union(generator_ids1, input_zono.zonotope.Z₁.generator_ids)
        generator_ids2 = union(generator_ids2, input_zono.zonotope.Z₂.generator_ids)
        generator_idsd = union(generator_idsd, input_zono.zonotope.∂Z.generator_ids)
        for (i,gid) in enumerate(input_zono.zonotope.Z₁.generator_ids)
            if haskey(generator_dims, gid)
                @assert generator_dims[gid] == size(input_zono.zonotope.Z₁.Gs[i], 2) "Generator dimension mismatch for generator id $gid"
            else
                generator_dims[gid] = size(input_zono.zonotope.Z₁.Gs[i], 2)
            end
        end
        for (i,gid) in enumerate(input_zono.zonotope.Z₂.generator_ids)
            if haskey(generator_dims, gid)
                @assert generator_dims[gid] == size(input_zono.zonotope.Z₂.Gs[i], 2) "Generator dimension mismatch for generator id $gid"
            else
                generator_dims[gid] = size(input_zono.zonotope.Z₂.Gs[i], 2)
            end
        end
        for (i,gid) in enumerate(input_zono.zonotope.∂Z.generator_ids)
            if haskey(generator_dims, gid)
                @assert generator_dims[gid] == size(input_zono.zonotope.∂Z.Gs[i], 2) "Generator dimension mismatch for generator id $gid"
            else
                generator_dims[gid] = size(input_zono.zonotope.∂Z.Gs[i], 2)
            end
        end
    end
    owned_generator_id_1 = nothing
    owned_generator_id_2 = nothing
    owned_generator_id_d = nothing
    # for input in input_zonos
    #     if diff_layer.layer_idx == input.first_usage && !isnothing(input.zonotope.∂Z.owned_generators)
    #         owned_generator_id_d = input.zonotope.∂Z.generator_ids[input.zonotope.∂Z.owned_generators]
    #     end
    #     if diff_layer.layer_idx == input.first_usage && !isnothing(input.zonotope.Z₁.owned_generators)
    #         owned_generator_id_1 = input.zonotope.Z₁.generator_ids[input.zonotope.Z₁.owned_generators]
    #     end
    #     if diff_layer.layer_idx == input.first_usage && !isnothing(input.zonotope.Z₂.owned_generators)
    #         owned_generator_id_2 = input.zonotope.Z₂.generator_ids[input.zonotope.Z₂.owned_generators]
    #     end
    # end
    compute_influence = any([!isnothing(z.zonotope.Z₁.influence) for z in input_zonos])
    influence_dim = compute_influence ? 0 : size(input_zonos[1].zonotope.Z₁.influence[1], 1)
    Z₁ = init_zono_from_dims(size(input_zonos[1].zonotope.Z₁.c,1), generator_ids1, generator_dims, owned_generator_id_1; influence_matrix=compute_influence, influence_dim=influence_dim)
    Z₂ = init_zono_from_dims(size(input_zonos[1].zonotope.Z₂.c,1), generator_ids2, generator_dims, owned_generator_id_2; influence_matrix=compute_influence, influence_dim=influence_dim)
    ∂Z = init_zono_from_dims(size(input_zonos[1].zonotope.∂Z.c,1), generator_idsd, generator_dims, owned_generator_id_d; influence_matrix=false)
    Z = CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXRelu{S1}, ONNXRelu{S2}, ONNXRelu{S3}}, inputs :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
    @assert length(inputs) == 1 "ReLU DiffLayer should have exactly one input"
    @assert length(output_positions) == 1 "ReLU DiffLayer should have exactly one output"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    # Compute Bounds
    bounds₁ = zono_bounds(input_zono.Z₁)
    bounds₂ = zono_bounds(input_zono.Z₂)
    ∂bounds = zono_bounds(input_zono.∂Z)
    (
        _, _, _, _, _,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    ) = get_selectors(bounds₁, bounds₂, ∂bounds)
    # Do NOT use counts created above for new_gen₁ / new_gen₂,
    # because these omit dimensions where difference is still zero
    new_gen₁ = count(bounds₁[:,1] .< 0.0 .&& bounds₁[:,2] .> 0.0)
    new_gen₂ = count(bounds₂[:,1] .< 0.0 .&& bounds₂[:,2] .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
    Z₁ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₁, new_gen₁, diff_layer.layer_idx)
    Z₂ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₂, new_gen₂, diff_layer.layer_idx)
    generators_d = Matrix{Float64}[]
    # Three way merge of generators: All generators from ∂Z + all from new Z₁ + all from new Z₂
    # First build union of generator ids
    generator_ids = union(input_zono.∂Z.generator_ids, union(Z₁.generator_ids, Z₂.generator_ids))
    owned_generator_id = nothing
    #if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
    #    owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    #end
    # Now iterate over generator ids and figure out where the generators come from
    # Prefer Z₁ and Z₂ over ∂Z when there are overlaps because those might have new generators
    for gid in generator_ids
        if gid in Z₂.generator_ids
            idx = find_index_position(Z₂.generator_ids, gid)
            new_g = zeros(size(Z₂.Gs[idx],1), size(Z₂.Gs[idx],2))
            push!(generators_d, new_g)
        elseif gid in Z₁.generator_ids
            idx = find_index_position(Z₁.generator_ids, gid)
            new_g = zeros(size(Z₁.Gs[idx],1), size(Z₁.Gs[idx],2))
            push!(generators_d, new_g)
        else
            idx = find_index_position(input_zono.∂Z.generator_ids, gid)
            columns = size(input_zono.∂Z.Gs[idx],2)
            if gid == owned_generator_id
                columns += ∂new_gen
            end
            new_g = zeros(size(input_zono.∂Z.Gs[idx],1), columns)
            push!(generators_d, new_g)
        end
    end
    if isnothing(owned_generator_id)
        owned_generator_id = get_free_generator_id!(PS)
        #@info "No owned generator for ReLU layer at index $(diff_layer.layer_idx). Adding new block of $∂new_gen generators for ∂Z at the end of the generator list with ID $owned_generator_id."
        new_g = zeros(Float64, size(input_zono.∂Z.c,1), ∂new_gen)
        push!(generators_d, new_g)
        push!(generator_ids, owned_generator_id)
    end
    c = zeros(Float64, size(Z₂.c,1))
    ∂Z = Zonotope(generators_d, c, input_zono.∂Z.influence, generator_ids, find_index_position(generator_ids, owned_generator_id))
    @assert isnothing(input_zono.∂Z.influence) "ReLU DiffLayer does not support influenced zonotopes (yet?)"
    Z = CachedZonotope(
            DiffZonotope(
                Z₁,
                Z₂,
                ∂Z
            ),
            nothing
        )
    init_default_zono(Z)
    @assert !isnothing(Z.zonotope.Z₁.owned_generators) "Z₁ should have an owned generator assigned at this point"
    @assert !isnothing(Z.zonotope.Z₂.owned_generators) "Z₂ should have an owned generator assigned at this point"
    @assert !isnothing(Z.zonotope.∂Z.owned_generators) "∂Z should have an owned generator assigned at this point"
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXLinear{S1}, ZeroDense{S2}, ONNXLinear{S3}}, inputs :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    @assert length(output_positions) == 1 "Dense DiffLayer should have exactly one output"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    L2 = get_layer2(diff_layer)
    L1_W = L1.dense.weight
    L2_W = L2.dense.weight
    @assert size(L1_W,2) == size(input_zono.Z₁.Gs[1],1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2_W,2) == size(input_zono.Z₂.Gs[1],1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(size(L1_W,1), input_zono, input_zono_cache, diff_layer.layer_idx)
    ∂Z = init_zonotope(size(L1_W,1), input_zono.∂Z, input_zono.∂Z.influence, input_zono.∂Z.owned_generators)
    Z =CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXAddConst{S1},ONNXAddConst{S2},ONNXAddConst{S3}}, inputs :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    @assert length(output_positions) == 1 "Dense DiffLayer should have exactly one output"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    Z₁, Z₂ = init_layer_dense_z1_z2(length(input_zono.Z₁.c), input_zono, input_zono_cache, diff_layer.layer_idx)
    ∂Z = init_zonotope(length(input_zono.Z₁.c), input_zono.∂Z, input_zono.∂Z.influence, input_zono.∂Z.owned_generators)
    Z =CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXLinear{S1},ONNXLinear{S2},ONNXLinear{S3}}, inputs :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    @assert length(output_positions) == 1 "Dense DiffLayer should have exactly one output"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    diff_layer_output =get_diff_layer(diff_layer)
    L2 = get_layer2(diff_layer)
    L1_W = L1.dense.weight
    L2_W = L2.dense.weight
    diff_layer_output_W = diff_layer_output.dense.weight
    @assert size(L1_W,2) == size(input_zono.Z₁.Gs[1],1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2_W,2) == size(input_zono.Z₂.Gs[1],1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(size(L1_W,1), input_zono, input_zono_cache, diff_layer.layer_idx)
    # Instantiate ∂Z
    influence = input_zono.∂Z.influence
    #if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
    #    owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    #else
        owned_generator_id = nothing
    #end
    generators = Vector{Matrix{Float64}}()
    generator_ids = SortedVector{Int64}()
    i_d = 1
    i_2 = 1
    while i_d <= length(input_zono.∂Z.generator_ids) && i_2 <= length(input_zono.Z₂.generator_ids)
        if input_zono.∂Z.generator_ids[i_d] == input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output_W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
            i_2 += 1
        elseif input_zono.∂Z.generator_ids[i_d] < input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output_W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
        else
            # input_zono.∂Z.generator_ids[i_d] > input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output_W,1), size(input_zono.Z₂.Gs[i_2],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
            i_2 += 1
        end
    end
    while i_d <= length(input_zono.∂Z.generator_ids)
        new_g = zeros(Float64, size(diff_layer_output_W,1), size(input_zono.∂Z.Gs[i_d],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
        i_d += 1
    end
    while i_2 <= length(input_zono.Z₂.generator_ids)
        new_g = zeros(Float64, size(diff_layer_output_W,1), size(input_zono.Z₂.Gs[i_2],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
        i_2 += 1
    end
    ∂Z = Zonotope(generators,
    zeros(Float64, size(diff_layer_output_W,1)), influence, generator_ids, owned_generator_id === nothing ? nothing : find_index_position(generator_ids, owned_generator_id))
    Z = CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end
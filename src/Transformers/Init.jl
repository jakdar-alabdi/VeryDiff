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
    new_gen₁ = count(bounds₁[:, 1] .< 0.0 .&& bounds₁[:, 2] .> 0.0)
    new_gen₂ = count(bounds₂[:, 1] .< 0.0 .&& bounds₂[:, 2] .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
    
    all_generator_ids = union(input_zono.∂Z.generator_ids, union(input_zono.Z₁.generator_ids, input_zono.Z₂.generator_ids))
    
    owned_generator_idx₁, owned_generator_idx₂, ∂owned_generator_idx = nothing, nothing, nothing
    owned_generator_id₁, owned_generator_id₂, ∂owned_generator_id = nothing, nothing, nothing
    if diff_layer.layer_idx == input_zono_cache.first_usage
        owned_generator_idx₁ = input_zono.Z₁.owned_generators
        owned_generator_idx₂ = input_zono.Z₂.owned_generators
        ∂owned_generator_idx = input_zono.∂Z.owned_generators

        if !isnothing(owned_generator_idx₁)
            owned_generator_id₁ = input_zono.Z₁.generator_ids[owned_generator_idx₁]
        end
        if !isnothing(owned_generator_idx₂)
            owned_generator_id₂ = input_zono.Z₂.generator_ids[owned_generator_idx₂]
        end
        if !isnothing(∂owned_generator_idx)
            ∂owned_generator_id = input_zono.∂Z.generator_ids[∂owned_generator_idx]
        end
    end
    
    num_rows = length(input_zono.∂Z.c)
    generators₁, generators₂, ∂generators = Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[]
    for gid in all_generator_ids
        if gid ∈ input_zono.Z₁.generator_ids && gid != owned_generator_id₂ && gid != ∂owned_generator_id
            idx = find_index_position(input_zono.Z₁.generator_ids, gid)
            _, num_columns = size(input_zono.Z₁.Gs[idx])
            if gid == owned_generator_id₁
                num_columns += new_gen₁
            end
        elseif gid ∈ input_zono.Z₂.generator_ids && gid != ∂owned_generator_id
            idx = find_index_position(input_zono.Z₂.generator_ids, gid)
            _, num_columns = size(input_zono.Z₂.Gs[idx])
            if gid == owned_generator_id₂
                num_columns += new_gen₂
            end
        else
            idx = find_index_position(input_zono.∂Z.generator_ids, gid)
            _, num_columns = size(input_zono.∂Z.Gs[idx])
            if gid == ∂owned_generator_id
                num_columns += ∂new_gen
            end
        end
        push!(generators₁, zeros(num_rows, num_columns))
        push!(generators₂, zeros(num_rows, num_columns))
        push!(∂generators, zeros(num_rows, num_columns))
    end

    if isnothing(owned_generator_id₁)
        owned_generator_id₁ = get_free_generator_id!(PS)
        push!(generators₁, zeros(num_rows, new_gen₁))
        push!(generators₂, zeros(num_rows, new_gen₁))
        push!(∂generators, zeros(num_rows, new_gen₁))
        push!(all_generator_ids, owned_generator_id₁)
    end
    
    if isnothing(owned_generator_id₂)
        owned_generator_id₂ = get_free_generator_id!(PS)
        push!(generators₁, zeros(num_rows, new_gen₂))
        push!(generators₂, zeros(num_rows, new_gen₂))
        push!(∂generators, zeros(num_rows, new_gen₂))
        push!(all_generator_ids, owned_generator_id₂)
    end
    
    if isnothing(∂owned_generator_id)
        ∂owned_generator_id = get_free_generator_id!(PS)
        push!(generators₁, zeros(num_rows, ∂new_gen))
        push!(generators₂, zeros(num_rows, ∂new_gen))
        push!(∂generators, zeros(num_rows, ∂new_gen))
        push!(all_generator_ids, ∂owned_generator_id)
    end

    owned_generator_idx₁ = find_index_position(all_generator_ids, owned_generator_id₁)
    owned_generator_idx₂ = find_index_position(all_generator_ids, owned_generator_id₂)
    ∂owned_generator_idx = find_index_position(all_generator_ids, ∂owned_generator_id)

    @assert length(generators₁) == length(generators₂) == length(∂generators) == length(all_generator_ids) "All zonotopes must have the same number of generator matrices"

    influence₁, influence₂ = nothing, nothing
    if !isnothing(input_zono.Z₁.influence)
        influence₁ = Matrix{Float64}[]
        for (idx, g) in enumerate(generators₁)
            if idx ∈ [owned_generator_idx₁, owned_generator_idx₂, ∂owned_generator_idx]
                push!(influence₁, zeros(size(input_zono.Z₁.influence[1], 1), size(g, 2)))
            else
                push!(influence₁, input_zono.Z₁.influence[idx])
            end
        end
    end
    if !isnothing(input_zono.Z₂.influence)
        influence₂ = Matrix{Float64}[]
        for (idx, g) in enumerate(generators₂)
            if idx ∈ [owned_generator_idx₁, owned_generator_idx₂, ∂owned_generator_idx]
                push!(influence₂, zeros(size(input_zono.Z₂.influence[1], 1), size(g, 2)))
            else
                push!(influence₂, input_zono.Z₂.influence[idx])
            end
        end
    end

    c₁, c₂, ∂c = zeros(num_rows), zeros(num_rows), zeros(num_rows)
    Z₁ = Zonotope(generators₁, c₁, influence₁, deepcopy(all_generator_ids), owned_generator_idx₁)
    Z₂ = Zonotope(generators₂, c₂, influence₂, deepcopy(all_generator_ids), owned_generator_idx₂)
    ∂Z = Zonotope(∂generators, ∂c, input_zono.∂Z.influence, all_generator_ids, ∂owned_generator_idx)
    
    Z = CachedZonotope(DiffZonotope(Z₁, Z₂, ∂Z), nothing)
    init_default_zono(Z)
    PS.zono_storage.zonotopes[output_positions[1]] = Z
end

# function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ONNXRelu{S1}, ONNXRelu{S2}, ONNXRelu{S3}}, inputs :: Vector{CachedZonotope}, output_positions :: Vector{Int64}) where {S1, S2, S3}
#     @assert length(inputs) == 1 "ReLU DiffLayer should have exactly one input"
#     @assert length(output_positions) == 1 "ReLU DiffLayer should have exactly one output"
#     input_zono_cache = inputs[1]
#     input_zono = get_zonotope(input_zono_cache)
#     # Compute Bounds
#     bounds₁ = zono_bounds(input_zono.Z₁)
#     bounds₂ = zono_bounds(input_zono.Z₂)
#     ∂bounds = zono_bounds(input_zono.∂Z)
#     (
#         _, _, _, _, _,
#         any_neg,
#         neg_any,
#         any_pos,
#         pos_any,
#         any_any
#     ) = get_selectors(bounds₁, bounds₂, ∂bounds)
#     # Do NOT use counts created above for new_gen₁ / new_gen₂,
#     # because these omit dimensions where difference is still zero
#     new_gen₁ = count(bounds₁[:,1] .< 0.0 .&& bounds₁[:,2] .> 0.0)
#     new_gen₂ = count(bounds₂[:,1] .< 0.0 .&& bounds₂[:,2] .> 0.0)
#     ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
#     Z₁ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₁, new_gen₁, diff_layer.layer_idx)
#     Z₂ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₂, new_gen₂, diff_layer.layer_idx)
#     generators_d = Matrix{Float64}[]
#     # Three way merge of generators: All generators from ∂Z + all from new Z₁ + all from new Z₂
#     # First build union of generator ids
#     generator_ids = union(input_zono.∂Z.generator_ids, union(Z₁.generator_ids, Z₂.generator_ids))
#     owned_generator_id = nothing
#     if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
#         owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
#     end
#     # Now iterate over generator ids and figure out where the generators come from
#     # Prefer Z₁ and Z₂ over ∂Z when there are overlaps because those might have new generators
#     for gid in generator_ids
#         if gid in Z₂.generator_ids
#             idx = find_index_position(Z₂.generator_ids, gid)
#             new_g = zeros(size(Z₂.Gs[idx],1), size(Z₂.Gs[idx],2))
#             push!(generators_d, new_g)
#         elseif gid in Z₁.generator_ids
#             idx = find_index_position(Z₁.generator_ids, gid)
#             new_g = zeros(size(Z₁.Gs[idx],1), size(Z₁.Gs[idx],2))
#             push!(generators_d, new_g)
#         else
#             idx = find_index_position(input_zono.∂Z.generator_ids, gid)
#             columns = size(input_zono.∂Z.Gs[idx],2)
#             if gid == owned_generator_id
#                 columns += ∂new_gen
#             end
#             new_g = zeros(size(input_zono.∂Z.Gs[idx],1), columns)
#             push!(generators_d, new_g)
#         end
#     end
#     if isnothing(owned_generator_id)
#         owned_generator_id = get_free_generator_id!(PS)
#         new_g = zeros(Float64, size(input_zono.∂Z.c,1), ∂new_gen)
#         push!(generators_d, new_g)
#         push!(generator_ids, owned_generator_id)
#     end
#     c = zeros(Float64, size(Z₂.c,1))
#     ∂Z = Zonotope(generators_d, c, input_zono.∂Z.influence, generator_ids, find_index_position(generator_ids, owned_generator_id))
#     @assert isnothing(input_zono.∂Z.influence) "ReLU DiffLayer does not support influenced zonotopes (yet?)"
#     Z = CachedZonotope(
#             DiffZonotope(
#                 Z₁,
#                 Z₂,
#                 ∂Z
#             ),
#             nothing
#         )
#     init_default_zono(Z)
#     PS.zono_storage.zonotopes[output_positions[1]] = Z
# end

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
    if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
        owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    else
        owned_generator_id = nothing
    end
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
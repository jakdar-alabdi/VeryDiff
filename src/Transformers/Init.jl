function init_default_zono(Z :: CachedZonotope)
    Z.zonotope = DiffZonotope(
        Zonotope(
            Vector{AbstractMatrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.Z₁.Gs]),
            Z.zonotope_proto.Z₁.c,
            (isnothing(Z.zonotope_proto.Z₁.influence) ? nothing : Vector{AbstractMatrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.Z₁.influence])),
            Z.zonotope_proto.Z₁.generator_ids,
            Z.zonotope_proto.Z₁.owned_generators
        ),
        Zonotope(
            Vector{AbstractMatrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.Z₂.Gs]),
            Z.zonotope_proto.Z₂.c,
            (isnothing(Z.zonotope_proto.Z₂.influence) ? nothing : Vector{AbstractMatrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.Z₂.influence])),
            Z.zonotope_proto.Z₂.generator_ids,
            Z.zonotope_proto.Z₂.owned_generators
        ),
        Zonotope(
            Vector{AbstractMatrix{Float64}}([(@view g[:, :] ) for g in Z.zonotope_proto.∂Z.Gs]),
            Z.zonotope_proto.∂Z.c,
            (isnothing(Z.zonotope_proto.∂Z.influence) ? nothing : Vector{<:AbstractMatrix{Float64}}([(@view inf[:, :] ) for inf in Z.zonotope_proto.∂Z.influence])),
            Z.zonotope_proto.∂Z.generator_ids,
            Z.zonotope_proto.∂Z.owned_generators
        )
    )
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ReLU, ReLU, ReLU}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "ReLU DiffLayer should have exactly one input"
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
    # @info "Initiating Z₁ ($new_gen₁ new generators)"
    Z₁ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₁, new_gen₁, diff_layer.layer_idx)
    # @info "Initiating Z₂ ($new_gen₂ new generators)"
    Z₂ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₂, new_gen₂, diff_layer.layer_idx)
    generators_d = Matrix{Float64}[]
    # Three way merge of generators: All generators from ∂Z + all from new Z₁ + all from new Z₂
    # First build union of generator ids
    generator_ids = union(input_zono.∂Z.generator_ids, union(Z₁.generator_ids, Z₂.generator_ids))
    owned_generator_id = nothing
    if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
        owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    end
    # @info "Initiating ∂Z ($∂new_gen new generators)"
    # Now iterate over generator ids and figure out where the generators come from
    # Prefer Z₁ and Z₂ over ∂Z when there are overlaps because those might have new generators
    for gid in generator_ids
        if gid in Z₂.generator_ids
            idx = find_index_position(Z₂.generator_ids, gid)
            new_g = zeros(size(Z₂.Gs[idx],1), size(Z₂.Gs[idx],2))
            # @info "Generator ID $gid: $(size(new_g,2)) columns (from Z₂)"
            push!(generators_d, new_g)
        elseif gid in Z₁.generator_ids
            idx = find_index_position(Z₁.generator_ids, gid)
            new_g = zeros(size(Z₁.Gs[idx],1), size(Z₁.Gs[idx],2))
            # @info "Generator ID $gid: $(size(new_g,2)) columns (from Z₁)"
            push!(generators_d, new_g)
        else
            idx = find_index_position(input_zono.∂Z.generator_ids, gid)
            columns = size(input_zono.∂Z.Gs[idx],2)
            if gid == owned_generator_id
                columns += ∂new_gen
            end
            new_g = zeros(size(input_zono.∂Z.Gs[idx],1), columns)
            # @info "Generator ID $gid: $(size(new_g,2)) columns (from ∂Z)"
            push!(generators_d, new_g)
        end
    end
    if isnothing(owned_generator_id)
        owned_generator_id = get_free_generator_id!(PS)
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
    push!(PS.zono_storage.zonotopes, Z)
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{Dense, ZeroDense, Dense}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    L2 = get_layer2(diff_layer)
    @assert size(L1.W,2) == size(input_zono.Z₁.Gs[1],1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2.W,2) == size(input_zono.Z₂.Gs[1],1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(L1, L2, input_zono, input_zono_cache, diff_layer.layer_idx)
    ∂Z = init_zonotope(L2, input_zono.∂Z, input_zono.∂Z.influence, input_zono.∂Z.owned_generators)
    Z =CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    push!(PS.zono_storage.zonotopes, Z)
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{Dense,Dense,Dense}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    diff_layer_output =get_diff_layer(diff_layer)
    L2 = get_layer2(diff_layer)
    @assert size(L1.W,2) == size(input_zono.Z₁.Gs[1],1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2.W,2) == size(input_zono.Z₂.Gs[1],1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(L1, L2, input_zono, input_zono_cache, diff_layer.layer_idx)
    # Instantiate ∂Z
    influence = input_zono.∂Z.influence
    if diff_layer.layer_idx == input_zono_cache.first_usage && !isnothing(input_zono.∂Z.owned_generators)
        owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    else
        owned_generator_id = nothing
    end
    generators = Vector{AbstractMatrix{Float64}}()
    generator_ids = SortedVector{Int64}()
    i_d = 1
    i_2 = 1
    while i_d <= length(input_zono.∂Z.generator_ids) && i_2 <= length(input_zono.Z₂.generator_ids)
        if input_zono.∂Z.generator_ids[i_d] == input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
            i_2 += 1
        elseif input_zono.∂Z.generator_ids[i_d] < input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
        else
            # input_zono.∂Z.generator_ids[i_d] > input_zono.Z₂.generator_ids[i_2]
            new_g = zeros(Float64, size(diff_layer_output.W,1), size(input_zono.Z₂.Gs[i_2],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
            i_2 += 1
        end
    end
    while i_d <= length(input_zono.∂Z.generator_ids)
        new_g = zeros(Float64, size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
        i_d += 1
    end
    while i_2 <= length(input_zono.Z₂.generator_ids)
        new_g = zeros(Float64, size(diff_layer_output.W,1), size(input_zono.Z₂.Gs[i_2],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
        i_2 += 1
    end
    ∂Z = Zonotope(generators,
    zeros(Float64, size(diff_layer_output.W,1)), influence, generator_ids, owned_generator_id === nothing ? nothing : find_index_position(generator_ids, owned_generator_id))
    Z = CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    )
    init_default_zono(Z)
    push!(PS.zono_storage.zonotopes, Z)
end
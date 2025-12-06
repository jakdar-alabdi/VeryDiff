function init_layer!(PS :: PropState, diff_layer :: DiffLayer{ReLU, ReLU, ReLU}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "ReLU DiffLayer should have exactly one input"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    # Compute Bounds
    bounds₁ = zono_bounds(input_zono.Z₁)
    bounds₂ = zono_bounds(input_zono.Z₂)
    ∂bounds = zono_bounds(Z.∂Z)
    (
        _, _, _, _, _,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    ) = get_selectors(bounds₁, bounds₂, ∂bounds)
    new_gen₁ = count(any_neg) + count(any_pos) + count(any_any)
    new_gen₂ = count(neg_any) + count(pos_any) + count(any_any)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
    Z₁ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₁, new_gen₁, diff_layer.layer_idx)
    Z₂ = init_relu_zonotope(PS, input_zono_cache, input_zono.Z₂, new_gen₂, diff_layer.layer_idx)
    generators_d = Matrix{Float64}[]
    # Three way merge of generators: All generators from ∂Z + all from new Z₁ + all from new Z₂
    # First build union of generator ids
    generator_ids = union(input_zono.∂Z.generator_ids, union(Z₁.generator_ids, Z₂.generator_ids))
    owned_generator_id = nothing
    if diff_layer.layer_idx == input_zono_cache.first_usage
        owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    end
    # Now iterate over generator ids and figure out where the generators come from
    # Prefer Z₁ and Z₂ over ∂Z when there are overlaps because those might have new generators
    for gid in generator_ids
        if gid in Z₂.generator_ids
            idx = findfirst(==(gid), Z₂.generator_ids)
            new_g = zeros(size(Z₂.Gs[idx],1), size(Z₂.Gs[idx],2))
            push!(generators_d, new_g)
        elseif gid in Z₁.generator_ids
            idx = findfirst(==(gid), Z₁.generator_ids)
            new_g = zeros(size(Z₁.Gs[idx],1), size(Z₁.Gs[idx],2))
            push!(generators_d, new_g)
        else
            idx = findfirst(==(gid), input_zono.∂Z.generator_ids)
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
        new_g = zeros(Float64, size(input_zono.∂Z.Gs[1],1), ∂new_gen)
        push!(generators_d, new_g)
        push!(generator_ids, owned_generator_id)
    end
    c = similar(input_zono.∂Z.c, size(Z₂.c,1))
    ∂Z = Zonotope(generators_d, c, input_zono.∂Z.influence, generator_ids, findfirst(==(owned_generator_id), generator_ids))
    @assert isnothing(input_zono.∂Z.influence) "ReLU DiffLayer does not support influenced zonotopes (yet?)"
    push!(PS.zono_storage.zonotopes, CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    ))
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{Dense, ZeroDense, Dense}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    L2 = get_layer2(diff_layer)
    @assert size(L1.W,2) == size(input_zono.Z₁,1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2.W,2) == size(input_zono.Z₂,1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(L1, L2, input_zono, input_zono_cache, diff_layer.layer_idx)
    ∂Z = init_zonotope(L2, input_zono.∂Z, input_zono.∂Z.influence, input_zono.∂Z.owned_generators)
    push!(PS.zono_storage.zonotopes, CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    ))
end

function init_layer!(PS :: PropState, diff_layer :: DiffLayer{Dense,Dense,Dense}, inputs :: Vector{CachedZonotope})
    @assert length(inputs) == 1 "Dense DiffLayer should have exactly one input"
    input_zono_cache = inputs[1]
    input_zono = get_zonotope(input_zono_cache)
    L1 = get_layer1(diff_layer)
    diff_layer_output =get_diff_layer(diff_layer)
    L2 = get_layer2(diff_layer)
    @assert size(L1.W,2) == size(input_zono.Z₁,1) "Input dimension mismatch for Dense layer 1"
    @assert size(L2.W,2) == size(input_zono.Z₂,1) "Input dimension mismatch for Dense layer 2"
    Z₁, Z₂ = init_layer_dense_z1_z2(L1, L2, input_zono, input_zono_cache, diff_layer.layer_idx)
    # Instantiate ∂Z
    influence = input_zono.∂Z.influence
    if diff_layer.layer_idx == input_zono_cache.first_usage
        owned_generator_id = input_zono.∂Z.generator_ids[input_zono.∂Z.owned_generators]
    else
        owned_generator_id = nothing
    end
    generators = Matrix{Float64}[]
    generator_ids = Int64[]
    i_d = 1
    i_2 = 1
    while i_d <= length(input_zono.∂Z.generator_ids) && i_2 <= length(input_zono.Z₂.generator_ids)
        if input_zono.∂Z.generator_ids[i_d] == input_zono.Z₂.generator_ids[i_2]
            new_g = similar(input_zono.∂Z.Gs[i_d], size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
            i_2 += 1
        elseif input_zono.∂Z.generator_ids[i_d] < input_zono.Z₂.generator_ids[i_2]
            new_g = similar(input_zono.∂Z.Gs[i_d], size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
            i_d += 1
        else
            # input_zono.∂Z.generator_ids[i_d] > input_zono.Z₂.generator_ids[i_2]
            new_g = similar(input_zono.Z₂.Gs[i_2], size(diff_layer_output.W,1), size(input_zono.Z₂.Gs[i_2],2))
            push!(generators, new_g)
            push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
            i_2 += 1
        end
    end
    while i_d <= length(input_zono.∂Z.generator_ids)
        new_g = similar(input_zono.∂Z.Gs[i_d], size(diff_layer_output.W,1), size(input_zono.∂Z.Gs[i_d],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.∂Z.generator_ids[i_d])
        i_d += 1
    end
    while i_2 <= length(input_zono.Z₂.generator_ids)
        new_g = similar(input_zono.Z₂.Gs[i_2], size(diff_layer_output.W,1), size(input_zono.Z₂.Gs[i_2],2))
        push!(generators, new_g)
        push!(generator_ids, input_zono.Z₂.generator_ids[i_2])
        i_2 += 1
    end
    ∂Z = Zonotope(generators, similar(input_zono.∂Z.c, size(diff_layer_output.W,1)), influence, generator_ids, owned_generator_id === nothing ? nothing : findfirst(==(owned_generator_id), generator_ids))
    push!(PS.zono_storage.zonotopes, CachedZonotope(
        DiffZonotope(
            Z₁,
            Z₂,
            ∂Z
        ),
        nothing
    ))
end
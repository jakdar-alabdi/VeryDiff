function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXLinear{S1},
        ONNXLinear{S2},
        ONNXLinear{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing,
    data :: Union{Nothing,NeuronSplittingLayerData}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    # @debug "Propagating DiffDense Layer"
    Zin = inputs[1]
    # Compute differential zonotope dimensions
    # TODO(steuber): Is there a more elegant way?
    # At the very least we could probably extract this into a function
    i_out = 1
    ∂g_dims = Int64[]
    resize!(∂g_dims, length(ZoutRef.zonotope_proto.∂Z.Gs))
    for i_out in 1:length(∂g_dims)
        res = attempt_find_index_position(Zin.∂Z.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
        if res > 0
            ∂g_dims[i_out] = size(Zin.∂Z.Gs[res],2)
        else
            res = find_index_position(Zin.Z₂.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
            ∂g_dims[i_out] = size(Zin.Z₂.Gs[res],2)
        end
    end
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), ∂g_dims)
    L1 = get_layer1(Ls)
    ∂L = get_diff_layer(Ls)
    L2 = get_layer2(Ls)
    L1_W = L1.dense.weight
    ∂L_W = ∂L.dense.weight
    ∂L_b = ∂L.dense.bias
    if VeryDiff.USE_DIFFZONO[]
        # @debug "IDs of Output Zonotope Generators: $(Zout.∂Z.generator_ids)"
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1_W, g)
        end
        indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zin.Z₂.generator_ids)
        for (i, g) in zip(indices₂, Zin.Z₂.Gs)
            mul!(Zout.∂Z.Gs[i], ∂L_W, g, 1.0, 1.0)
        end
        # @assert length(intersect_indices(Zout.∂Z.generator_ids, union(Zin.∂Z.generator_ids, Zin.Z₂.generator_ids))) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation!"
        mul!(Zout.∂Z.c, L1_W, Zin.∂Z.c)
        mul!(Zout.∂Z.c, ∂L_W, Zin.Z₂.c, 1.0, 1.0)
        Zout.∂Z.c .+= ∂L_b
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXAddConst{S1},
        ONNXAddConst{S2},
        ONNXAddConst{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing,
    data :: Union{Nothing,NeuronSplittingLayerData}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    i_out = 1
    ∂g_dims = Int64[]
    resize!(∂g_dims, length(ZoutRef.zonotope_proto.∂Z.Gs))
    for i_out in 1:length(∂g_dims)
        res = attempt_find_index_position(Zin.∂Z.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
        if res > 0
            ∂g_dims[i_out] = size(Zin.∂Z.Gs[res],2)
        else
            res = find_index_position(Zin.Z₂.generator_ids, ZoutRef.zonotope_proto.∂Z.generator_ids[i_out])
            ∂g_dims[i_out] = size(Zin.Z₂.Gs[res],2)
        end
    end
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), ∂g_dims)
    L1 = get_layer1(Ls)
    ∂L = get_diff_layer(Ls)
    ∂L_b = ∂L.c
    L2 = get_layer2(Ls)
    if VeryDiff.USE_DIFFZONO[]
        # @debug "IDs of Output Zonotope Generators: $(Zout.∂Z.generator_ids)"
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            Zout.∂Z.Gs[i] .= g
        end
        Zout.∂Z.c .= Zin.∂Z.c .+ ∂L_b
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXLinear{S1},
        ZeroDense{S2},
        ONNXLinear{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing,
    data :: Union{Nothing,NeuronSplittingLayerData}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), convert(Vector{Int64},size.(Zin.∂Z.Gs,2)))
    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    L1_W = L1.dense.weight
    L2_W = L2.dense.weight
    L1_b = L1.dense.bias
    L2_b = L2.dense.bias
    if VeryDiff.USE_DIFFZONO[]
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        # @assert length(union(Zout.∂Z.generator_ids,Zin.∂Z.generator_ids)) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation. Output IDs: $(Zout.∂Z.generator_ids), Processed IDs: $(Zin.∂Z.generator_ids)"
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1_W, g)
        end
        mul!(Zout.∂Z.c, L1_W, Zin.∂Z.c)
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
end

function range(lower, upper)
    return (upper .- lower)
end

function α(lower, upper)
    return (.-lower ./ range(lower, upper))
end

function ∂λ(∂lower, ∂upper)
    return (clamp.(∂upper ./ range(∂lower, ∂upper),0.0,1.0))
end

function μ(lower, upper)
    return (0.5 .* α(lower, upper) .* upper)
end

function ∂μ(∂lower, ∂upper)
    return (0.5 .* max.(.-∂lower, ∂upper))
end

function ∂ν(∂lower, ∂upper)
    return (∂λ(∂lower, ∂upper) .* max.(0.0, .-∂lower))
end

function ∂a(any_any, ∂lower, ∂upper)
    return ifelse(any_any, ∂λ(∂lower,∂upper), 1.0)
        #ifelse.(pos_pos .|| any_pos .|| pos_any, 1.0, 0.0))
end

function propagate_layer!(
    ZoutRefVec :: Vector{CachedZonotope},
    Ls :: DiffLayer{
        ONNXRelu{S1},
        ONNXRelu{S2},
        ONNXRelu{S3}},
    inputs :: Vector{DiffZonotope};
    bounds_cache :: Union{Nothing,BoundsCache}=nothing,
    data :: Union{Nothing,NeuronSplittingLayerData}=nothing) where {S1, S2, S3}
    @assert length(inputs) == 1 "ReLU layer should have exactly one input zonotope"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output zonotope"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

    if !isnothing(data) && VeryDiff.INTER_CONTRACT[] && !isempty(data.split_nodes)
        sort_split_nodes!(data.split_nodes, Zin)
        box = InputBox(Zin)
        contract_zono_all!(box, data.split_nodes, Zin)
        if isnothing(box)
            data.is_unsatisfiable = true
            return
        end
        if !is_unit_hypercube(box)
            transform_offset_diff_zono!(box, Zin)
            transform_verification_task!(box, data.task)
        end
    end

    # Compute Bounds
    bounds₁ = zono_bounds(Zin.Z₁)
    bounds₂ = zono_bounds(Zin.Z₂)
    ∂bounds = zono_bounds(Zin.∂Z)
    
    if !bounds_cache.initialized
        bounds_cache.lower₁ = copy(bounds₁[:,1])
        bounds_cache.upper₁ = copy(bounds₁[:,2])
        bounds_cache.lower₂ = copy(bounds₂[:,1])
        bounds_cache.upper₂ = copy(bounds₂[:,2])
        bounds_cache.∂lower = copy(∂bounds[:,1])
        bounds_cache.∂upper = copy(∂bounds[:,2])
        bounds_cache.crossing₁ = bounds_cache.lower₁ .< 0.0 .&& bounds_cache.upper₁ .> 0.0
        bounds_cache.crossing₂ = bounds_cache.lower₂ .< 0.0 .&& bounds_cache.upper₂ .> 0.0
        bounds_cache.initialized = true
    end

    split_nodes₁, split_nodes₂ = nothing, nothing
    if VeryDiff.USE_NEURON_SPLITTING[]
        if !isnothing(data) && !isempty(data.split_nodes)
            bounds = (bounds₁, bounds₂)
            if VeryDiff.USE_VERTICAL_SPLITTING[]
                cached_lowers = (bounds_cache.lower₁, bounds_cache.lower₂)
                cached_uppers = (bounds_cache.upper₁, bounds_cache.upper₂)
                for node in data.split_nodes
                    net, n = node.network, node.neuron
                    l = max(bounds[net][n, 1], cached_lowers[net][n])
                    u = min(bounds[net][n, 2], cached_uppers[net][n])
                    if l < 0.0 && u > 0.0
                        if node.direction == 1
                            if isnothing(node.bounds)
                                node.bounds = [l u] ./ 2
                            end
                            l̲, u̲ = node.bounds[1], node.bounds[2]
                            l, u = max(l, l̲), min(u, u̲)
                            node.bounds = [l u]
                            data.is_unsatisfiable |= l > u
                        else
                            if isnothing(node.bounds)
                                s₁, s₂ = (l, u) ./ 2
                                node.bounds = [l s₁; s₂ u]
                            end
                            l̅, s₁ = node.bounds[1, 1], node.bounds[1, 2]
                            s₂, u̅ = node.bounds[2, 1], node.bounds[2, 2]
                            l = max(l, ifelse(l >= s₁, s₂, l̅))
                            u = min(u, ifelse(u <= s₂, s₁, u̅))
                            # l = ifelse(l >= s₁, s₂, max(l, l̅))
                            # u = ifelse(u <= s₂, s₁, min(u, u̅))
                            node.bounds = [l s₁; s₂ u]
                            data.is_unsatisfiable |= l >= s₁ && u <= s₂
                        end
                        if data.is_unsatisfiable
                            break
                        end
                    end
                    bounds[net][n, 1] = l
                    bounds[net][n, 2] = u
                end
                if data.is_unsatisfiable
                   return
                end
            else
                for (;network, neuron, direction) in data.split_nodes
                    if direction == 1
                        bounds[network][neuron, 1] = max(bounds[network][neuron, 1], 0.0)
                    else
                        bounds[network][neuron, 2] = min(bounds[network][neuron, 2], 0.0)
                    end
                end
            end

            split_nodes₁ = @view data.split_nodes[findall(n -> n.network == 1, data.split_nodes)]
            split_nodes₂ = @view data.split_nodes[findall(n -> n.network == 2, data.split_nodes)]
        end

        bounds₁[:, 1] .= max.(bounds₂[:, 1] .+ ∂bounds[:, 1], bounds₁[:, 1], bounds_cache.lower₁)
        bounds₁[:, 2] .= min.(bounds₂[:, 2] .+ ∂bounds[:, 2], bounds₁[:, 2], bounds_cache.upper₁)
        bounds₂[:, 1] .= max.(bounds₁[:, 1] .- ∂bounds[:, 2], bounds₂[:, 1], bounds_cache.lower₂)
        bounds₂[:, 2] .= min.(bounds₁[:, 2] .- ∂bounds[:, 1], bounds₂[:, 2], bounds_cache.upper₂)
        ∂bounds[:, 1] .= max.(bounds₁[:, 1] .- bounds₂[:, 2], ∂bounds[:, 1], bounds_cache.∂lower)
        ∂bounds[:, 2] .= min.(bounds₁[:, 2] .- bounds₂[:, 1], ∂bounds[:, 2], bounds_cache.∂upper)
    end

    lower₁ = bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
    upper₁ = bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
    lower₂ = bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
    upper₂ = bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
    ∂lower = bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
    ∂upper = bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    #@info "Bounds Cache: Z₁=[$(lower₁), $(upper₁)], Z₂=[$(lower₂), $(upper₂)], ∂Z=[$(∂lower), $(∂upper)]"

    (
        zero_diff,
        neg_neg,
        neg_pos,
        pos_neg,
        pos_pos,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    ) = get_selectors(bounds₁, bounds₂, ∂bounds)
    # Do NOT use counts created above for new_gen₁ / new_gen₂,
    # because these omit dimensions where difference is still zero
    bounds_cache.crossing₁ .= lower₁ .< 0.0 .&& upper₁ .> 0.0
    bounds_cache.crossing₂ .= lower₂ .< 0.0 .&& upper₂ .> 0.0
    new_gen₁ = count(bounds_cache.crossing₁)
    new_gen₂ = count(bounds_cache.crossing₂)
    data.num_instable += new_gen₁ + new_gen₂
    # new_gen₁ = count(lower₁ .< 0.0 .&& upper₁ .> 0.0)
    # new_gen₂ = count(lower₂ .< 0.0 .&& upper₂ .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
    # @debug "Instable Neurons: Network 1: $new_gen₁, Network 2: $new_gen₂, Differential: $∂new_gen"
    Zout_proto = ZoutRef.zonotope_proto # Need this to be able to access the generator ids
    gen_sizes₁ = zeros(Int64,length(Zout_proto.Z₁.generator_ids))
    gen_sizes₂ = zeros(Int64,length(Zout_proto.Z₂.generator_ids))
    ∂gen_sizes = zeros(Int64,length(Zout_proto.∂Z.generator_ids))

    pre_indices_Z₁ = intersect_indices(Zout_proto.Z₁.generator_ids, Zin.Z₁.generator_ids)
    pre_indices_Z₂ = intersect_indices(Zout_proto.Z₂.generator_ids, Zin.Z₂.generator_ids)
    pre_indices₁ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₁.generator_ids)
    pre_indices₂ = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.Z₂.generator_ids)
    ∂pre_indices = intersect_indices(Zout_proto.∂Z.generator_ids, Zin.∂Z.generator_ids)

    for (i, idx) in enumerate(pre_indices_Z₁)
        gen_sizes₁[idx] = size(Zin.Z₁.Gs[i],2)
    end
    gen_sizes₁[Zout_proto.Z₁.owned_generators] += new_gen₁
    for (i, idx) in enumerate(pre_indices_Z₂)
        gen_sizes₂[idx] = size(Zin.Z₂.Gs[i],2)
    end
    gen_sizes₂[Zout_proto.Z₂.owned_generators] += new_gen₂
    # This mayoverwrite sizes, but columns should be consistent
    # TODO(steuber): Can we make this cleaner?
    for (i, idx) in enumerate(∂pre_indices)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.∂Z.Gs[i],2)) (from ∂Z)"
        ∂gen_sizes[idx] = size(Zin.∂Z.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₁)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₁.Gs[i],2)) (from Z₁)"
        ∂gen_sizes[idx] = size(Zin.Z₁.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₂)
        # @info "Setting ∂Z generator $idx size to $(size(Zin.Z₂.Gs[i],2)) (from Z₂)"
        ∂gen_sizes[idx] = size(Zin.Z₂.Gs[i],2)
    end
    # @info "Generator sizes before new gens: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    ∂old_gen = ∂gen_sizes[Zout_proto.∂Z.owned_generators]
    ∂gen_sizes[Zout_proto.∂Z.owned_generators] += ∂new_gen
    # Find idx of generators owned by Z₁ and Z₂ in ∂Z
    idx1 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₁.generator_ids[Zout_proto.Z₁.owned_generators])
    idx2 = find_index_position(Zout_proto.∂Z.generator_ids, Zout_proto.Z₂.generator_ids[Zout_proto.Z₂.owned_generators])
    ∂gen_sizes[idx1] += new_gen₁
    ∂gen_sizes[idx2] += new_gen₂
    Zout_proto = nothing # Avoid missuse
    # @info "ReLU DiffZonotope Generators: Z₁=$(gen_sizes₁), Z₂=$(gen_sizes₂), ∂Z=$(∂gen_sizes)"
    Zout = get_zonotope!(ZoutRef, gen_sizes₁, gen_sizes₂, ∂gen_sizes)
    post_indices₁ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₁.generator_ids)
    post_indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₂.generator_ids)

    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    # Compute Zonotopes for individual networks
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁, split_nodes=split_nodes₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂, split_nodes=split_nodes₂)

    if VeryDiff.USE_DIFFZONO[]
        dim = length(any_neg)
        â₁_pos = @simd_bool_expr dim (any_neg | pos_neg)
        a₁_pos = any_pos
        â₂_pos = @simd_bool_expr dim (neg_any | neg_pos)
        a₂_pos = pos_any
        ∂a_pos_∂λ = any_any
        # This one *must* be addition
        ∂a_pos_1 = @simd_bool_expr dim (any_pos | pos_any)
        ∂a_pos_1_assign = pos_pos
        
        # Reset to zero
        Zout.∂Z.c .= 0.0
        selector = @simd_bool_expr dim (neg_neg | zero_diff)
        for g in Zout.∂Z.Gs
            g[selector, :] .= 0.0
        end
        
        # Assign Zin.Z₁ with a₁
        cur_α₁ = .-α.((@view lower₁[a₁_pos]), (@view upper₁[a₁_pos]))
        updateGeneratorsMul!(Zout.∂Z.Gs, pre_indices₁, Zin.Z₁.Gs, cur_α₁, a₁_pos)
        Zout.∂Z.c[a₁_pos] .= cur_α₁ .* (@view Zin.Z₁.c[a₁_pos])

        # Assign Zout.Z₁ with â₁ = 1
        updateGenerators!(Zout.∂Z.Gs, post_indices₁, Zout.Z₁.Gs, â₁_pos)
        Zout.∂Z.c[â₁_pos] .= (@view Zout.Z₁.c[â₁_pos])

        # Assign Zin.Z₂ with a₂
        cur_α₂ = α.((@view lower₂[a₂_pos]), (@view upper₂[a₂_pos]))
        updateGeneratorsMul!(Zout.∂Z.Gs, pre_indices₂, Zin.Z₂.Gs, cur_α₂, a₂_pos)
        Zout.∂Z.c[a₂_pos] .= cur_α₂ .* (@view Zin.Z₂.c[a₂_pos])

        # Assign Zout.Z₂ with â₂ = -1
        updateGeneratorsMul!(Zout.∂Z.Gs, post_indices₂, Zout.Z₂.Gs, -1.0, â₂_pos)
        Zout.∂Z.c[â₂_pos] .= .-(@view Zout.Z₂.c[â₂_pos])

        # Add Zin.∂Z with 1.0
        updateGeneratorsAdd!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, ∂a_pos_1)
        Zout.∂Z.c[∂a_pos_1] .+= (@view Zin.∂Z.c[∂a_pos_1])

        # Assign Zin.∂Z with 1.0
        updateGenerators!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, ∂a_pos_1_assign)
        Zout.∂Z.c[∂a_pos_1_assign] .= (@view Zin.∂Z.c[∂a_pos_1_assign])

        # Add Zin.∂Z with ∂λ
        cur_∂λ = ∂λ.((@view ∂lower[any_any]), (@view ∂upper[any_any]))
        # TODO(steuber): Add requires copy vs. assign does not!
        updateGeneratorsMul!(Zout.∂Z.Gs, ∂pre_indices, Zin.∂Z.Gs, cur_∂λ, ∂a_pos_∂λ)
        Zout.∂Z.c[∂a_pos_∂λ] .= cur_∂λ .* (@view Zin.∂Z.c[∂a_pos_∂λ])

        # Add new generators from c
        c_pos = findall(@simd_bool_expr dim (any_any | any_pos | pos_any))
        A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
        @inbounds for i in 1:length(c_pos)
            row = c_pos[i]
            col = ∂old_gen + i
            if any_any[row]
                A[row, col] = ∂μ(∂lower[row], ∂upper[row])
            elseif any_pos[row]
                A[row, col] = μ(lower₁[row], upper₁[row])
            else # pos_any[row]
                A[row, col] = μ(lower₂[row], upper₂[row])
            end
        end

        # Add bias
        Zout.∂Z.c .+= ifelse.(
            any_any, ∂ν.(∂lower, ∂upper) .- ∂μ.(∂lower, ∂upper),
                ifelse.(any_pos, μ.(lower₁,upper₁),
                    ifelse.(pos_any, .-μ.(lower₂,upper₂), 0.0)))
    end
end
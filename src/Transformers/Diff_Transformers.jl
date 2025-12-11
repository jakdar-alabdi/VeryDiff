import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

# TODO: ReLU, ZeroDense layers

function propagate_layer!(ZoutRef :: CachedZonotope, Ls :: DiffLayer{Dense,Dense,Dense}, inputs :: Vector{DiffZonotope}; bounds_cache :: Union{Nothing,BoundsCache}=nothing)
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    @debug "Propagating DiffDense Layer"
    Zin = inputs[1]
    #@timeit to "DiffZonotope_GetZonotope" begin
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
    if USE_DIFFZONO
        @debug "IDs of Output Zonotope Generators: $(Zout.∂Z.generator_ids)"
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1.W, g)
        end
        indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zin.Z₂.generator_ids)
        for (i, g) in zip(indices₂, Zin.Z₂.Gs)
            mul!(Zout.∂Z.Gs[i], ∂L.W, g, 1.0, 1.0)
        end
        @assert length(intersect_indices(Zout.∂Z.generator_ids, union(Zin.∂Z.generator_ids, Zin.Z₂.generator_ids))) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation!"
        mul!(Zout.∂Z.c, L1.W, Zin.∂Z.c)
        mul!(Zout.∂Z.c, ∂L.W, Zin.Z₂.c, 1.0, 1.0)
        Zout.∂Z.c .+= ∂L.b
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
    # Print Bounds:
    # @info "Z₁ Bounds: $(zono_bounds(Zout.Z₁))"
    # @info "Z₂ Bounds: $(zono_bounds(Zout.Z₂))"
    # @info "∂Z Bounds: $(zono_bounds(Zout.∂Z))"
end

function propagate_layer!(ZoutRef :: CachedZonotope, Ls :: DiffLayer{Dense,ZeroDense,Dense}, inputs :: Vector{DiffZonotope}; bounds_cache :: Union{Nothing,BoundsCache}=nothing)
    @assert length(inputs) == 1 "Dense layer should have exactly one input zonotope"
    Zin = inputs[1]
    Zout = get_zonotope!(ZoutRef, size.(Zin.Z₁.Gs,2), size.(Zin.Z₂.Gs,2), convert(Vector{Int64},size.(Zin.∂Z.Gs,2)))
    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    if USE_DIFFZONO
        ∂indices = intersect_indices(Zout.∂Z.generator_ids, Zin.∂Z.generator_ids)
        @assert length(union(Zout.∂Z.generator_ids,Zin.∂Z.generator_ids)) == length(Zout.∂Z.generator_ids) "Not all generators in ∂Z were processed during Dense propagation. Output IDs: $(Zout.∂Z.generator_ids), Processed IDs: $(Zin.∂Z.generator_ids)"
        for (i, g) in zip(∂indices, Zin.∂Z.Gs)
            mul!(Zout.∂Z.Gs[i], L1.W, g)
        end
        mul!(Zout.∂Z.c, L1.W, Zin.∂Z.c)
    end
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂)
    # Print Bounds:
    # @info "Z₁ Bounds: $(zono_bounds(Zout.Z₁))"
    # @info "Z₂ Bounds: $(zono_bounds(Zout.Z₂))"
    # @info "∂Z Bounds: $(zono_bounds(Zout.∂Z))"
end

# TODO(steuber): Remove unnecessary stuff from loops, e.g.:
# A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
# @inbounds for (i,row) in enumerate(findall(selector))
#     A[row, generator_offset] = abs(μ[i])
#     generator_offset += 1
# end

function propagate_layer_new!(ZoutRef :: CachedZonotope, Ls :: DiffLayer{ReLU,ReLU,ReLU}, inputs :: Vector{DiffZonotope}; bounds_cache :: Union{Nothing,BoundsCache}=nothing)
    @assert length(inputs) == 1 "ReLU layer should have exactly one input zonotope"
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

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
        bounds_cache.initialized = true
    else
        bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
        bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
        bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
        bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
        bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
        bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    end
    lower₁ = bounds_cache.lower₁
    upper₁ = bounds_cache.upper₁
    lower₂ = bounds_cache.lower₂
    upper₂ = bounds_cache.upper₂
    ∂lower = bounds_cache.∂lower
    ∂upper = bounds_cache.∂upper
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
    new_gen₁ = count(lower₁ .< 0.0 .&& upper₁ .> 0.0)
    new_gen₂ = count(lower₂ .< 0.0 .&& upper₂ .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
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
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂)

    if USE_DIFFZONO
        range₁ = upper₁ .- lower₁ #max.(1e-12, upper₁ .- lower₁)
        range₂ = upper₂ .- lower₂ #max.(1e-12, upper₂ .- lower₂)
        ∂range = ∂upper .- ∂lower #max.(1e-12, ∂upper .- ∂lower)

        λ₁ = .-lower₁ ./ range₁
        λ₂ = .-lower₂ ./ range₂
        ∂λ = clamp.(∂upper ./ ∂range,0.0,1.0)
        μ₁ = 0.5 .* λ₁ .* upper₁
        μ₂ = 0.5 .* λ₂ .* upper₂
        ∂μ = 0.5 .* max.(.-∂lower, ∂upper)
        ∂ν = ∂λ .* max.(0.0, .-∂lower)

        â₁ = ifelse.(any_neg .|| pos_neg, 1.0, 0.0)
        a₁ = ifelse.(any_pos, -λ₁, 0.0)
        â₂ = ifelse.(neg_any .|| neg_pos, -1.0, 0.0)
        a₂ = ifelse.(pos_any, λ₂, 0.0)
        ∂a = ifelse.(any_any, ∂λ,
                ifelse.(pos_pos .|| any_pos .|| pos_any, 1.0, 0.0))
        b = ifelse.(any_any, ∂ν .- ∂μ,
                ifelse.(any_pos, μ₁,
                ifelse.(pos_any, .-μ₂, 0.0)))
        c = abs.(ifelse.(any_any, ∂μ,
                ifelse.(any_pos, μ₁,
                ifelse.(pos_any, μ₂,
                0.0))))
        
        # Reset to zero
        Zout.∂Z.c .= 0.0
        for g in Zout.∂Z.Gs
            g .= 0.0
        end
        
        # Add Zin.Z₁ with a₁
        debug_j = 1
        for (i, g) in zip(pre_indices₁, Zin.Z₁.Gs)
            cols = size(g,2)
            @assert Zout.∂Z.generator_ids[i] == Zin.Z₁.generator_ids[debug_j] "Generator ID mismatch between Zin.Z₁ ($(Zin.Z₁.generator_ids[debug_j])) and Zout.∂Z ($(Zout.∂Z.generator_ids[i]))!"
            debug_j += 1
            Zout.∂Z.Gs[i][:,1:cols] .+= a₁ .* g
        end
        Zout.∂Z.c .+= a₁ .* Zin.Z₁.c
        # Add Zout.Z₁ with â₁
        for (i, g) in zip(post_indices₁, Zout.Z₁.Gs)
            cols = size(g,2)
            Zout.∂Z.Gs[i][:,1:cols] .+= â₁ .* g
        end
        Zout.∂Z.c .+= â₁ .* Zout.Z₁.c
        # Add Zin.Z₂ with a₂
        for (i, g) in zip(pre_indices₂, Zin.Z₂.Gs)
            cols = size(g,2)
            Zout.∂Z.Gs[i][:,1:cols] .+= a₂ .* g
        end
        Zout.∂Z.c .+= a₂ .* Zin.Z₂.c
        # Add Zout.Z₂ with â₂
        for (i, g) in zip(post_indices₂, Zout.Z₂.Gs)
            cols = size(g,2)
            Zout.∂Z.Gs[i][:,1:cols] .+= â₂ .* g
        end
        Zout.∂Z.c .+= â₂ .* Zout.Z₂.c
        # Add Zin.∂Z with ∂a
        for (i, g) in zip(∂pre_indices, Zin.∂Z.Gs)
            cols = size(g,2)
            Zout.∂Z.Gs[i][:,1:cols] .+= ∂a .* g
        end
        Zout.∂Z.c .+= ∂a .* Zin.∂Z.c

        # Add new generators from c
        c_non_zero_indices = findall(x->x!=0.0,c)
        generator_offset = ∂old_gen + 1
        A = Zout.∂Z.Gs[Zout.∂Z.owned_generators]
        @inbounds for i in c_non_zero_indices
            A[i, generator_offset] = c[i]
            generator_offset += 1
        end

        # Add bias
        Zout.∂Z.c .+= b
    end
end



function propagate_layer!(ZoutRef :: CachedZonotope, Ls :: DiffLayer{ReLU,ReLU,ReLU}, inputs :: Vector{DiffZonotope}; bounds_cache :: Union{Nothing,BoundsCache}=nothing)

    # Deepcopy all inputs and run new implementation
    ZoutRef_new = deepcopy(ZoutRef)
    inputs_new = [deepcopy(inp) for inp in inputs]
    bounds_cache_new = deepcopy(bounds_cache)
    propagate_layer_new!(ZoutRef_new, Ls, inputs_new; bounds_cache=bounds_cache_new)

    @assert length(inputs) == 1 "ReLU layer should have exactly one input zonotope"
    Zin = inputs[1]

    @assert !isnothing(bounds_cache)

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
        bounds_cache.initialized = true
    else
        bounds_cache.lower₁ .= max.(bounds₁[:,1], bounds_cache.lower₁)
        bounds_cache.upper₁ .= min.(bounds₁[:,2], bounds_cache.upper₁)
        bounds_cache.lower₂ .= max.(bounds₂[:,1], bounds_cache.lower₂)
        bounds_cache.upper₂ .= min.(bounds₂[:,2], bounds_cache.upper₂)
        bounds_cache.∂lower .= max.(∂bounds[:,1], bounds_cache.∂lower)
        bounds_cache.∂upper .= min.(∂bounds[:,2], bounds_cache.∂upper)
    end
    lower₁ = bounds_cache.lower₁
    upper₁ = bounds_cache.upper₁
    lower₂ = bounds_cache.lower₂
    upper₂ = bounds_cache.upper₂
    ∂lower = bounds_cache.∂lower
    ∂upper = bounds_cache.∂upper
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
    new_gen₁ = count(lower₁ .< 0.0 .&& upper₁ .> 0.0)
    new_gen₂ = count(lower₂ .< 0.0 .&& upper₂ .> 0.0)
    ∂new_gen = count(any_pos) + count(pos_any) + count(any_any)
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
    for (i, idx) in enumerate(∂pre_indices)
        ∂gen_sizes[idx] = size(Zin.∂Z.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₁)
        ∂gen_sizes[idx] = size(Zin.Z₁.Gs[i],2)
    end
    for (i, idx) in enumerate(pre_indices₂)
        ∂gen_sizes[idx] = size(Zin.Z₂.Gs[i],2)
    end
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

    L1 = get_layer1(Ls)
    L2 = get_layer2(Ls)
    # Compute Zonotopes for individual networks
    propagate_layer!(Zout.Z₁, L1, Zin.Z₁;lower=lower₁, upper=upper₁)
    propagate_layer!(Zout.Z₂, L2, Zin.Z₂;lower=lower₂, upper=upper₂)

    if USE_DIFFZONO
        selector = zeros(Bool,size(Zout.∂Z.c,1))

        Debugger.@diffrelu_case_hook zero_diff context="Zero Diff"
        Debugger.@diffrelu_case_hook neg_neg context="Neg Neg"

        post_indices₁ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₁.generator_ids)
        post_indices₂ = intersect_indices(Zout.∂Z.generator_ids, Zout.Z₂.generator_ids)
        

        # Reset everything to zero (this also covers zero_diff and neg_neg cases)
        Zout.∂Z.c .= 0.0
        for g in Zout.∂Z.Gs
            g .= 0.0
        end

        # Neg Pos:
        selector .= neg_pos
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Neg Pos"
            Zout.∂Z.c[selector] .-= (@view Zout.Z₂.c[selector])
            for (i, g) in zip(post_indices₂, Zout.Z₂.Gs)
                Zout.∂Z.Gs[i][selector,:] .-= (@view g[selector,:])
            end
        end

        # Pos Neg:
        selector .= pos_neg
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Pos Neg"
            Zout.∂Z.c[selector] .+= (@view Zout.Z₁.c[selector])
            for (i, g) in zip(post_indices₁, Zout.Z₁.Gs)
                Zout.∂Z.Gs[i][selector,:] .+= (@view g[selector,:])
            end
        end

        # Pos Pos:
        # This just copies the row from the input ∂Z
        # We also need this for Any Pos and Pos Any and thus we copy for those as well
        selector .= pos_pos .| any_pos .| pos_any
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Pos Pos+"
            Zout.∂Z.c[selector] .= (@view Zin.∂Z.c[selector])
            for (i, g) in zip(∂pre_indices, Zin.∂Z.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .= (@view g[selector,:])
            end
        end

        # Any Neg
        selector .= any_neg
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Any Neg"
            Zout.∂Z.c[selector] .+= (@view Zout.Z₁.c[selector])
            for (i, g) in zip(post_indices₁, Zout.Z₁.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .+= (@view g[selector,:])
            end
        end

        # Neg Any
        selector .= neg_any
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Neg Any"
            Zout.∂Z.c[selector] .-= (@view Zout.Z₂.c[selector])
            for (i, g) in zip(post_indices₂, Zout.Z₂.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .-= (@view g[selector,:])
            end
        end

        instable_new_generators = 0

        generator_offset = ∂old_gen + 1
        # Any Pos
        selector .= any_pos
        instable_new_generators += count(selector)
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Any Pos"
            # println("ANY_POS")
            α = -lower₁[selector]
            α ./= (upper₁[selector] .- lower₁[selector])
            Zout.∂Z.c[selector] .-= α .* (@view Zin.Z₁.c[selector])
            for (i, g) in zip(pre_indices₁, Zin.Z₁.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .-= α .* (@view g[selector,:])
            end
            @assert all(α .> 0.0)
            α .*= 0.5 .* upper₁[selector]
            Zout.∂Z.c[selector] .+= α
            for (i,row) in enumerate(findall(selector))
                Zout.∂Z.Gs[Zout.∂Z.owned_generators][row, generator_offset] = abs(α[i])
                generator_offset += 1
            end
        end

        # Pos Any
        selector .= pos_any
        instable_new_generators += count(selector)
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Pos Any"
            
            # println("POS_ANY")
            α = -lower₂[selector]
            α ./= (upper₂[selector] .- lower₂[selector])
            Zout.∂Z.c[selector] .+= α .* (@view Zin.Z₂.c[selector])
            for (i, g) in zip(pre_indices₂, Zin.Z₂.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .+= α .* (@view g[selector,:])
            end
            @assert all(α .> 0.0)
            #α .= max.(0.5.*α.*upper₂[selector],(1.0.-α).*((-).(lower₂[selector])))
            α .*= 0.5 .* upper₂[selector] #max.((-).(lower₂[selector]), upper₂[selector])
            #0.5 .* max.((-).(lower₂[selector]), upper₂[selector])
            Zout.∂Z.c[selector] .-= α
            for (i,row) in enumerate(findall(selector))
                Zout.∂Z.Gs[Zout.∂Z.owned_generators][row, generator_offset] = abs(α[i])
                generator_offset += 1
            end
        end

        # Any Any
        selector .= any_any
        instable_new_generators += count(selector)
        if any(selector)
            Debugger.@diffrelu_case_hook selector context="Any Any"
            # Find cases where ∂upper and ∂lower are 0
            α = ∂upper[selector]
            α ./= (∂upper[selector] .- ∂lower[selector])
            # TODO: what's this?
            α .= clamp.(α,0.0,1.0)
            @assert all(α .>= 0.0) && all(α .<= 1.0) "Alpha had wrong values: $(α)"
            for (i, g) in zip(∂pre_indices, Zin.∂Z.Gs)
                cols = size(g,2)
                Zout.∂Z.Gs[i][selector,1:cols] .= α .* (@view g[selector,:])
            end
            Zout.∂Z.c[selector] .= α .* (@view Zin.∂Z.c[selector])
            α .*= -min.(0.0,∂lower[selector])
            μ = 0.5 .* max.(∂upper[selector],-∂lower[selector])
            for (i,row) in enumerate(findall(selector))
                Zout.∂Z.Gs[Zout.∂Z.owned_generators][row, generator_offset] = abs(μ[i])
                generator_offset += 1
            end
            Zout.∂Z.c[selector] .+= (α .- μ)
        end

        # if FIRST_ROUND
        #     print("Instable Generators: ",instable_new_generators,"\n")
        # end
    end
    Debugger.@post_diffzono_prop_hook diff_zono_new context="Post ReLU"

    # Compare both implementations
    diff_zono_new = ZoutRef_new.zonotope
    diff_zono_old = ZoutRef.zonotope
    # Compare centers
    abstolerance = 1e-7
    reltolerance = 1e-7
    @assert all(isapprox.(diff_zono_new.Z₁.c, diff_zono_old.Z₁.c; atol=abstolerance,rtol=reltolerance)) "Z₁ centers do not match!"
    @assert all(isapprox.(diff_zono_new.Z₂.c, diff_zono_old.Z₂.c; atol=abstolerance,rtol=reltolerance)) "Z₂ centers do not match!"
    if !all(isapprox.(diff_zono_new.∂Z.c, diff_zono_old.∂Z.c; atol=abstolerance,rtol=reltolerance)) 
        println("∂Z centers do not match!")
        for i in 1:length(diff_zono_new.∂Z.c)
            if !isapprox(diff_zono_new.∂Z.c[i], diff_zono_old.∂Z.c[i]; atol=abstolerance,rtol=reltolerance)
                println(" Index $i: Deviation = $(abs(diff_zono_new.∂Z.c[i] - diff_zono_old.∂Z.c[i])) (New=$(diff_zono_new.∂Z.c[i]), Old=$(diff_zono_old.∂Z.c[i]))")
                println(" Bounds: Z₁=[$(lower₁[i]), $(upper₁[i])], Z₂=[$(lower₂[i]), $(upper₂[i])], ∂Z=[$(∂lower[i]), $(∂upper[i])]")
            end
        end
        @assert false
    end
    # Compare generators
    for (g_new, g_old) in zip(diff_zono_new.Z₁.Gs, diff_zono_old.Z₁.Gs)
        @assert all(isapprox.(g_new, g_old; atol=abstolerance, rtol=reltolerance)) "Z₁ generators do not match!"
    end
    for (g_new, g_old) in zip(diff_zono_new.Z₂.Gs, diff_zono_old.Z₂.Gs)
        @assert all(isapprox.(g_new, g_old; atol=abstolerance, rtol=reltolerance)) "Z₂ generators do not match!"
    end
    for (i,(g_new, g_old)) in enumerate(zip(diff_zono_new.∂Z.Gs, diff_zono_old.∂Z.Gs))
        if i == diff_zono_new.∂Z.owned_generators
            # Can only compare "old" generators here -- new ones may be reordered
            if isnothing(Zin.∂Z.owned_generators)
                continue
            else
                cols = size(Zin.∂Z.Gs[Zin.∂Z.owned_generators],2)
            end
        else
            cols = size(g_new,2)
        end
        if !all(isapprox.(g_new[:,1:cols], g_old[:,1:cols]; atol=abstolerance, rtol=reltolerance))
            println("∂Z generators do not match for block $i ($(diff_zono_new.∂Z.generator_ids[i]))!")
            println("Rows with differences:")
            for row in 1:size(g_new,1)
                if any(.!isapprox.(g_new[row,1:cols], g_old[row,1:cols]; atol=abstolerance, rtol=reltolerance))
                    println(" Row $row: Deviation = $(maximum(abs.(g_new[row,:] .- g_old[row,:])))")
                    println(" Bounds: Z₁=[$(lower₁[row]), $(upper₁[row])], Z₂=[$(lower₂[row]), $(upper₂[row])], ∂Z=[$(∂lower[row]), $(∂upper[row])]")
                end
            end
            @assert false
        end
    end
    # Compare bounds cache
    @assert all(isapprox.(bounds_cache_new.lower₁, bounds_cache.lower₁; atol=abstolerance, rtol=reltolerance)) "Bounds Cache lower₁ do not match!"
    @assert all(isapprox.(bounds_cache_new.upper₁, bounds_cache.upper₁; atol=abstolerance, rtol=reltolerance)) "Bounds Cache upper₁ do not match!"
    @assert all(isapprox.(bounds_cache_new.lower₂, bounds_cache.lower₂; atol=abstolerance, rtol=reltolerance)) "Bounds Cache lower₂ do not match!"
    @assert all(isapprox.(bounds_cache_new.upper₂, bounds_cache.upper₂; atol=abstolerance, rtol=reltolerance)) "Bounds Cache upper₂ do not match!"
    @assert all(isapprox.(bounds_cache_new.∂lower, bounds_cache.∂lower; atol=abstolerance, rtol=reltolerance)) "Bounds Cache ∂lower do not match!"
    @assert all(isapprox.(bounds_cache_new.∂upper, bounds_cache.∂upper; atol=abstolerance, rtol=reltolerance)) "Bounds Cache ∂upper do not match!"
end


function (N::GeminiNetwork)(Z :: DiffZonotope, P :: PropState)
    #println("Prop network")
    return foldl((Z,Ls) -> propagate_diff_layer(Ls,Z,P),zip(N.network1.layers,N.diff_network.layers,N.network2.layers),init=Z)
end



    


function deepsplit_heuristic(
    prop_state :: PropState, 
    relu_layers :: Vector{
        DiffLayer{
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S1}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S2}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S3}
        } where {S1,S2,S3}
    }, 
    relative_impact_func)

    @assert prop_state.num_instable > 0

    zonos = get_zonos_at_pos(:, prop_state)
    Zin = get_zonotope(zonos[1])
    Zout = get_zonotope(zonos[end])
    input_dim = length(prop_state.task.distance)
    bounds_caches = prop_state.task_bounds.bounds_cache
    L = length(relu_layers)

    # Zins = (Zin.Z₁, Zin.Z₂)
    # Zouts = (Zout.Z₁, Zout.Z₂)
    max_score = -Inf
    max_node = nothing
    
    for net in 1:2
        Z, Z_in = ifelse(net == 1, (Zout.Z₁, Zin.Z₁), (Zout.Z₂, Zin.Z₂))
        
        if length(Z.Gs) <= length(Z_in.Gs)
            continue
        end
        
        Z_dir = Z
        if USE_DIFF_GENERATORS_DEEPSPLIT[]
            Z_dir = Zout.∂Z
        end
        G_dir_idx = find_index_position(Z_dir.generator_ids, Z.generator_ids[Z.owned_generators])

        s = [zeros(length(get_zonotope(zonos[l.layer_idx]).∂Z.c)) for l in relu_layers]
        s_input = zeros(input_dim)
        offset_l₁ = 0

        for l₁ in L:-1:1
            idx_l₁ = relu_layers[l₁].layer_idx
            Z_l₁ = get_zonotope(zonos[idx_l₁]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
            
            if !isnothing(Z_l₁.owned_generators)
                bc₁ = bounds_caches[idx_l₁]
                crossing_l₁ = ifelse(net == 1, bc₁.crossing₁, bc₁.crossing₂)
                num_instable_l₁ = count(crossing_l₁)
                G_dir_idx = find_index_position(Z_dir.generator_ids, Z_l₁.generator_ids[Z_l₁.owned_generators])
    
                ϵ = get_generators!(Z_dir, G_dir_idx, offset_l₁, num_instable_l₁)
                # s[l₁][:] .= sum(abs, ϵ, dims=1)[:]
                s[l₁][crossing_l₁] .= sum(abs, ϵ, dims=1)[:]
    
                offset_l₂ = 0
                for l₂ in (l₁ + 1):L
                    idx_l₂ = relu_layers[l₂].layer_idx
                    Z_l₂ = get_zonotope(zonos[idx_l₂]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
                    bc_l₂ = bounds_caches[idx_l₂]
                    crossing_l₂ = ifelse(net == 1, bc_l₂.crossing₁, bc_l₂.crossing₂)
                    num_instable_l₂ = count(crossing_l₂)
                    G_idx = find_index_position(Z_l₂.generator_ids, Z_l₁.generator_ids[Z_l₁.owned_generators])
    
                    ϵ = get_generators!(Z_l₂, G_idx, offset_l₂, num_instable_l₁)
                    α = relative_impact_func(Z_l₂, ϵ, crossing_l₂)
                    s[l₁][crossing_l₁] .+ sum(α .* s[l₂], dims=1)[:]
    
                    offset_l₂ = offset_l₂ + num_instable_l₂
                end
    
                if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
                    lower = ifelse(net == 1, bc₁.lower₁, bc₁.lower₂)
                    upper = ifelse(net == 1, bc₁.upper₁, bc₁.upper₂)
                    bounds_width = upper[crossing_l₁] - lower[crossing_l₁]
                    α = abs.(Z_l₁.Gs[1][crossing_l₁, :]) ./ bounds_width
                    s_input .+= sum(α .* s[l₁][crossing_l₁] .* VeryDiff.INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
                end
    
                if !USE_VERTICAL_SPLITTING[] || l₁ < L
                    instables = findall(crossing_l₁)
                    if isempty(instables)
                        continue
                    end
    
                    n = argmax(i -> s[l₁][i], instables)
                    if s[l₁][n] > max_score
                        max_score = s[l₁][n]
                        max_node = SplitNode(net, idx_l₁, n, 0, nothing)
                    end
                end
    
                offset_l₁ = offset_l₁ + num_instable_l₁
            end
        end

        if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
            d = argmax(s_input)
            if s_input[d] > max_score
                max_score = s_input[d]
                d = prop_state.task.distance_indices[d]
                max_node = SplitNode(0, 0, d, 0, nothing)
            end
        end
    end

    # ids = Zs .|> Z -> Z.generator_ids[Z.owned_generators]
    # if USE_DIFF_GENERATORS_DEEPSPLIT[]
    #     Zs = (Zout.∂Z, Zout.∂Z)
    # end
    # indices = map((Z, id) -> find_index_position(Z.generator_ids, id), Zs, ids)
    
    # s = [zeros(2, length(get_zonotope(zonos[l.layer_idx]).∂Z.c)) for l in relu_layers]
    # s_input = zeros(2, input_dim)
    # offset_l₁ = (0, 0)
    # max_score = -Inf
    # max_node = nothing

    # for l₁ in L:-1:1
    #     l₁_idx = relu_layers[l₁].layer_idx
    #     Z_l₁ = get_zonotope(zonos[l₁_idx]) |> DZ -> (DZ.Z₁, DZ.Z₂)
    #     bc_l₁ = bounds_caches[l₁_idx]
    #     crossing_l₁ = (bc_l₁.crossing₁, bc_l₁.crossing₂)
    #     num_instable_l₁ = count.(crossing_l₁)

    #     ϵ = get_generators!.(Zs, indices, offset_l₁, num_instable_l₁)
    #     for i in 1:2
    #         s[l₁][i:i, crossing_l₁[i]] .= sum(abs, ϵ[i], dims=1)
    #     end
        
    #     offset_l₂ = (0, 0)
    #     for l₂ in (l₁ + 1):L
    #         l₂_idx = relu_layers[l₂].layer_idx
    #         Z_l₂ = get_zonotope(zonos[l₂_idx]) |> DZ -> (DZ.Z₁, DZ.Z₂)
    #         crossing_l₂ = bounds_caches[l₂_idx] |> bc -> (bc.crossing₁, bc.crossing₂)
    #         num_instable_l₂ = count.(crossing_l₂)

    #         _idxs = map((Z, id) -> find_index_position(Z.generator_ids, id), Z_l₂, ids)
    #         ϵ = get_generators!.(Z_l₂, _idxs, offset_l₂, num_instable_l₁)
    #         α = relative_impact_func.(Z_l₂, ϵ, crossing_l₂)
    #         for i in 1:2
    #             s[l₁][i:i, crossing_l₁[i]] .+= sum(abs, α[i] .* s[l₂][i, :], dims=1)
    #         end
            
    #         offset_l₂ = offset_l₂ .+ num_instable_l₂
    #     end

    #     if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
    #         lowers = (bc_l₁.lower₁, bc_l₁.lower₂)
    #         uppers = (bc_l₁.upper₁, bc_l₁.upper₂)
    #         for i in 1:2
    #             bounds_width = uppers[i][crossing_l₁[i]] - lowers[i][crossing_l₁[i]]
    #             α = abs.(Z_l₁[i].Gs[1][crossing_l₁[i], :]) ./ bounds_width
    #             s_input[i:i, :] .+= sum(α .* s[l₁][i, crossing_l₁[i]] .* VeryDiff.INDIRECT_INPUT_MULTIPLIER[], dims=1)
    #         end
    #     end
        
    #     if !USE_VERTICAL_SPLITTING[] || l₁ < L
    #         for (i, crossing) in enumerate(crossing_l₁)
    #             instables = findall(crossing)
    #             if isempty(instables)
    #                 continue
    #             end

    #             n = argmax(j -> s[l₁][i, j], instables)
    #             if s[l₁][i, n] > max_score
    #                 max_score = s[l₁][i, n]
    #                 max_node = SplitNode(i, l₁_idx, n, 0, nothing)
    #             end
    #         end
    #     end

    #     offset_l₁ = offset_l₁ .+ num_instable_l₁
    # end

    # if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
    #     d = argmax(s_input)
    #     if s_input[d] > max_score
    #         max_score = s_input[d]
    #         d = prop_state.task.distance_indices[Tuple(d)[2]]
    #         max_node = SplitNode(0, 0, d, 0, nothing)
    #     end
    # end

    @assert !isnothing(max_node)
    @assert max_score > 0.0 || USE_DIFF_GENERATORS_DEEPSPLIT[]

    return max_node
end

function get_relative_impact_func()
    if DEEPSPLIT_HEURISTIC_MODE[] == ZonoBiased
        zono_biased_relative_impact
    elseif DEEPSPLIT_HEURISTIC_MODE[] == ZonoUnbiased
        zono_unbiased_relative_impact
    elseif DEEPSPLIT_HEURISTIC_MODE[] == DeepSplitBiased
        deepsplit_biased_relative_impact
    else
        deepsplit_unbiased_relative_impact
    end
end

function zono_biased_relative_impact(Z::Zonotope, ϵ::Matrix{Float64}, crossing::BitVector)
    bounds = zono_bounds(Z)
    lower = @view bounds[:, 1]
    upper = @view bounds[:, 2]
    return ifelse.(crossing, ifelse.(ϵ .>= 0.0, ϵ ./ upper, ϵ ./ lower), 0.0)
end

function zono_unbiased_relative_impact(Z::Zonotope, ϵ::Matrix{Float64}, crossing::BitVector)
    return ifelse.(crossing, abs.(ϵ) ./ sum(sum(abs, G, dims=2) for G in Z.Gs), 0.0)
end

function deepsplit_biased_relative_impact(Z::Zonotope, ϵ::Matrix{Float64}, crossing::BitVector)
    d_lower = sum(sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2) for G in Z.Gs) + Z.c
    d_upper = sum(sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2) for G in Z.Gs) + Z.c
    return ifelse.(crossing, 2.0 * ifelse.(ϵ .>= 0, ϵ ./ d_upper, ϵ ./ d_lower), 0.0)
end

function deepsplit_unbiased_relative_impact(Z::Zonotope, ϵ::Matrix{Float64}, crossing::BitVector)
    d_lower = sum(sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2) for G in Z.Gs)
    d_upper = sum(sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2) for G in Z.Gs)
    return ifelse.(crossing, 2.0 * ifelse.(ϵ .>= 0, ϵ ./ d_upper, ϵ ./ d_lower), 0.0)
end

function get_generators!(Z::Zonotope, idx::Int, offset::Int, num::Int) :: Matrix{Float64}
    return @view Z.Gs[idx][:, end - offset - num + 1 : end - offset]
end

# function deepsplit_heuristic(
#     prop_state :: PropState, 
#     relu_layers :: Vector{
#         DiffLayer{
#             VeryDiff.VNNLib.OnnxParser.ONNXRelu{S1}, 
#             VeryDiff.VNNLib.OnnxParser.ONNXRelu{S2}, 
#             VeryDiff.VNNLib.OnnxParser.ONNXRelu{S3}
#         } where {S1,S2,S3}
#     }, 
#     relative_impact_func)

#     @assert prop_state.num_instable > 0

#     zonos = get_zonos_at_pos(:, prop_state)
#     Zin = get_zonotope(zonos[1])
#     Zout = get_zonotope(zonos[end])
#     input_dim = length(prop_state.task.distance)
#     bounds_caches = prop_state.task_bounds.bounds_cache
#     L = length(relu_layers)

#     Zs = (Zout.Z₁, Zout.Z₂)
#     in_ids = (Zin.Z₁.generator_ids, Zin.Z₂.generator_ids)
#     ids = Zs .|> Z -> Z.generator_ids[Z.owned_generators]
#     ids = ifelse.(ids .∈ in_ids, -1, ids)
#     if USE_DIFF_GENERATORS_DEEPSPLIT[]
#         Zs = (Zout.∂Z, Zout.∂Z)
#     end
#     indices = map((Z, id) -> attempt_find_index_position(Z.generator_ids, id), Zs, ids)
    
#     max_score = -Inf
#     max_node = nothing
#     for net in 1:2
#         if ids[net] == -1
#             @info "Skipped NN $(net) in the DeepSplit heuristic."
#             continue
#         end

#         s = [zeros(length(get_zonotope(zonos[l.layer_idx]).∂Z.c)) for l in relu_layers]
#         s_input = zeros(input_dim)
#         offset_l₁ = 0

#         for l₁ in L:-1:1
#             l₁_idx = relu_layers[l₁].layer_idx
#             Z_l₁ = get_zonotope(zonos[l₁_idx]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
#             bc_l₁ = bounds_caches[l₁_idx]
#             crossing_l₁ = ifelse(net == 1, bc_l₁.crossing₁, bc_l₁.crossing₂)
#             num_instable_l₁ = count(crossing_l₁)
            
#             ϵ = get_generators!(Zs[net], indices[net], offset_l₁, num_instable_l₁)
#             for i in 1:2
#                 s[l₁][crossing_l₁] .= sum(abs, ϵ, dims=1)[:]
#             end
            
#             offset_l₂ = 0
#             for l₂ in (l₁ + 1):L
#                 l₂_idx = relu_layers[l₂].layer_idx
#                 Z_l₂ = get_zonotope(zonos[l₂_idx]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
#                 crossing_l₂ = bounds_caches[l₂_idx] |> bc -> ifelse(net == 1, bc.crossing₁, bc.crossing₂)
#                 num_instable_l₂ = count(crossing_l₂)
    
#                 _idx = find_index_position(Z_l₂.generator_ids, ids[net])
#                 ϵ = get_generators!(Z_l₂, _idx, offset_l₂, num_instable_l₁)
#                 α = relative_impact_func(Z_l₂, ϵ, crossing_l₂)
#                 s[l₁][crossing_l₁] .+= sum(abs, α .* s[l₂], dims=1)[:]
                
#                 offset_l₂ = offset_l₂ .+ num_instable_l₂
#             end
    
#             if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
#                 lower = ifelse(net == 1, bc_l₁.lower₁, bc_l₁.lower₂)
#                 upper = ifelse(net == 1, bc_l₁.upper₁, bc_l₁.upper₂)
#                 bounds_width = upper[crossing_l₁] - lower[crossing_l₁]
#                 α = abs.(Z_l₁.Gs[1][crossing_l₁, :]) ./ bounds_width
#                 s_input .+= sum(α .* s[l₁][crossing_l₁] .* VeryDiff.INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
#             end           
            
#             if !USE_VERTICAL_SPLITTING[] || l₁ < L
#                 instables = findall(crossing_l₁)
#                 if isempty(instables)
#                     continue
#                 end

#                 n = argmax(i -> s[l₁][i], instables)
#                 if s[l₁][n] > max_score
#                     max_score = s[l₁][n]
#                     max_node = SplitNode(net, l₁_idx, n, 0, nothing)
#                 end
#             end
    
#             offset_l₁ = offset_l₁ .+ num_instable_l₁
#         end
    
#         if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
#             d = argmax(s_input)
#             if s_input[d] > max_score
#                 max_score = s_input[d]
#                 d = prop_state.task.distance_indices[d]
#                 max_node = SplitNode(0, 0, d, 0, nothing)
#             end
#         end
#     end

#     @assert !isnothing(max_node)
#     @assert max_score > 0.0 || USE_DIFF_GENERATORS_DEEPSPLIT[]

#     return max_node
# end

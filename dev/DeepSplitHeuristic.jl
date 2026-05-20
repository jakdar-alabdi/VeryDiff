function deepsplit_heuristic(prop_state::PropState, 
    relu_layers :: Vector{
        DiffLayer{
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S1}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S2}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S3}
        } where {S1,S2,S3}
    }, relative_impact_func)

    L = length(relu_layers)
    zonos = prop_state.zono_storage.zonotopes
    Zout = zonos[end].zonotope
    input_dim = length(prop_state.task.distance)
    bounds_cache = prop_state.task_bounds.bounds_cache

    if USE_DIFF_GENERATORS_DEEPSPLIT[]
        Z_dirs = (Zout.∂Z, Zout.∂Z)
        G_dir_idxs = (2, 3)
    else
        Z_dirs = (Zout.Z₁, Zout.Z₂)
        G_dir_idxs = (2, 2)
    end
    
    s = [zeros(2, length(zonos[l.layer_idx].zonotope.∂Z.c)) for l in relu_layers]
    s_input = zeros(2, input_dim)
    offset₁ = (0, 0)
    max_score = -Inf
    max_node = nothing

    for l₁ in L:-1:1
        diff_layer₁ = relu_layers[l₁]
        inputs₁ = get_zonos_at_pos(get_inputs(diff_layer₁), prop_state)
        Zin₁ = get_zonotope(inputs₁[1]) |> DZ -> (DZ.Z₁, DZ.Z₂)
        bc₁ = bounds_cache[diff_layer₁.layer_idx]
        crossing₁ = (bc₁.crossing₁, bc₁.crossing₂)
        num_instable₁ = count.(crossing₁)
        zero_gens = zeros.(1, num_instable₁)
        
        ϵ = ifelse.(num_instable₁ .== 0, zero_gens, get_generators!.(Z_dirs, G_dir_idxs, offset₁, num_instable₁))
        for i in 1:2
            s[l₁][i:i, crossing₁[i]] .= sum(abs, ϵ[i], dims=1)
        end
        
        offset₂ = (0, 0)
        for l₂ in (l₁ + 1):L
            diff_layer₂ = relu_layers[l₂]
            inputs₂ = get_zonos_at_pos(get_inputs(diff_layer₂), prop_state)
            Zin₂ = get_zonotope(inputs₂[1]) |> DZ -> (DZ.Z₁, DZ.Z₂)
            bc₂ = bounds_cache[diff_layer₂.layer_idx]
            crossing₂ = (bc₂.crossing₁, bc₂.crossing₂)
            num_instable₂ = count.(crossing₂)

            ϵ = ifelse.(num_instable₂ .== 0, zero_gens, get_generators!.(Zin₂, 2, offset₂, num_instable₁))
            α = relative_impact_func.(Zin₂, ϵ, crossing₂)
            for i in 1:2
                s[l₁][i:i, crossing₁[i]] .+= sum(abs, α[i] .* s[l₂][i, :], dims=1)
            end
            offset₂ = offset₂ .+ num_instable₂
        end

        if INCORPORATE_INPUT_SPLITTING[]
            lowers = (bc₁.lower₁, bc₁.lower₂)
            uppers = (bc₁.upper₁, bc₁.upper₂)
            for i in 1:2
                bounds_width = uppers[i][crossing₁[i]] - lowers[i][crossing₁[i]]
                α = abs.(Zin₁[i].Gs[1][crossing₁[i], :]) ./ bounds_width
                s_input[i:i, :] .+= INDIRECT_INPUT_MULTIPLIER[] * sum(α .* s[l₁][i, crossing₁[i]], dims=1)
            end
        end           
        
        if !USE_VERTICAL_SPLITTING[] || l₁ < L
            for (i, instable) in enumerate(findall.(crossing₁))
                if isempty(instable)
                    continue
                end
                n = argmax(j -> s[l₁][i, j], instable)
                if s[l₁][i, n] > max_score
                    max_score = s[l₁][i, n]
                    max_node = SplitNode(i, diff_layer₁.layer_idx, n, diff_layer₁)
                end
            end
        end
        offset₁ = offset₁ .+ num_instable₁
    end

    if isnothing(max_node) && USE_VERTICAL_SPLITTING[]
        net, n = Tuple(argmax(s[L]))
        if s[L][net, n] > 0.0
            max_score = s[L][net, n]
            max_node = SplitNode(net, L, n, relu_layers[L])
        end
    end

    @assert !isnothing(max_node)
    @assert max_score > 0.0 || USE_DIFF_GENERATORS_DEEPSPLIT[]

    if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
        net, n = Tuple(argmax(s_input))
        if s_input[net, n] > max_score
            d = prop_state.task.distance_indices[n]
            max_score = s_input[net, n]
            max_node = SplitNode(0, 0, d)
        end
    end

    return max_node
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
#     Zin = get_zonotope(prop_state.zono_storage.zonotopes[1])
#     Zout = get_zonotope(prop_state.zono_storage.zonotopes[end])
#     input_dim = length(prop_state.task.distance)
#     bounds_cache = prop_state.task_bounds.bounds_cache
#     L = length(relu_layers)
    
#     max_score = -Inf
#     max_node = nothing

#     for net in 1:2
#         Z = ifelse(net == 1, Zout.Z₁, Zout.Z₂)
        
#         Z_dir = Z
#         G_dir_idx = 2
#         if USE_DIFF_GENERATORS_DEEPSPLIT[]
#             Z_dir = Zout.∂Z
#             G_dir_idx = net + 1
#         end

#         s = [zeros(length(bounds_cache[l.layer_idx].lower₁)) for l in relu_layers]
#         s_input = zeros(input_dim)
#         offset₁ = 0

#         for l₁ in L:-1:1
#             diff_layer₁ = relu_layers[l₁]
#             input_positions₁ = get_inputs(diff_layer₁)
#             output_positions₁ = get_outputs(diff_layer₁)
#             inputs₁ = get_zonos_at_pos(input_positions₁, prop_state)
#             outputs₁ = get_zonos_at_pos(output_positions₁, prop_state)
#             Zin₁ = get_zonotope(inputs₁[1]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
#             Zout₁ = get_zonotope(outputs₁[1]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)

#             if !isnothing(Zout₁.owned_generators)
#                 bc₁ = bounds_cache[diff_layer₁.layer_idx]
#                 crossing₁ = ifelse(net == 1, bc₁.crossing₁, bc₁.crossing₂)
#                 num_instable₁ = count(crossing₁)
#                 # G_dir_idx = find_index_position(Z_dir.generator_ids, Zout₁.generator_ids[Zout₁.owned_generators])

#                 ϵ = get_generators!(Z_dir, G_dir_idx, offset₁, num_instable₁)
#                 s[l₁][crossing₁] .= sum(abs, ϵ, dims=1)[:]
    
#                 offset₂ = 0
#                 for l₂ in (l₁ + 1):L
#                     diff_layer₂ = relu_layers[l₂]
#                     input_positions₂ = get_inputs(diff_layer₂)
#                     output_positions₂ = get_outputs(diff_layer₂)
#                     inputs₂ = get_zonos_at_pos(input_positions₂, prop_state)
#                     outputs₂ = get_zonos_at_pos(output_positions₂, prop_state)
#                     Zin₂ = get_zonotope(inputs₂[1]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
#                     Zout₂ = get_zonotope(outputs₂[1]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)

#                     if !isnothing(Zout₂.owned_generators)
#                         bc₂ = bounds_cache[diff_layer₂.layer_idx]
#                         crossing₂ = ifelse(net == 1, bc₂.crossing₁, bc₂.crossing₂)
#                         num_instable₂ = count(crossing₂)
#                         # G_idx = find_index_position(Zin₂.generator_ids, Zout₁.generator_ids[Zout₁.owned_generators])
        
#                         ϵ = get_generators!(Zin₂, 2, offset₂, num_instable₁)
#                         α = relative_impact_func(Zin₂, ϵ, crossing₂)
#                         s[l₁][crossing₁] .+ sum(α .* s[l₂], dims=1)[:]
        
#                         offset₂ += num_instable₂
#                     end
#                 end
    
#                 if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
#                     lower, upper = ifelse(net == 1, (bc₁.lower₁, bc₁.upper₁), (bc₁.lower₂, bc₁.upper₂))
#                     bounds_width = upper[crossing₁] - lower[crossing₁]
#                     α = abs.(Zin₁.Gs[1][crossing₁, :]) ./ bounds_width
#                     s_input .+= sum(α .* s[l₁][crossing₁] .* VeryDiff.INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
#                 end
    
#                 if !USE_VERTICAL_SPLITTING[] || l₁ < L
#                     instables = findall(crossing₁)
#                     if isempty(instables)
#                         continue
#                     end
    
#                     n = argmax(i -> s[l₁][i], instables)
#                     if s[l₁][n] > max_score
#                         max_score = s[l₁][n]
#                         max_node = SplitNode(net, diff_layer₁.layer_idx, n, diff_layer₁)
#                     end
#                 end
    
#                 offset₁ += num_instable₁
#             end
#         end

#         if !isnothing(max_node) && VeryDiff.INCORPORATE_INPUT_SPLITTING[]
#             d = argmax(s_input)
#             if s_input[d] > max_score
#                 max_score = s_input[d]
#                 d = prop_state.task.distance_indices[d]
#                 max_node = SplitNode(0, 0, d)
#             end
#         end
#     end

#     @assert !isnothing(max_node)
#     @assert max_score > 0.0 || USE_DIFF_GENERATORS_DEEPSPLIT[]

#     return max_node
# end

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

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
    Zout = get_zonotope(zonos[end])
    input_dim = length(prop_state.task.distance)
    bounds_caches = prop_state.task_bounds.bounds_cache
    L = length(relu_layers)
    
    max_score = -Inf
    max_node = nothing
    
    for net in 1:2

        Z = ifelse(net == 1, Zout.Z₁, Zout.Z₂)
        if isnothing(Z.owned_generators)
            continue
        end
        
        Z_dir = Z
        if USE_DIFF_GENERATORS_DEEPSPLIT[]
            Z_dir = Zout.∂Z
        end

        s = [zeros(length(get_zonotope(zonos[l.layer_idx]).∂Z.c)) for l in relu_layers]
        s_input = zeros(input_dim)
        offset_l₁ = 0

        for l₁ in L:-1:1
            idx_l₁ = relu_layers[l₁].layer_idx
            Z_l₁ = get_zonotope(zonos[idx_l₁]) |> DZ -> ifelse(net == 1, DZ.Z₁, DZ.Z₂)
            
            if !isnothing(Z_l₁.owned_generators)
                bc_l₁ = bounds_caches[idx_l₁]
                crossing_l₁ = ifelse(net == 1, bc_l₁.crossing₁, bc_l₁.crossing₂)
                num_instable_l₁ = count(crossing_l₁)
                G_dir_idx = find_index_position(Z_dir.generator_ids, Z_l₁.generator_ids[Z_l₁.owned_generators])

                ϵ = get_generators!(Z_dir, G_dir_idx, offset_l₁, num_instable_l₁)
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
    
                    offset_l₂ += num_instable_l₂
                end
    
                if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
                    lower, upper = ifelse(net == 1, (bc_l₁.lower₁, bc_l₁.upper₁), (bc_l₁.lower₂, bc_l₁.upper₂))
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
    
                offset_l₁ += num_instable_l₁
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

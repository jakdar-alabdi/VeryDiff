function deepsplit_heuristic(prop_state::PropState, 
    relu_layers :: Vector{
        DiffLayer{
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S1}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S2}, 
            VeryDiff.VNNLib.OnnxParser.ONNXRelu{S3}
        } where {S1,S2,S3}
    }, relative_impact_func)

    @assert prop_state.num_instable > 0

    L = length(relu_layers)
    zonos = prop_state.zono_storage.zonotopes
    Zin = zonos[1].zonotope
    Zout = zonos[end].zonotope
    input_dim = length(Zin.Z₁.c)
    bounds_cache = prop_state.task_bounds.bounds_cache

    Zouts = (Zout.Z₁, Zout.Z₂)
    owned_generator_ids = Zouts .|> Z -> isnothing(Z.owned_generators) ? -1 : Z.generator_ids[Z.owned_generators]
    if USE_DIFF_GENERATORS_DEEPSPLIT[]
        Zouts = (Zout.∂Z, Zout.∂Z)
    end
    G_dir_idxs = attempt_find_index_position.(Zouts .|> Z -> Z.generator_ids, owned_generator_ids)
    
    s = [zeros(2, length(zonos[l.layer_idx].zonotope.∂Z.c)) for l in relu_layers]
    s_input = zeros(2, input_dim)

    max_score = -Inf
    max_node = nothing
    
    offset₁ = (0, 0)
    for l₁ in L:-1:1
        diff_layer₁ = relu_layers[l₁]
        inputs₁ = get_zonos_at_pos(get_inputs(diff_layer₁), prop_state)
        Zin₁ = get_zonotope(inputs₁[1]) |> DZ -> (DZ.Z₁, DZ.Z₂)
        bc₁ = bounds_cache[diff_layer₁.layer_idx]
        crossing₁ = (bc₁.crossing₁, bc₁.crossing₂)
        num_instable₁ = count.(crossing₁)
        zero_gens = zeros.(1, num_instable₁)
        
        ϵ = ifelse.(num_instable₁ .== 0, zero_gens, get_generators!.(Zouts, G_dir_idxs, offset₁, num_instable₁))
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

            ϵ = ifelse.(num_instable₂ .== 0, zero_gens, get_owned_generators!.(Zin₂, offset₂, num_instable₁))
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
        max_score = s[L][net, n]
        max_node = SplitNode(net, relu_layers[L].layer_idx, n, relu_layers[L])
    end

    @assert !isnothing(max_node)

    if VeryDiff.INCORPORATE_INPUT_SPLITTING[]
        net, n = Tuple(argmax(s_input))
        if s_input[net, n] > max_score
            max_score = s_input[net, n]
            max_node = SplitNode(0, 0, n)
        end
    end

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

function get_owned_generators!(Z::Zonotope, offset::Int, num::Int) :: Matrix{Float64}
    return get_generators!(Z, Z.owned_generators, offset, num)
end

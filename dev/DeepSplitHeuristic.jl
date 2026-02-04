function deepsplit_heuristic(Zout::DiffZonotope, prop_state::PropState, distance_indices::Vector{Int}, undetermined::BitVector)
    input_dim = size(Zout.Z₁, 2) - Zout.num_approx₁
    
    max_score = -Inf64
    max_node = nothing
    for net in 1:2
        generators = get_generators(Zout, net, undetermined)
        intermediates = prop_state.intermediate_zonos[net]
        crossings = prop_state.instable_nodes[net]
        L = size(crossings, 1)
        
        s = [zeros(size(crossings[l])) for l in 1:L]
        s_input = zeros(input_dim)
        offset₁ = 0

        for l₁ in L:-1:1
            Z₁ = intermediates[l₁]
            crossing₁ = crossings[l₁]
            num_instable₁ = count(crossing₁)
            s[l₁][crossing₁] .= sum(abs, generators(offset₁, num_instable₁), dims=1)[:]
            
            offset₂ = 0
            for l₂ in (l₁ + 1):L
                Z₂ = intermediates[l₂]
                crossing₂ = crossings[l₂]
                num_instable₂ = count(crossing₂)

                α = compute_relative_impact(Z₂, offset₂, num_instable₁, crossing₂)
                s[l₁][crossing₁] .+= sum(α .* s[l₂], dims=1)[:]

                offset₂ += num_instable₂
            end

            if DEEPSPLIT_INPUT_SPLITTING[]
                bounds = zono_bounds(Z₁)
                bounds_width = (bounds[:, 2] - bounds[:, 1])[crossing₁]
                α = abs.(Z₁.G[crossing₁, 1:input_dim]) ./ bounds_width
                s_input .+= sum(α .* s[l₁][crossing₁] .* INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
            end

            n = argmax(s[l₁])
            if s[l₁][n] > max_score
                max_score = s[l₁][n]
                max_node = SplitNode(net, l₁, n, 0)
            end

            offset₁ += num_instable₁
        end

        if DEEPSPLIT_INPUT_SPLITTING[]
            d = argmax(s_input)
            if s_input[d] > max_score
                max_score = s_input[d]
                max_node = SplitNode(0, 0, distance_indices[d], 0)
            end
        end
    end

    return max_node
end

function deepsplit_heuristic_alternative(Zout::DiffZonotope, prop_state::PropState, distance_indices::Vector{Int}, undetermined::BitVector)
    input_dim = size(Zout.Z₁, 2) - Zout.num_approx₁

    max_score = -Inf64
    max_node = nothing
    for net in 1:2
        generators = get_generators(Zout, net, undetermined)
        crossings = prop_state.instable_nodes[net]
        L = size(crossings, 1)
        relative_impactes = prop_state.relative_impactes[net]
        input_relative_impactes = prop_state.input_relative_impactes[net]

        s = [zeros(size(crossings[l])) for l in 1:L]
        s_input = zeros(input_dim)
        offset = 0

        for l₁ in L:-1:1
            crossing₁ = crossings[l₁]
            num_instable = count(crossing₁)

            s[l₁][crossing₁] .= sum(abs, generators(offset, num_instable), dims=1)[:]
            
            for i in 1:(L - l₁)
                l₂ = l₁ + i
                α = @view relative_impactes[l₁][i]
                s[l₁][crossing₁] .+= sum(α .* s[l₂], dims=1)[:]
            end

            if DEEPSPLIT_INPUT_SPLITTING[]
                α = @view input_relative_impactes[l₁][crossing₁, :]
                s_input .+= sum(α .* s[l₁][crossing₁] .* INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
            end

            n = argmax(s[l₁])
            if s[l₁][n] > max_score
                max_score = s[l₁][n]
                max_node = SplitNode(net, l₁, n, 0)
            end

            offset += num_instable
        end

        if DEEPSPLIT_INPUT_SPLITTING[]
            d = argmax(s_input)
            if s_input[d] > max_score
                max_score = s_input[d]
                max_node = SplitNode(0, 0, distance_indices[d], 0)
            end
        end
    end

    return max_node
end

function compute_relative_impact(Z::Zonotope, offset::Int64, num::Int64, crossing::BitVector)
    ϵ = @view Z.G[:, end - offset - num + 1 : end - offset]

    return begin
        if DEEPSPLIT_HEURISTIC_MODE[] == ZonoBiased
            bounds = zono_bounds(Z)
            lower = @view bounds[:, 1]
            upper = @view bounds[:, 2]
            ifelse.(crossing, ifelse.(ϵ .>= 0.0, ϵ ./ upper, ϵ ./ lower), 0.0)
        elseif DEEPSPLIT_HEURISTIC_MODE[] == ZonoUnbiased
            ifelse.(crossing, abs.(ϵ) ./ sum(abs, Z.G, dims=2), 0.0)
        else
            d_lower = sum(x -> ifelse(x < 0.0, x, 0.0), Z.G, dims=2)
            d_upper = sum(x -> ifelse(x > 0.0, x, 0.0), Z.G, dims=2)
            if DEEPSPLIT_HEURISTIC_MODE[] == DeepSplitBiased
                d_lower += Z.c
                d_upper += Z.c
            end
            ifelse.(crossing, 2.0 * ifelse.(ϵ .>= 0, ϵ ./ d_upper, ϵ ./ d_lower), 0.0)
        end
    end
end

function get_generators(Z::DiffZonotope, network::Int64, undetermined::BitVector)
    if DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS[]
        Z̃ = Z.∂Z
        net_offset = Z.∂num_approx + ifelse(network == 1, Z.num_approx₂, 0)
    else
        Z̃ = ifelse(network == 1, Z.Z₁, Z.Z₂)
        net_offset = 0
    end

    return (offset::Int64, num::Int64) -> begin
        return @view Z̃.G[undetermined, end - offset - net_offset - num + 1 : end - net_offset - offset]
        # return @view Z̃.G[:, end - offset - net_offset - num + 1 : end - net_offset - offset]
    end
end

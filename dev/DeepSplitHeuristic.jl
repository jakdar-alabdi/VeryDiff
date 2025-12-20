function deepsplit_heuristic(Zout::DiffZonotope, prop_state::PropState)
    input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
    
    max_node = SplitNode(-1, -1, -1, -Inf64)
    for net in 1:2
        generators = get_generators(Zout, net)
        intermediates = prop_state.intermediate_zonos[net]
        crossings = prop_state.instable_nodes[net]
        L = size(crossings, 1)
        
        s = [zeros(size(crossings[l])) for l in 1:L]
        s_input = zeros(input_dim)
        offset₁ = 0

        for l₁ in L:-1:1
            Z₁ = intermediates[l₁]
            bounds₁ = zono_bounds(Z₁)
            lower₁ = bounds₁[:, 1]
            upper₁ = bounds₁[:, 2]
            crossing₁ = crossings[l₁]
            num_instable₁ = count(crossing₁)
            s[l₁][crossing₁] .= sum(abs, generators(offset₁, num_instable₁), dims=1)[:]
            
            offset₂ = 0
            for l₂ in (l₁ + 1):L
                Z₂ = intermediates[l₂]
                bounds₂ = zono_bounds(Z₂)
                lower₂ = bounds₂[:, 1]
                upper₂ = bounds₂[:, 2]
                crossing₂ = crossings[l₂]
                num_instable₂ = count(crossing₂)

                ϵ = @view Z₂.G[:, end - offset₂ - num_instable₁ + 1 : end - offset₂]
                α = ifelse.(crossing₂, ifelse.(ϵ .>= 0, ϵ ./ upper₂, ϵ ./ lower₂), 0.0)
                s[l₁][crossing₁] .+= sum(α .* s[l₂], dims=1)[:]

                offset₂ += num_instable₂
            end

            if DEEPPSPLIT_INPUT_SPLITTING[]
                bounds_width = (upper₁ - lower₁)[crossing₁]
                # c = 2 .* abs.(Z₁.G[crossing₁, 1:input_dim])
                # α =  (bounds_width_input ./ 2) .* (c ./ bounds_width)
                α = abs.(Z₁.G[crossing₁, 1:input_dim]) ./ bounds_width
                s_input .+= sum(α .* s[l₁][crossing₁] .* INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
            end

            n = argmax(s[l₁])
            if s[l₁][n] > max_node.score
                max_node = SplitNode(net, l₁, n, s[l₁][n])
            end

            offset₁ += num_instable₁
        end

        if DEEPPSPLIT_INPUT_SPLITTING[]
            n = argmax(s_input)
            # if max_node.layer != 0 || s_input[n] > max_node.score
            if s_input[n] > max_node.score
                max_node = SplitNode(0, 0, n, s_input[n])
            end
        end
    end

    return max_node
end

function deepsplit_heuristic_alternative(Zout::DiffZonotope, prop_state::PropState)
    input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁

    max_node = SplitNode(0, 0, 0, -Inf64)
    for net in 1:2
        generators = get_generators(Zout, net)
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
                α = relative_impactes[l₁][i]
                s[l₁][crossing₁] .+= sum(α .* s[l₂], dims=1)[:]
            end

            if DEEPPSPLIT_INPUT_SPLITTING[]
                # α = (bounds_width_input ./ 2) .* input_relative_impactes[l₁][crossing₁, :]
                α = @view input_relative_impactes[l₁][crossing₁, :]
                s_input .+= sum(α .* s[l₁][crossing₁] .* INDIRECT_INPUT_MULTIPLIER[], dims=1)[:]
            end

            n = argmax(s[l₁])
            if s[l₁][n] > max_node.score
                max_node = SplitNode(net, l₁, n, s[l₁][n])
            end

            offset += num_instable
        end

        if DEEPPSPLIT_INPUT_SPLITTING[]
            n = argmax(s_input)
            # if max_node.layer != 0 || s_input[n] > max_node.score
            if s_input[n] > max_node.score
                max_node = SplitNode(net, 0, n, s_input[n])
            end
        end
    end

    return max_node
end

function get_generators(Z::DiffZonotope, network::Int64)
    if DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS[]
        Z̃ = Z.∂Z
        net_offset = Z.∂num_approx + (network == 1 ? Z.num_approx₂ : 0)
    else
        Z̃ = network == 1 ? Z.Z₁ : Z.Z₂
        net_offset = 0
    end

    return (offset::Int64, num::Int64) -> begin
        return @view Z̃.G[:, end - offset - net_offset - num + 1 : end - net_offset - offset]
    end
end

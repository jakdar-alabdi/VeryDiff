global DEEPPSPLIT_HUERISTIC_ALTERNATIVE = false
function deepsplit_heuristic(Zout::DiffZonotope, prop_state::PropState, undetermined::BitVector)
    zonos = [Zout.Z₁, Zout.Z₂]
    max_node = SplitNode(0, 0, 0, -Inf64)
    for net in 1:2
        Z = zonos[net]
        intermediates = prop_state.intermediate_zonotopes[net]
        crossings = prop_state.instable_nodes[net]
        L = size(crossings, 1)

        s = [zeros(size(crossings[l])) for l in 1:L]
        for l₁ in L:-1:1
            Z₁ = intermediates[l₁]
            input_dim = size(Z₁.G, 2)
            crossing = crossings[l₁]
            num_approx_range = input_dim + 1 : input_dim + count(crossing)
            s[l₁][crossing] .= sum(abs, Z.G[:, num_approx_range], dims=1)[:]
            
            for l₂ in (l₁ + 1):L
                Z₂ = intermediates[l₂]
                bounds₂ = zono_bounds(Z₂)
                lower₂ = bounds₂[:, 1]
                upper₂ = bounds₂[:, 2]
                ϵ = Z₂.G[:, num_approx_range]
                α = ifelse.(crossings[l₂], ifelse.(ϵ .>= 0, ϵ ./ upper₂, ϵ ./ lower₂), 0.0)
                s[l₁][crossing] .+= sum(α .* s[l₂], dims=1)[:]
            end
            
            n = argmax(s[l₁])
            if s[l₁][n] > max_node.score
                max_node = SplitNode(net, l₁, n, s[l₁][n])
            end
        end
    end

    return max_node
end

function deepsplit_heuristic_alternative(Zout::DiffZonotope, prop_state::PropState, undetermined::BitVector)
    zonos = [Zout.Z₁, Zout.Z₂]
    max_node = SplitNode(0, 0, 0, -Inf64)
    for net in 1:2
        Z = zonos[net]
        crossings = prop_state.instable_nodes[net]
        L = size(crossings, 1)
        relative_impactes = prop_state.relative_impactes[net]

        s = [zeros(size(crossings[l])) for l in 1:L]
        k = 0
        for l₁ in L:-1:1
            num_instable = count(crossings[l₁])
            s[l₁][crossings[l₁]] .= sum(abs, Z.G[:, end - k - num_instable + 1 : end - k], dims=1)[:]
            
            for i in 1 : (L - l₁)
                l₂ = l₁ + i
                α = relative_impactes[l₁][i]
                s[l₁][crossings[l₁]] .+= sum(α .* s[l₂], dims=1)[:]
            end

            n = argmax(s[l₁])
            if s[l₁][n] > max_node.score
                max_node = SplitNode(net, l₁, n, s[l₁][n])
            end
            k += num_instable
        end
    end

    return max_node
end


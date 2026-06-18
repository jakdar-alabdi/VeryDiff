function contract_zono!(box::InputBox, node::SplitNode, Z::Zonotope) :: Union{Nothing,InputBox}
    (;neuron, direction) = node
    gs = [G[neuron, :] for G in Z.Gs]
    c = Z.c[neuron]

    gs = -direction .* gs
    c = direction * c

    common_indices = intersect_indices(box.generator_ids, Z.generator_ids)[:]
    lowers = @view box.lowers[common_indices]
    uppers = @view box.uppers[common_indices]

    vs = [ifelse.(g .>= 0.0, l[1:length(g)], u[1:length(g)]) for (g, l, u) in zip(gs, lowers, uppers)]

    s = sum(g'v for (g, v) in zip(gs, vs))
    if s > c
        return nothing
    end

    for (g, v, l, u) in zip(gs, vs, lowers, uppers)
        for i in 1:length(g)
            if g[i] != 0
                x = (c - (s - g[i] * v[i])) / g[i]
                if g[i] > 0
                    u[i] = min(u[i], x)
                else
                    l[i] = max(l[i], x)
                end
            end
        end
    end

    return box
end

function contract_zono_all!(box::InputBox, split_nodes::Vector{SplitNode}, prop_state::PropState) :: Union{Nothing,InputBox}
    for node in split_nodes
        box = contract_zono!(box, node, get_split_node_zono(node, prop_state))
        if isnothing(box)
            break
        end
    end
    return box
end

# This function assumes that all the split nodes and the DiffZonotope correspond to the same layer.
function contract_zono_all!(box::InputBox, split_nodes::Vector{SplitNode}, DZ::DiffZonotope) :: Union{Nothing,InputBox}
    Zs = (DZ.Z₁, DZ.Z₂)
    for node in split_nodes
        box = contract_zono!(box, node, Zs[node.network])
        if isnothing(box)
            break
        end
    end
    return box
end

function transform_offset_zono!(box::InputBox, Z::Zonotope) :: Zonotope
    common_indices = intersect_indices(box.generator_ids, Z.generator_ids)
    for (i, idx) in enumerate(common_indices)
        lower = box.lowers[idx]
        upper = box.uppers[idx]
        α = (upper - lower) ./ 2
        β = (upper + lower) ./ 2
        Z.c .+= Z.Gs[i] * β
        Z.Gs[i] .*= α'
    end
    return Z
end

function transform_offset_diff_zono!(box::InputBox, Z::DiffZonotope) :: DiffZonotope
    transform_offset_zono!(box, Z.Z₁)
    transform_offset_zono!(box, Z.Z₂)
    transform_offset_zono!(box, Z.∂Z)
    return Z
end

function transform_constraints!(box::InputBox, split_nodes::Vector{SplitNode}, prop_state::PropState)
    αs = [(u - l) ./ 2 for (l, u) in zip(box.lowers, box.uppers)]
    βs = [(u + l) ./ 2 for (l, u) in zip(box.lowers, box.uppers)]
    for node in split_nodes
        Z = get_split_node_zono(node, prop_state)
        common_indices = intersect_indices(box.generator_ids, Z.generator_ids)
        for (G, idx) in zip(Z.Gs, common_indices)
            α = @view αs[idx][1:size(G, 2)]
            β = @view βs[idx][1:size(G, 2)]
            Z.c[node.neuron] += G[node.neuron, :]'β
            G[node.neuron, :] .*= α
        end
    end
end

function transform_verification_task!(box::InputBox, task::VerificationTask) :: VerificationTask
    lower = box.lowers[1]
    upper = box.uppers[1]
    α = (upper - lower) ./ 2
    β = (upper + lower) ./ 2
    task.middle[task.distance_indices] .+= task.distance .* β
    task.distance .*= α
    return task
end

function contract_to_verification_task!(box::InputBox, node::SplitNode, Z::Zonotope, task::VerificationTask) :: Union{Nothing,VerificationTask}
    box = contract_zono!(box, node, Z)
    if !isnothing(box)
        if !is_unit_hypercube(box)
            return transform_verification_task!(box, task)
        end
        return task
    end
    return nothing
end

function offset_zono_bounds(box::InputBox, Z::Zonotope) :: Matrix{Float64}
    common_indices = intersect_indices(box.generator_ids, Z.generator_ids)
    bounds = zeros(2, length(Z.c))
    for (i, idx) in enumerate(common_indices)
        lower = @view lowers[idx][1:size(Z.Gs[i], 2)]
        upper = @view uppers[idx][1:size(Z.Gs[i], 2)]
        row_bounds = g -> ifelse.(g .>= 0, g .* [lower upper], g .* [upper lower], dims=1)
        bounds .+= mapreduce(row_bounds, vcat, eachrow(Z.Gs[i]))
    end
    bounds .+= Z.c
    return bounds
end

function geometric_distance(box::InputBox, neuron::Int, Z::Zonotope) :: Float64
    gs = [@view G[neuron, :] for G in Z.Gs]
    common_indices = intersect_indices(box.generator_ids, Z.generator_ids)[:]
    lowers = @view box.lowers[common_indices]
    uppers = @view box.uppers[common_indices]
    centers = (lowers .+ uppers) ./ 2
    a = Z.c[neuron] + sum(g'x[1:length(g)] for (g, x) in zip(gs, centers))
    b = sum(g'g for g in gs)
    return abs(a) / sqrt(b)
end

function geometric_distance(box::InputBox, node::SplitNode, prop_state::PropState) :: Float64
    return geometric_distance(box, node.neuron, get_split_node_zono(node, prop_state))
end

function geometric_distance0(node::SplitNode, Z::Zonotope) :: Float64
    gs = [@view G[node.neuron, :] for G in Z.Gs]
    return abs(Z.c[node.neuron]) / sqrt(sum(g'g for g in gs))
end

function sort_split_nodes!(split_nodes::Vector{SplitNode}, prop_state::PropState) :: Vector{SplitNode}
    sort!(split_nodes, by=node -> geometric_distance0(node, get_split_node_zono(node, prop_state)))
end

# This function assumes that all the split nodes and the DiffZonotope correspond to the same layer.
function sort_split_nodes!(split_nodes::Vector{SplitNode}, Z::DiffZonotope) :: Vector{SplitNode}
    sort!(split_nodes, by=node -> geometric_distance0(node, ifelse(node.network == 1, Z.Z₁, Z.Z₂)))
end

function is_unit_hypercube(box::InputBox) :: Bool
    return all(l -> all(x -> isone(-x), l), box.lowers) && all(u -> all(x -> isone(x), u), box.uppers)
end

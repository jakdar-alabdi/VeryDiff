function deepsplit_lp_search_epsilon(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    return (N₁::Network, N₂::Network, Zin::Zonotope) -> begin
        N = GeminiNetwork(N₁, N₂)
        prop_state = PropState(true)

        ∂Z = Zonotope(zeros(Float64, size(Zin.G)), zeros(size(Zin.c)), nothing)
        ∂Zin = DiffZonotope(Zin, deepcopy(Zin), ∂Z, 0, 0, 0)

        splits = Deque{Vector{SplitNode}}()
        push!(splits, SplitNode[])

        neuron_splits = 0
        
        while !isempty(splits)
            split_nodes = popfirst!(splits)
            
            split_candidates = SplitCandidate[]
            Zout = N(∂Zin, prop_state; split_nodes=split_nodes, split_candidates=split_candidates)

            prop_satisfied, cex, _, _, _ = property_check(N₁, N₂, ∂Zin, Zout, nothing)

            if !prop_satisfied
                if !isnothing(cex)
                    return UNSAFE, cex
                end

                # Initialize the LP solver
                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                set_time_limit_sec(model, 10)
                
                # Add variables and input and output constraints
                var_num = size(Zout.∂Z.G, 2)
                @variable(model, -1.0 <= x[1:var_num] <= 1.0)
                
                @constraint(model, copy(Zout.∂Z.G) * x + copy(Zout.∂Z.c) .>= epsilon)
                
                # Add split constraints
                for split_node in split_nodes
                    (;g, c) = prop_state.split_generators[to_dict_key(split_node)]
                    # append!(g, zeros(var_num - size(g, 1)))
                    @constraint(model, split_node.direction * (g' * x[1:size(g, 1)] + c) >= 0.0)
                end

                @objective(model, Max, 0)
                optimize!(model)

                if is_solved_and_feasible(model)
                    cex = value.(x)[1:size(∂Zin.Z₁.G, 2)]
                    sample_distance = get_sample_distance(N₁, N₂, cex)
                    if sample_distance > epsilon
                        return UNSAFE, cex
                    end
                end

                if isempty(split_candidates)
                    return UNKNOWN
                end

                if termination_status(model) != MOI.INFEASIBLE
                    split₁, split₂ = split_neuron(split_candidates[1].node, split_nodes)
                    push!(splits, split₁, split₂)

                    neuron_splits += 1
                end
            end
        end
        println("Neuron splits: $neuron_splits")
        return SAFE
    end
end

function split_neuron(node :: SplitNode, prev_split :: Vector{SplitNode})
    split₁ = prev_split
    split₂ = deepcopy(prev_split)

    push!(split₁, SplitNode(node.network, node.layer, node.neuron, -1))
    push!(split₂, SplitNode(node.network, node.layer, node.neuron, 1))

    return split₁, split₂
end

function to_dict_key(node :: SplitNode)
    (;network, layer, neuron) = node
    return "$network,$layer,$neuron"
end
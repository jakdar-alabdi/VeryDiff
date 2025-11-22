function deepsplit_lp_search_epsilon(N₁::Network, N₂::Network, bounds, epsilon::Float64; timeout=Inf)
    lower = @view bounds[:, 1]
    upper = @view bounds[:, 2]
    Zin = to_zonotope(lower, upper)

    status, cex = deepsplit_lp_search_epsilon(epsilon)(N₁, N₂, Zin)

    show(VeryDiff.to)
    println()
    println(status)
    println(cex)

    try
        if !isnothing(cex)
            distance = get_sample_distance(N₁, N₂, cex)
            @assert all(lower .<= (Zin.G * cex + Zin.c) .<= upper) "The found counterexample $cex is not within specified bounds"
            @assert distance > epsilon "The found counterexample $cex with sample distance $distance is spurious"
            println("Counterexample: $cex with sample distance: $distance")
        end 
    catch e
        println(e)
    end

    return status
end

function deepsplit_lp_search_epsilon(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    return (N₁::Network, N₂::Network, Zin::Zonotope) -> begin
        reset_timer!(to)
        @timeit to "Initialize" begin
            VeryDiff.NEW_HEURISTIC = false
            N = GeminiNetwork(N₁, N₂)
            prop_state = PropState(true)

            generator = (node::SplitNode) -> prop_state.split_generators[to_dict_key(node)]

            input_dim = size(Zin.G, 2)

            ∂Z = Zonotope(zeros(Float64, size(Zin.G)), zeros(size(Zin.c)), nothing)
            ∂Zin = DiffZonotope(Zin, deepcopy(Zin), ∂Z, 0, 0, 0)

            splits = Deque{Tuple{BitMatrix, Vector{SplitNode}}}()
            push!(splits, (hcat(falses(0), falses(0)), SplitNode[]))
        end
        
        @timeit to "Search" begin
            while !isempty(splits)
                mask, split_nodes = popfirst!(splits)
                
                split_candidates = SplitCandidate[]
                @timeit to "Zonotope Propagate" begin
                Zout = N(∂Zin, prop_state; split_nodes=split_nodes, split_candidates=split_candidates)
                end

                @timeit to "Property Check" begin
                    prop_satisfied, cex, _, _, _ = property_check(N₁, N₂, ∂Zin, Zout, nothing)
                end

                if !prop_satisfied
                    if !isnothing(cex)
                        return UNSAFE, cex[1]
                    end

                    @timeit to "Initialize LP-solver" begin
                        # Initialize the LP solver
                        model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                        set_time_limit_sec(model, 10)
                        
                        # Add variables and input and output constraints
                        var_num = size(Zout.∂Z.G, 2)
                        @variable(model, -1.0 <= x[1:var_num] <= 1.0)

                        # Add split constraints
                        Gₛ = zeros(Float64, size(split_nodes, 1), var_num)
                        cₛ = zeros(Float64, size(split_nodes))
                        dₛ = zeros(Float64, size(split_nodes))
                        for (i, split_node) in enumerate(split_nodes)
                            (;g, c) = generator(split_node)
                            Gₛ[i, 1:size(g, 1)] = g
                            cₛ[i] = c
                            dₛ[i] = split_node.direction
                        end
                        @constraint(model, dₛ .* (Gₛ * x + cₛ) .>= 0.0)
                    end

                    @timeit to "Solve LP" begin
                        bounds = zono_bounds(Zout.∂Z)
                        # Compute all output dimensions that still need to be proven
                        mask = hcat(bounds[:, 1] .< -epsilon, bounds[:, 2] .> epsilon) .&& (isempty(mask) ? true : mask)

                        # For each unproven output dimension we solve a LP for corresponding lower and upper bound
                        # for i in sort!((1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]], by=k->sum(abs.(bounds[k, :])))
                        for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                            for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]

                                @objective(model, Max, σ * (Zout.∂Z.G[i, :]' * x + Zout.∂Z.c[i]))
                                optimize!(model)

                                @timeit to "Check LP value" begin
                                    if is_solved_and_feasible(model)
                                        cex = Zin.G * value.(x)[1:input_dim] + Zin.c
                                        sample_distance = get_sample_distance(N₁, N₂, cex)
                                        if sample_distance > epsilon
                                            @timeit to "LP Solution" begin
                                                return UNSAFE, cex
                                            end
                                        end
                                    end
                                end

                                mask[i, j] = termination_status(model) != MOI.INFEASIBLE
                            end
                        end
                    end

                    if !any(mask)
                        continue
                    end

                    if isempty(split_candidates)
                        return UNKNOWN, nothing
                    end

                    @timeit to "Split Neuron" begin
                        split₁, split₂ = split_neuron(split_candidates[1].node, (mask, split_nodes))
                        push!(splits, split₁, split₂)
                    end
                end
            end
        end
        return SAFE, nothing
    end
end

function split_neuron(node :: SplitNode, prev_split :: Tuple{BitMatrix, Vector{SplitNode}})
    mask₁, split₁ = prev_split
    mask₂, split₂ = deepcopy(prev_split)

    push!(split₁, (SplitNode(node.network, node.layer, node.neuron, -1)))
    push!(split₂, (SplitNode(node.network, node.layer, node.neuron, 1)))

    return (mask₁, split₁), (mask₂, split₂)
end

function to_dict_key(node :: SplitNode)
    (;network, layer, neuron) = node
    return "$network,$layer,$neuron"
end
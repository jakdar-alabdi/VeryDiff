function deepsplit_lp_search_epsilon(N₁::Network, N₂::Network, Zin::Zonotope, ϵ::Float64)
    return deepsplit_lp_search_epsilon(N₁, N₂, zono_bounds(Zin), ϵ)
end

function deepsplit_lp_search_epsilon(N₁::Network, N₂::Network, bounds, epsilon::Float64; timeout=Inf64)
    try
        reset_timer!(to)
        @timeit to "Initialize" begin
            VeryDiff.NEW_HEURISTIC = false

            lower = @view bounds[:, 1]
            upper = @view bounds[:, 2]
        
            mid = (upper .+ lower) ./ 2
            distance = mid .- lower
            non_zero_indices = findall((!).(iszero.(distance)))
            distance = distance[non_zero_indices]
        
            input_dim = length(lower)
            ∂Z = Zonotope(Matrix(0.0I, input_dim, size(non_zero_indices, 1)), zeros(Float64, input_dim), nothing)
            initial_task = VerificationTask(mid, distance, non_zero_indices, ∂Z, nothing, Inf64, Branch(trues(1, 2)))

            split_heuristic = deepsplit_heuristic
            if DEEPPSPLIT_HUERISTIC_ALTERNATIVE[]
                split_heuristic = deepsplit_heuristic_alternative
            end
            
            N = GeminiNetwork(N₁, N₂)
        end
    
        @timeit to "Verify" begin
            status, cex, δ_bounds = deepsplit_lp_search_epsilon(epsilon)(N, N₁, N₂, initial_task, split_heuristic; timeout=timeout)
            if !isnothing(cex)
                println("\nFound counterexample: $cex")
            end
            println("\nInitial δ-bound: $(δ_bounds[1])")
            println("Final δ-bound: $(δ_bounds[2])")
            println(status)
        end
        show(VeryDiff.to)
        
        return status, δ_bounds
    catch e
        println("Caught an exception:")
        showerror(stderr, e, catch_backtrace())
        return UNKNOWN, (Inf64, Inf64)
    end
end

function deepsplit_lp_search_epsilon(ϵ::Float64)
    property_check = get_epsilon_property(ϵ)
    
    return (N::GeminiNetwork, N₁::Network, N₂::Network, initial_task::VerificationTask, split_heuristic; timeout=Inf64) -> begin
        start_time = time_ns()
        first_task = true
        initial_δ_bound = Inf64
        final_δ_bound = Inf64

        queue = Queue()
        push!(queue, (1.0, initial_task))
        
        @timeit to "Zonotope Loop" begin
            while !isempty(queue)
                work_share, task = pop!(queue)
                final_δ_bound = task.distance_bound
                # println(task.distance_bound)
                
                @timeit to "Zonotope Propagate" begin
                    prop_state = PropState(true)
                    prop_state.split_nodes = task.branch.split_nodes
                    Zin = to_diff_zono(task)
                    input_dim = size(Zin.Z₁.G, 2)
                    Zout = N(Zin, prop_state)
                end

                @timeit to "Property Check" begin
                    prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing)

                    if first_task
                        println("Zono Bounds:")
                        bounds = zono_bounds(Zout.∂Z)
                        println(bounds[:, 1])
                        println(bounds[:, 2])
                        initial_δ_bound = distance_bound
                        final_δ_bound = distance_bound
                        first_task = false
                    end
                end

                if !prop_satisfied
                    if !isnothing(cex)
                        return UNSAFE, cex, (initial_δ_bound, final_δ_bound)
                    end

                    if !isempty(prop_state.split_nodes)
                        @timeit to "Initialize LP-solver" begin
                            # Initialize the LP solver
                            model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                            set_time_limit_sec(model, 10)
                            
                            # Add input variables
                            var_num = size(Zout.∂Z.G, 2)
                            @variable(model, -1.0 <= x[1:var_num] <= 1.0)
    
                            # Add split constraints
                            for (;g, c, direction) in prop_state.split_nodes
                                @constraint(model, direction * (g' * x[1:size(g, 1)] + c) >= 0.0)
                            end
                        end
    
                        @timeit to "Solve LP" begin
                            distance_bound = 0.0
                            bounds = zono_bounds(Zout.∂Z)
                            # Compute all output dimensions that still need to be proven
                            # mask = hcat(bounds[:, 1] .< -ϵ, bounds[:, 2] .> ϵ) .&& (isempty(task.branch.mask) ? true : task.branch.mask)
                            # mask = abs.(bounds) .> ϵ .&& (isempty(task.branch.undetermined) || task.branch.undetermined)
                            mask = abs.(bounds) .> ϵ .&& task.branch.undetermined
    
                            # For each unproven output dimension we solve a LP for corresponding lower and upper bound
                            for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                                for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]
    
                                    @objective(model, Max, σ * (Zout.∂Z.G[i, :]' * x + Zout.∂Z.c[i]))
                                    optimize!(model)
    
                                    if is_solved_and_feasible(model)
                                        cex = Zin.Z₁.G * value.(x)[1:input_dim] + Zin.Z₁.c
                                        sample_distance = get_sample_distance(N₁, N₂, cex)
                                        if sample_distance > ϵ
                                            @timeit to "LP Solution" begin
                                                return UNSAFE, (cex, (N₁(cex), N₂(cex), sample_distance)), (initial_δ_bound, final_δ_bound)
                                            end
                                        end
                                    end
    
                                    if has_values(model)
                                        δ = abs(objective_value(model))
                                        mask[i, j] &= δ > ϵ
                                        distance_bound = max(distance_bound, δ)
                                    end
    
                                    mask[i, j] &= termination_status(model) != MOI.INFEASIBLE
                                end
                            end
                            task.branch.undetermined = mask
                        end
                    end

                    if any(task.branch.undetermined)
                        @timeit to "Compute Split" begin
                            split_candidate = split_heuristic(Zout, prop_state, task.distance_indices)
                            distance_bound = min(distance_bound, task.distance_bound)
                            task₁, task₂ = split_node(split_candidate, task, work_share, distance_bound)
                            push!(queue, task₁)
                            push!(queue, task₂)
                        end

                        if (time_ns() - start_time) / 1.0e9 > timeout
                            _, next_task = first(queue)
                            final_δ_bound = next_task.distance_bound
                            println("\nTIMEOUT REACHED")
                            return UNKNOWN, nothing, (initial_δ_bound, final_δ_bound)
                        end
                    end
                end
            end
        end
        return SAFE, nothing, (initial_δ_bound, final_δ_bound)
    end
end

function split_node(node::SplitNode, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    if node.layer == 0
        @timeit to "Split Input" begin
            return split_zono(node.neuron, task, work_share, nothing, distance_bound)
        end
    end
    @timeit to "Split Neuron" begin
        return split_neuron(node, task, work_share, distance_bound)
    end
end

function split_neuron(node::SplitNode, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    branch₁ = task.branch
    branch₂ = deepcopy(task.branch)

    push!(branch₁.split_nodes, SplitNode(node.network, node.layer, node.neuron, node.score, -1))
    push!(branch₂.split_nodes, SplitNode(node.network, node.layer, node.neuron, node.score, 1))

    task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, nothing, distance_bound, branch₁)
    task₂ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, nothing, distance_bound, branch₂)

    return (work_share / 2.0, task₁), (work_share / 2.0, task₂)
end

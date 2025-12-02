LP_BOUND_TESTING = true

function deepsplit_lp_search_epsilon(N₁::Network, N₂::Network, Zin :: Zonotope, ϵ :: Float64)
    return deepsplit_lp_search_epsilon(N₁, N₂, zono_bounds(Zin), ϵ)
end

function deepsplit_lp_search_epsilon(N₁::Network, N₂::Network, bounds, epsilon::Float64; timeout=Inf)
    lower = @view bounds[:, 1]
    upper = @view bounds[:, 2]
    Zin = to_zonotope(lower, upper)

    mid = (upper .+ lower) ./ 2
    distance = mid .- lower
    non_zero_indices = findall((!).(iszero.(distance)))
    distance = distance[non_zero_indices]

    input_dim = length(lower)
    ∂Z = Zonotope(Matrix(0.0I, input_dim, size(non_zero_indices, 1)), zeros(Float64, input_dim), nothing)
    verification_task = VerificationTask(mid, distance, non_zero_indices, ∂Z, nothing, epsilon, Branch())

    status, cex = deepsplit_lp_search_epsilon(epsilon)(N₁, N₂, verification_task, input_dim)

    println(status)
    show(VeryDiff.to)
    println()

    try
        if !isnothing(cex)
            distance = get_sample_distance(N₁, N₂, cex)
            @assert all(lower .<= (Zin.G * cex + Zin.c) .<= upper) "The found counterexample $cex with sample distance $distance is not within specified bounds $bounds."
            @assert distance > epsilon "The found counterexample $cex with sample distance $distance is spurious."
            println("Counterexample: $cex with sample distance: $distance.")
        end 
    catch e
        println(e)
    end

    return status
end

function deepsplit_lp_search_epsilon(ϵ::Float64)
    property_check = get_epsilon_property(ϵ)

    return (N₁::Network, N₂::Network, task::VerificationTask, input_dim::Int64) -> begin
        reset_timer!(to)
        @timeit to "Initialize" begin
            VeryDiff.NEW_HEURISTIC = false
            N = GeminiNetwork(N₁, N₂)
            
            queue = Queue()
            push!(queue, (1.0, task))
        end
        
        @timeit to "Search" begin
            while !isempty(queue)
                work_share, task = pop!(queue)
                # println(task.distance_bound)
                
                @timeit to "Zonotope Propagate" begin
                    prop_state = PropState(true)
                    prop_state.split_nodes = task.branch.split_nodes
                    Zin = to_diff_zono(task)
                    Zout = N(Zin, prop_state)
                end

                @timeit to "Property Check" begin
                    prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing)
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
                        for (;g, c, direction) in prop_state.split_nodes
                            @constraint(model, direction * (g' * x[1:size(g, 1)] + c) >= 0.0)
                        end
                    end

                    @timeit to "Solve LP" begin
                        distance_bound = 0.0
                        bounds = zono_bounds(Zout.∂Z)
                        # Compute all output dimensions that still need to be proven
                        mask = hcat(bounds[:, 1] .< -ϵ, bounds[:, 2] .> ϵ) .&& (isempty(task.branch.mask) ? true : task.branch.mask)

                        # For each unproven output dimension we solve a LP for corresponding lower and upper bound
                        for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                            for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]

                                @objective(model, Max, σ * (Zout.∂Z.G[i, :]' * x + Zout.∂Z.c[i]))
                                optimize!(model)

                                if is_solved_and_feasible(model)
                                    cex = Zin.Z₁.G * value.(x)[1:size(Zin.Z₁.G, 1)] + Zin.Z₁.c
                                    sample_distance = get_sample_distance(N₁, N₂, cex)
                                    if sample_distance > ϵ
                                        @timeit to "LP Solution" begin
                                            return UNSAFE, cex
                                        end
                                    end
                                end

                                if has_values(model)
                                    δ = objective_value(model)
                                    mask[i, j] &= δ > ϵ
                                    distance_bound = max(distance_bound, δ)
                                end

                                mask[i, j] &= termination_status(model) != MOI.INFEASIBLE
                            end
                        end
                        task.branch.mask = mask
                    end

                    if any(mask)
                        # Tests empirically whether the bounds computed by LP are valid
                        global LP_BOUND_TESTING
                        if LP_BOUND_TESTING
                            @timeit to "Random Test" begin
                                δ = distance_bound
                                if !isempty(queue)
                                    _, next_task = first(queue)
                                    δ = max(next_task.distance_bound, distance_bound)
                                end
                                for _ in 1:1000
                                    x = Zin.Z₁.G * rand(Float64, input_dim) + Zin.Z₁.c
                                    sample_distance = get_sample_distance(N₁, N₂, x)
                                    @assert sample_distance <= δ "Input x = $x has a difference distance of $sample_distance which is not within $δ-bounds, seems like a bug."
                                end
                            end
                        end

                        @timeit to "Split Neuron" begin
                            # net, col, score = 1, 0, 0.0
                            # for i in 1:(size(Zout.Z₁.G, 2) - input_dim)
                            #     err = sum(abs.(Zout.Z₁.G[:, i + input_dim]))
                            #     if err > score
                            #         col, score = i, err
                            #     end
                            # end
                            # for i in 1:(size(Zout.Z₂.G, 2) - input_dim)
                            #     err = sum(abs.(Zout.Z₂.G[:, i + input_dim]))
                            #     if err > score
                            #         net, col, score = 2, i, err
                            #     end
                            # end
                            # split_candidate = prop_state.instable_nodes[net][col]
                            task₁, task₂ = split_neuron(prop_state.split_candidate, task, distance_bound)
                            push!(queue, (work_share / 2.0, task₁))
                            push!(queue, (work_share / 2.0, task₂))
                        end
                    end
                end
            end
        end
        return SAFE, nothing
    end
end

function split_neuron(node :: SplitNode, task :: VerificationTask, distance_bound :: Float64)
    branch₁ = task.branch
    branch₂ = deepcopy(task.branch)

    push!(branch₁.split_nodes, (SplitNode(node.network, node.layer, node.neuron, -1)))
    push!(branch₂.split_nodes, (SplitNode(node.network, node.layer, node.neuron, 1)))

    task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, nothing, distance_bound, branch₁)
    task₂ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, nothing, distance_bound, branch₂)

    return task₁, task₂
end

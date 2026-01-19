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
            if DEEPSPLIT_HUERISTIC_ALTERNATIVE[]
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
                
                if !check_resources(start_time, timeout, 1)
                    return UNKNOWN, nothing, (initial_δ_bound, final_δ_bound)
                end

                @timeit to "Zonotope Propagate" begin
                    prop_state = PropState()
                    Zin = to_diff_zono(task)
                    prop_state.split_nodes = task.branch.split_nodes
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
                    
                    input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
                    var_num = size(Zout.∂Z.G, 2)
                    input_bounds = [-ones(var_num) ones(var_num)]

                    if !isempty(prop_state.split_nodes)
                        bounds = zono_bounds(Zout.∂Z)
                        # Compute all output dimensions that still need to be proven
                        mask = abs.(bounds) .> ϵ .&& task.branch.undetermined

                        for node in prop_state.split_nodes
                            offset = ifelse(node.network == 1, 0, Zout.num_approx₁)
                            node.g = algin_vector(node.g, var_num, input_dim, offset)
                        end
                        
                        if NEURON_SPLITTING_APPROACH[] == LP

                            @timeit to "Initialize LP-solver" begin
                                # Initialize the LP solver
                                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                                set_time_limit_sec(model, 10)
                                
                                # Add input variables
                                @variable(model, -1 <= x[i=1:var_num] <= 1)
        
                                # Add split constraints
                                for (;g, c, direction) in prop_state.split_nodes
                                    @constraint(model, direction * (g'x + c) >= 0.0)
                                end
                            end
        
                            @timeit to "Solve LP" begin
                                distance_bound = 0.0
    
                                # For each unproven output dimension we solve an LP for corresponding lower and upper bound
                                for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                                    for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]
        
                                        @objective(model, Max, σ * (Zout.∂Z.G[i, :]' * x + Zout.∂Z.c[i]))
                                        optimize!(model)
        
                                        if is_solved_and_feasible(model)
                                            cex_input = Zin.Z₁.G * value.(x)[1:input_dim] + Zin.Z₁.c
                                            sample_distance = get_sample_distance(N₁, N₂, cex_input)
                                            if sample_distance > ϵ
                                                @timeit to "LP Solution" begin
                                                    return UNSAFE, (cex_input, (N₁(cex_input), N₂(cex_input), sample_distance)), (initial_δ_bound, final_δ_bound)
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
                            end
                            
                        elseif NEURON_SPLITTING_APPROACH[] == ZonoContraction
                            @timeit to "Contract Zono" begin
                                    
                                centroid = (input_bounds[:, 1] + input_bounds[:, 2]) ./ 2.0
                                sort_prop = node -> geometric_distance(centroid, node.g, node.c)
                                sorted_consts = sort(prop_state.split_nodes, by=sort_prop)
                                
                                empty_intersection = false
                                for (;g, c, direction) in sorted_consts
                                    input_bounds = contract_zono(input_bounds, g, c, direction)
                                    if isnothing(input_bounds)
                                        empty_intersection = true
                                        break
                                    end
                                end
                                
                                if empty_intersection
                                    continue
                                end

                                Zin.Z₁ = transform_offset_zono(input_bounds, Zin.Z₁)
                                Zin.Z₂.G .= Zin.Z₁.G
                                Zin.Z₂.c .= Zin.Z₁.c
                                Zout.∂Z = transform_offset_zono(input_bounds, Zout.∂Z)
                                prop_satisfied, cex, _, _, _ = property_check(N₁, N₂, Zin, Zout, nothing)

                                if !prop_satisfied
                                    if !isnothing(cex)
                                        @timeit to "ContractZono Solution" begin
                                            return UNSAFE, cex, (initial_δ_bound, final_δ_bound)
                                        end
                                    end

                                    bounds = zono_bounds(Zout.∂Z)
                                    distance_bound = maximum(abs, bounds)
                                    mask .&= abs.(bounds) .> ϵ
                                else
                                    continue
                                end
                            end
                        else
                            # TODO implement vertical splitting
                            throw(ErrorException("Vertical splitting not implemented yet :("))
                        end

                        task.branch.undetermined = mask
                    end

                    if any(task.branch.undetermined)
                        @timeit to "Compute Split" begin
                            split_candidate = split_heuristic(Zout, prop_state, task.distance_indices)
                            distance_bound = min(distance_bound, task.distance_bound)
                            task₁, task₂ = split_node(split_candidate, input_bounds, Zin, task, work_share, distance_bound)
                            if !isnothing(task₁[2])
                                push!(queue, task₁)
                            end
                            if !isnothing(task₂[2])
                                push!(queue, task₂)
                            end
                        end
                    end
                end
            end
        end
        return SAFE, nothing, (initial_δ_bound, final_δ_bound)
    end
end

function split_node(node::SplitNode, input_bounds::Matrix{Float64}, Zin::DiffZonotope, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    if node.layer == 0
        @timeit to "Split Input" begin
            return split_zono(node.neuron, task, work_share, nothing, distance_bound)
        end
    end
    @timeit to "Split Neuron" begin
        return split_neuron(node, input_bounds, Zin, task, work_share, distance_bound)
    end
end

function split_neuron(node::SplitNode, input_bounds::Matrix{Float64}, Zin::DiffZonotope, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    
    (;network, layer, neuron, score, g, c) = node
    direction₁, direction₂ = -1, 1 # inactive, active

    branch₁, branch₂ = task.branch, deepcopy(task.branch)
    push!(branch₁.split_nodes, SplitNode(network, layer, neuron, score, direction₁, g, c))
    push!(branch₂.split_nodes, SplitNode(network, layer, neuron, score, direction₂, g, c))
    
    task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, branch₁)
    task₂ = VerificationTask(deepcopy(task.middle), deepcopy(task.distance), task.distance_indices, deepcopy(task.∂Z), deepcopy(task.verification_status), distance_bound, branch₂)

    if NEURON_SPLITTING_APPROACH[] == ZonoContraction
        input_bounds₁, input_bounds₂ = input_bounds, deepcopy(input_bounds)

        task₁ = contract_to_verification_task(input_bounds₁, g, c, direction₁, Zin.Z₁, task₁)
        task₂ = contract_to_verification_task(input_bounds₂, g, c, direction₂, Zin.Z₂, task₂)
    end

    return (work_share / 2.0, task₁), (work_share / 2.0, task₂)
end

function algin_vector(g::Vector{Float64}, len::Int64, offset₁::Int64, offset₂::Int64)
    ĝ = zeros(len)
    ĝ[1:offset₁] .= g[1:offset₁]
    ĝ[(offset₁ + offset₂ + 1) : (offset₂ + size(g, 1))] .= g[(offset₁ + 1) : end]
    return ĝ
end

function check_resources(start_time::UInt64, timeout::Int64, mem_min::Int64)
    timeout_reached = (time_ns() - start_time) / 1.0e9 > timeout
    memout_reached = (Sys.free_memory() / (1 << 30)) < mem_min
    
    if timeout_reached
        println("\nTIMEOUT REACHED")
    end

    if memout_reached
        println("\nMEMOUT REACHED")
    end
    
    return !timeout_reached && !memout_reached
end

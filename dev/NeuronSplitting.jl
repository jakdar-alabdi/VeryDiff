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
            initial_task = VerificationTask(mid, distance, non_zero_indices, ∂Z, nothing, Inf64, Branch())

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

        use_lp = NEURON_SPLITTING_APPROACH[] == LP
        use_zono_contract = NEURON_SPLITTING_APPROACH[] == ZonoContraction
        inter_contract = use_zono_contract && ZONO_CONTRACT_MODE[] == ZonoContractInter
        pre_contract = use_zono_contract && (ZONO_CONTRACT_MODE[] == ZonoContract || ZONO_CONTRACT_MODE[] == ZonoContractPre)
        post_contract = use_zono_contract && (ZONO_CONTRACT_MODE[] == ZonoContract || ZONO_CONTRACT_MODE[] == ZonoContractPost)

        queue = Queue()
        push!(queue, (1.0, initial_task))
        
        @timeit to "Zonotope Loop" begin
            try
                while !isempty(queue)
                    work_share, task = pop!(queue)
                    final_δ_bound = task.distance_bound
                    
                    # @timeit to "Resource Check" begin
                    if !check_resources(start_time, timeout, 0.1)
                        empty!(queue)
                        GC.gc()
                        return UNKNOWN, nothing, (initial_δ_bound, final_δ_bound)
                    end
                    # end
    
                    @timeit to "Zonotope Propagate" begin
                        Zin = to_diff_zono(task)
                        prop_state = PropState(task, inter_contract)
                        Zout = N(Zin, prop_state)
                    end
                    
                    @timeit to "Property Check" begin
                        if prop_state.contract
                            if prop_state.isempty_intersection
                                continue
                            end
                        end
    
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
                        N̂ = size(Zout.∂Z.G, 2)
                        input_bounds = [-ones(N̂) ones(N̂)]
    
                        if !isempty(prop_state.split_constraints)
                            bounds = zono_bounds(Zout.∂Z)
    
                            # Compute all output dimensions that still need to be proven
                            mask = abs.(bounds) .> ϵ .&& task.branch.undetermined
    
                            # Append zeros to the constraints vectors so that the match the output dimension
                            @timeit to "Align Constraints" begin
                                for constraint in prop_state.split_constraints
                                    offset = ifelse(constraint.node.network == 1, 0, Zout.num_approx₁)
                                    constraint.g = align_vector(constraint.g, N̂, input_dim, offset)
                                end
                            end
                            
                            if use_lp
    
                                @timeit to "Initialize LP-solver" begin
                                    # Initialize the LP solver
                                    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                                    set_time_limit_sec(model, 10)
                                    
                                    # Add input variables
                                    @variable(model, -1 <= x[1:N̂] <= 1)
            
                                    # Add split constraints
                                    for (;node, g, c) in prop_state.split_constraints
                                        @constraint(model, node.direction * (g'x + c) >= 0.0)
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
                                
                            elseif use_zono_contract
                                if post_contract
                                    @timeit to "Post-Contract Zono" begin
                                        @timeit to "Sort Constraints" begin
                                            sort_constraints!(prop_state.split_constraints, zeros(N̂))
                                        end
    
                                        empty_intersection = false
                                        input_bounds_old = zeros(N̂, 2)
                                        @timeit to "Fixpoint Contract" begin
                                            first_round = true
                                            iter_count = 0
                                            initial_bounds, final_bounds = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                                            while !empty_intersection && input_bounds != input_bounds_old
                                                input_bounds_old .= input_bounds
                                                for (;node, g, c) in prop_state.split_constraints
                                                    @timeit to "Contract Zono" begin
                                                        input_bounds = contract_zono(input_bounds, g, c, node.direction)
                                                        if isnothing(input_bounds)
                                                            empty_intersection = true
                                                            break
                                                        end
                                                    end
                                                end
                                                if first_round && !empty_intersection
                                                    first_round = false
                                                    compute_bounds = Z -> offset_zono_bounds(input_bounds, Z)
                                                    initial_bounds = maximum.(abs, compute_bounds.((Zout.Z₁, Zout.Z₂, Zout.∂Z)))
                                                    final_bounds = initial_bounds
                                                end
                                                iter_count += 1
                                            end
                                        end
                                        
                                        if empty_intersection
                                            @timeit to "Empty Intersection" begin
                                                if iter_count > 1
                                                    println("Initial Bounds (NN₁, NN₂, ∂NN): $initial_bounds")
                                                    println("Final Bounds (NN₁, NN₂, ∂NN): $final_bounds")
                                                    println("Fixpoint Iterations: $iter_count")
                                                end
                                                continue
                                            end
                                        end
    
                                        if !all(isone.(abs.(input_bounds)))
                                            @timeit to "Transform Zono" begin
                                                transform_offset_diff_zono!(input_bounds, Zout)
                                                final_bounds = maximum.(abs, zono_bounds.((Zout.Z₁, Zout.Z₂, Zout.∂Z)))
                                                println("Initial Bounds (NN₁, NN₂, ∂NN): $initial_bounds")
                                                println("Final Bounds (NN₁, NN₂, ∂NN): $final_bounds")
                                                println("Fixpoint Iterations: $iter_count")
                                            end
    
                                            @timeit to "Property Check" begin
                                                prop_satisfied, cex, _, _, _ = property_check(N₁, N₂, Zin, Zout, nothing)
            
                                                if !prop_satisfied
                                                    if !isnothing(cex)
                                                        return UNSAFE, cex, (initial_δ_bound, final_δ_bound)
                                                    end
            
                                                    bounds = zono_bounds(Zout.∂Z)
                                                    distance_bound = maximum(abs, bounds)
                                                    mask .&= abs.(bounds) .> ϵ
                                                else
                                                    continue
                                                end
                                            end
        
                                            if !pre_contract
                                                transform_verification_task!(task, input_bounds)
                                            end
                                        end
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
    
                                @timeit to "DeepSplit Heuristic" begin
                                    split_candidate = split_heuristic(Zout, prop_state, task.distance_indices)
                                end
    
                                distance_bound = min(distance_bound, task.distance_bound)
                                task₁, task₂ = split_node(split_candidate, pre_contract, input_bounds, task, work_share, distance_bound)
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
            catch e
                if e isa OutOfMemoryError
                    empty!(queue)
                    GC.gc()
                    println("\nMEMOUT")
                    return UNKNOWN, nothing, (initial_δ_bound, final_δ_bound)
                else
                    rethrow(e)
                end
            end
        end
        return SAFE, nothing, (initial_δ_bound, final_δ_bound)
    end
end

function split_node(node::SplitConstraint, pre_contract::Bool, input_bounds::Matrix{Float64}, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    if node.node.layer == 0
        @timeit to "Split Input" begin
            return split_zono(node.node.neuron, task, work_share, nothing, distance_bound)
        end
    end
    @timeit to "Split Neuron" begin
        return split_neuron(node, pre_contract, input_bounds, task, work_share, distance_bound)
    end
end

function split_neuron(split_node::SplitConstraint, pre_contract::Bool, input_bounds::Matrix{Float64}, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    
    @timeit to "Create Tasks" begin
        (;node, g, c) = split_node
        direction₁, direction₂ = -1, 1 # inactive, active
    
        branch₁, branch₂ = task.branch, deepcopy(task.branch)
        push!(branch₁.split_nodes, SplitNode(node.network, node.layer, node.neuron, direction₁))
        push!(branch₂.split_nodes, SplitNode(node.network, node.layer, node.neuron, direction₂))
        
        task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, branch₁)
        task₂ = VerificationTask(deepcopy(task.middle), deepcopy(task.distance), task.distance_indices, deepcopy(task.∂Z), deepcopy(task.verification_status), distance_bound, branch₂)
    end

    if pre_contract
        @timeit to "Pre-Contract Zono" begin
            input_bounds₁, input_bounds₂ = input_bounds, deepcopy(input_bounds)
    
            task₁ = contract_to_verification_task!(input_bounds₁, g, c, direction₁, task₁)
            task₂ = contract_to_verification_task!(input_bounds₂, g, c, direction₂, task₂)
        end
    end

    return (work_share / 2.0, task₁), (work_share / 2.0, task₂)
end

function align_vector(g::Vector{Float64}, len::Int64, offset₁::Int64, offset₂::Int64)
    ĝ = zeros(len)
    ĝ[1:offset₁] .= g[1:offset₁]
    ĝ[(offset₁ + offset₂ + 1) : (offset₂ + size(g, 1))] .= g[(offset₁ + 1) : end]
    return ĝ
end

function check_resources(start_time::UInt64, timeout::Int64, mem_min::Float64)
    timeout_reached = (time_ns() - start_time) / 1.0e9 > timeout
    
    if timeout_reached
        println("\nTIMEOUT REACHED")
    end
    
    return !timeout_reached
end

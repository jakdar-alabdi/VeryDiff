function deepsplit_verify_network(N₁::Network, N₂::Network, Zin::Zonotope, ϵ::Float64)
    return deepsplit_verify_network(N₁, N₂, zono_bounds(Zin), ϵ)
end

function deepsplit_verify_network(N₁::Network, N₂::Network, bounds, epsilon::Float64; timeout=Inf, fuzz_testing=nothing)
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
            status, cex, δ_bounds = deepsplit_verify_network(epsilon; fuzz_testing=fuzz_testing)(N, N₁, N₂, initial_task, split_heuristic; timeout=timeout)
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

function deepsplit_verify_network(ϵ::Float64; fuzz_testing=nothing)
    property_check = get_epsilon_property(ϵ)
    
    return (N::GeminiNetwork, N₁::Network, N₂::Network, initial_task::VerificationTask, split_heuristic; timeout=Inf) -> begin
        start_time = time_ns()
        first_task = true
        initial_δ_bound = Inf64
        final_δ_bound = Inf64

        approach = NEURON_SPLITTING_APPROACH[]
        contract = ZONO_CONTRACT_MODE[]
        use_vertical_splitting = approach == VerticalSplitting
        use_lp = approach == LP
        use_zono_contract = approach == ZonoContraction
        use_lp_zc = use_zono_contract && contract == LPZonoContract
        inter_contract = use_lp_zc || use_zono_contract && (contract == ZonoContractInter)
        post_contract = use_zono_contract && (contract == ZonoContract || contract == ZonoContractPost)
        pre_contract = use_zono_contract && (contract == ZonoContract || contract == ZonoContractPre)

        queue = Queue()
        push!(queue, (1.0, initial_task))
        
        @timeit to "Zonotope Loop" begin
            try
                while !isempty(queue)
                    work_share, task = pop!(queue)
                    final_δ_bound = task.distance_bound
                    
                    # @timeit to "Resource Check" begin
                    if !check_resources(start_time; timeout=timeout)
                        empty!(queue)
                        GC.gc()
                        return UNKNOWN, nothing, (initial_δ_bound, final_δ_bound)
                    end
                    # end
    
                    @timeit to "Zonotope Propagate" begin
                        Zin = to_diff_zono(task)
                        input_dim = size(Zin.Z₁.G, 2)
                        prop_state = PropState()
                        prop_state.split_nodes = task.branch.split_nodes
                        prop_state.input_bounds = [-ones(input_dim) ones(input_dim)]
                        Zout = N(Zin, prop_state)
                        if prop_state.is_unsatisfiable
                            @timeit to "Unsatisfiable" begin
                                continue
                            end
                        end
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

                        N̂ = size(Zout.∂Z.G, 2)
                        input_bounds = [-ones(N̂) ones(N̂)]
                                                
                        # Compute all output dimensions that still need to be proven
                        bounds = zono_bounds(Zout.∂Z)
                        mask = abs.(bounds) .> ϵ .&& task.branch.undetermined
    
                        if !isempty(prop_state.split_constraints)
    
                            # Append zeros to the constraints vectors so that the match the output dimension
                            @timeit to "Align Constraints" begin
                                for constraint in prop_state.split_constraints
                                    offset = ifelse(constraint.node.network == 1, 0, Zout.num_approx₁)
                                    constraint.g = align_vector(constraint.g, N̂, input_dim, offset)
                                end
                            end
                            
                            if use_lp || use_lp_zc
    
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
                                    lp_distance_bound = 0.0
        
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
                                                lp_distance_bound = max(lp_distance_bound, δ)
                                            end
            
                                            mask[i, j] &= termination_status(model) != MOI.INFEASIBLE
                                        end
                                    end
                                end

                                distance_bound = min(distance_bound, lp_distance_bound)
                                
                            elseif use_zono_contract
                                if post_contract
                                    @timeit to "Post-Contract Zono" begin
                                        @timeit to "Sort Constraints" begin
                                            sort_constraints!(prop_state.split_constraints, zeros(N̂))
                                        end
    
                                        empty_intersection = false
                                        # input_bounds_old = zeros(N̂, 2)
                                        # @timeit to "Fixpoint Contract" begin
                                        #     first_round = true
                                        #     iter_count = 0
                                        #     pre_contract_bounds = (Zout.Z₁, Zout.Z₂, Zout.∂Z) |> Base.Fix1(map, Z -> maximum(abs, zono_bounds(Z)))
                                        #     initial_bounds, final_bounds = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                                        #     while !empty_intersection && input_bounds != input_bounds_old
                                        #         input_bounds_old .= input_bounds
                                        for (;node, g, c) in prop_state.split_constraints
                                            @timeit to "Contract Zono" begin
                                                input_bounds = contract_zono(input_bounds, g, c, node.direction)

                                                if isnothing(input_bounds)
                                                    empty_intersection = true
                                                    break
                                                end
                                            end
                                        end
                                        #         if first_round && !empty_intersection
                                        #             first_round = false
                                        #             initial_bounds = (Zout.Z₁, Zout.Z₂, Zout.∂Z) |> Base.Fix1(map, Z -> maximum(abs, offset_zono_bounds(input_bounds, Z)))
                                        #             final_bounds = initial_bounds
                                        #         end
                                        #         iter_count += 1
                                        #     end
                                        # end
              
                                        # if iter_count > 1
                                        #     final_bounds = (0, 0, 0)
                                        #     if !empty_intersection
                                        #         final_bounds = (Zout.Z₁, Zout.Z₂, Zout.∂Z) |> Base.Fix1(map, Z -> maximum(abs, offset_zono_bounds(input_bounds, Z)))
                                        #     end
                                        # end

                                        # println("Initial         Bounds (NN₁, NN₂, ∂NN): $pre_contract_bounds")
                                        # println("First-Contract  Bounds (NN₁, NN₂, ∂NN): $initial_bounds")
                                        # println("Final-Contract  Bounds (NN₁, NN₂, ∂NN): $final_bounds")
                                        # println("Fixpoint Iterations: $iter_count")
                                        # println("Num Constraints: $(size(prop_state.split_constraints, 1))")
                                        # println("########################################################################################")
                                        
                                        if empty_intersection
                                            @timeit to "Empty Intersection" begin
                                                continue
                                            end
                                        end
    
                                        if !is_unit_hypercube(input_bounds)
                                            @timeit to "Transform Zono" begin
                                                transform_offset_diff_zono!(input_bounds, Zout)
                                            end
    
                                            @timeit to "Property Check" begin
                                                prop_satisfied, cex, _, _, _ = property_check(N₁, N₂, Zin, Zout, nothing)
            
                                                if !prop_satisfied
                                                    if !isnothing(cex)
                                                        return UNSAFE, cex, (initial_δ_bound, final_δ_bound)
                                                    end
            
                                                    bounds = zono_bounds(Zout.∂Z)
                                                    distance_bound = min(distance_bound, maximum(abs, bounds))
                                                    mask .&= abs.(bounds) .> ϵ
                                                else
                                                    continue
                                                end
                                            end
                                        end
                                    end
                                end
                            else
                                # TODO implement vertical splitting
                                println("Test")
                                throw(ErrorException("Vertical splitting not implemented yet :("))
                            end
                        end
                        
                        task.branch.undetermined = mask
                        if any(task.branch.undetermined)
                            @timeit to "Compute Split" begin
    
                                @timeit to "DeepSplit Heuristic" begin
                                    split_candidate = split_heuristic(Zout, task.branch.split_nodes, prop_state, task.distance_indices, mask[:, 1] .|| mask[:, 2])
                                end
    
                                distance_bound = min(distance_bound, task.distance_bound)
                                final_δ_bound = distance_bound

                                if inter_contract
                                    input_bounds[1:size(prop_state.input_bounds, 1), :] .= prop_state.input_bounds
                                end

                                if NEURON_SPLITTING_APPROACH[] == ZonoContraction && split_candidate.layer == 0
                                    @timeit to "Split Input" begin
                                        (ws₁, task₁), (ws₂, task₂) = split_contract_zono(split_candidate.neuron, input_bounds, prop_state.split_constraints, task, work_share, distance_bound)
                                    end
                                else
                                    if !pre_contract && !is_unit_hypercube(input_bounds)
                                        @timeit to "Transform Input Zono" begin
                                            task = transform_verification_task(task, input_bounds)
                                        end
                                    end
                                    (ws₁, task₁, direction₁), (ws₂, task₂, direction₂) = split_node(split_candidate, task, work_share, distance_bound)
                                end

                                if pre_contract && split_candidate.layer > 0
                                    @timeit to "Pre-Contract Zono" begin
                                        offset = ifelse(split_candidate.network == 1, 0, Zout.num_approx₁)
                                        Z = prop_state.intermediate_zonos[split_candidate.network][split_candidate.layer]
                                        g = align_vector(Z.G[split_candidate.neuron, :], N̂, input_dim, offset)
                                        c = Z.c[split_candidate.neuron]

                                        input_bounds₁, input_bounds₂ = input_bounds, deepcopy(input_bounds)

                                        task₁ = contract_to_verification_task(input_bounds₁, g, c, direction₁, task₁)
                                        task₂ = contract_to_verification_task(input_bounds₂, g, c, direction₂, task₂)
                                    end
                                end
                                
                                if !isnothing(task₁)
                                    push!(queue, (ws₁, task₁))
                                end
                                if !isnothing(task₂)
                                    push!(queue, (ws₂, task₂))
                                end
                            end
                        end

                        # Tests empirically whether the bounds computed by LP are valid
                        if !isnothing(fuzz_testing)
                            @timeit to "Random Test" begin
                                fuzz_testing(Zin.Z₁, distance_bound, queue)
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

function split_node(node::SplitNode, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    if node.layer == 0
        @timeit to "Split Input" begin
            (ws₁, task₁), (ws₂, task₂) = split_zono(node.neuron, task, work_share, nothing, distance_bound)
            return (ws₁, task₁, 0), (ws₂, task₂, 0)
        end
    end
    @timeit to "Split Neuron" begin
        return split_neuron(node, task, work_share, distance_bound)
    end
end

function split_neuron(node::SplitNode, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    direction₁, direction₂ = -1, 1 # inactive, active
    bounds₁ = bounds₂ = nothing
    (;network, layer, neuron) = node
    if NEURON_SPLITTING_APPROACH[] == VerticalSplitting
        split_nodes = task.branch.split_nodes
        n = findfirst(n -> (n.network, n.layer, n.neuron) == (network, layer, neuron), split_nodes)
        if !isnothing(n)
            node = split_nodes[n]
            task.branch.split_nodes = vcat(split_nodes[1:n-1], split_nodes[n+1:end])
            
            if node.direction == 1
                @timeit to "Lower Part" begin
                    l, u = node.bounds[1], node.bounds[2]
                    s₁, s₂ = (l, u) ./ 2
                    bounds₁ = [l s₁; s₂ u]
                    bounds₂ = [s₁ s₂]
                end
            else
                @timeit to "Upper Part" begin
                    direction₁ = direction₂ = -1
                    l₁, u₁ = node.bounds[1, 1], node.bounds[1, 2]
                    l₂, u₂ = node.bounds[2, 1], node.bounds[2, 1]
                    s₁, s₂ = (l₁ + u₁, l₂ + u₂) ./ 2
                    bounds₁ = [l₁ s₁; s₂ u₂]
                    bounds₂ = [s₁ u₁; l₂ s₂]
                end
            end
        end
    end
    
    branch₁, branch₂ = task.branch, deepcopy(task.branch)
    push!(branch₁.split_nodes, SplitNode(network, layer, neuron, direction₁, bounds₁))
    push!(branch₂.split_nodes, SplitNode(network, layer, neuron, direction₂, bounds₂))
    
    task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, branch₁)
    task₂ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, branch₂)

    return (work_share / 2.0, task₁, direction₁), (work_share / 2.0, task₂, direction₂)
end

function split_contract_zono(d::Int, input_bounds::Matrix{Float64}, constraints::Vector{SplitConstraint}, task::VerificationTask, work_share::Float64, distance_bound::Float64)
    distance_d = findfirst(x -> x == d, task.distance_indices)
    @assert !isnothing(distance_d)

    input_bounds₁, input_bounds₂ = input_bounds, deepcopy(input_bounds)
    cutting_point = (input_bounds[distance_d, 1] + input_bounds[distance_d, 2]) / 2
    input_bounds₁[distance_d, 1] = cutting_point
    input_bounds₂[distance_d, 2] = cutting_point

    task₁ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, task.branch)
    task₂ = VerificationTask(task.middle, task.distance, task.distance_indices, task.∂Z, task.verification_status, distance_bound, task.branch)

    task₁ = contract_all_to_verification_task(task₁, input_bounds₁, constraints)
    task₂ = contract_all_to_verification_task(task₂, input_bounds₂, constraints)

    return (work_share / 2.0, task₁), (work_share / 2.0, task₂)
end

function align_vector(g::Vector{Float64}, len::Int64, offset₁::Int64, offset₂::Int64)
    ĝ = zeros(len)
    ĝ[1:offset₁] .= g[1:offset₁]
    ĝ[(offset₁ + offset₂ + 1) : (offset₂ + size(g, 1))] .= g[(offset₁ + 1) : end]
    return ĝ
end

function check_resources(start_time::UInt64; timeout=Inf)
    timeout_reached = (time_ns() - start_time) / 1.0e9 > timeout
    
    if timeout_reached
        println("\nTIMEOUT REACHED")
    end
    
    return !timeout_reached
end

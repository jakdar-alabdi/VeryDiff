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
            input_bounds = [-ones(size(non_zero_indices, 1)) ones(size(non_zero_indices, 1))]
            ∂Z = Zonotope(Matrix(0.0I, input_dim, size(non_zero_indices, 1)), zeros(Float64, input_dim), nothing)
            initial_task = VerificationTask(mid, distance, non_zero_indices, ∂Z, nothing, Inf64, Branch(trues(1, 2), input_bounds))

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
                    
                    if !isempty(prop_state.split_nodes)
                        input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
                        
                        var_num = size(Zout.∂Z.G, 2)
                        input_bounds = [-ones(var_num) ones(var_num)]
                        input_bounds[1:input_dim, :] .= task.branch.input_bounds
                        
                        for node in prop_state.split_nodes
                            offset = ifelse(node.network == 1, 0, Zout.num_approx₁)
                            g = zeros(var_num)
                            g[1:input_dim] .= node.g[1:input_dim]
                            g[(input_dim + offset + 1) : (offset + size(node.g, 1))] .= node.g[(input_dim + 1) : end]
                            node.g = g
                        end
                        
                        @timeit to "Contract Zono" begin
                            
                            # sort_prop = node -> -node.score

                            centroid = (input_bounds[:, 1] + input_bounds[:, 2]) ./ 2.0
                            sort_prop = node -> geometric_distance(centroid, node.g, node.c)

                            sorted_consts = sort(prop_state.split_nodes, by=sort_prop)
                            # sorted_consts = prop_state.split_nodes
                            
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

                            task.branch.input_bounds = input_bounds[1:input_dim, :]

                            # in_lower = @view task.branch.input_bounds[:, 1]
                            # in_upper = @view task.branch.input_bounds[:, 2]
                            # b = vcat([sum(ifelse.(g .>= 0.0, g .* [in_lower in_upper], g .* [in_upper in_lower]), dims=1) for g in eachrow(Zin.Z₁.G)]...)
                            # bounds = b .+ Zin.Z₁.c

                            # lower = @view bounds[:, 1]
                            # upper = @view bounds[:, 2]
                            # mid = (upper .+ lower) ./ 2.0
                            # distance = mid .- lower
                            # middle = task.middle
                            # middle[task.distance_indices] .= mid

                            # task = VerificationTask(middle, distance, task.distance_indices, task.∂Z, nothing, task.distance_bound, task.branch)
                        end

                        @timeit to "Initialize LP-solver" begin
                            # Initialize the LP solver
                            model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                            set_time_limit_sec(model, 10)
                            
                            # Add input variables
                            @variable(model, input_bounds[i, 1] <= x[i=1:var_num] <= input_bounds[i, 2])
    
                            # Add split constraints
                            for (;g, c, direction) in prop_state.split_nodes
                                @constraint(model, direction * (g'x + c) >= 0.0)
                            end
                        end
    
                        @timeit to "Solve LP" begin
                            distance_bound = 0.0
                            bounds = zono_bounds(Zout.∂Z)
                            # Compute all output dimensions that still need to be proven
                            mask = abs.(bounds) .> ϵ .&& task.branch.undetermined

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
                            println("Next Branch Input Bounds:")
                            println(next_task.branch.input_bounds[:, 1])
                            println(next_task.branch.input_bounds[:, 2])
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

function contract_zono(bounds::Matrix{Float64}, g::Vector{Float64}, c::Float64, d::Int64)
    n, _ = size(bounds)
    # @assert d == 1 || d == -1 "Unspecified direction"
    
    l = @view bounds[:, 1]
    u = @view bounds[:, 2]
    
    # With (g, c, d) we impose a linear constraint on the input space
    # Depending on the given direction d (i.e., d = −1 or d = 1 for inactive or active ReLU-phase) 
    # we have one of the following constraints:
    # For d == −1 (incative phase):
    #   gᵀx + c <= 0.0 ⇔ gᵀx <= −c ⇔ -d * gᵀx <= d * c
    # For d == 1 (active phase):
    #   gᵀx + c >= 0.0 ⇔ −gᵀx <= c ⇔ -d * gᵀx <= d * c
    g, c = -d * g, d * c

    # Compute a vector v ∈ [l₁, u₁] × ... × [lₙ, uₙ] that minimizes the dot prodoct gᵀv
    v = ifelse.(g .>= 0.0, l, u)

    # If gᵀv > c then we have an empty intersection
    s = g'v
    if s > c
        return nothing
    end

    # For each input dimension i we attempt to increase lᵢ and decrease uᵢ
    for i in 1:n
        if g[i] != 0.0
            # x = (1 / g[i]) * (c - g[1:i-1]'v[1:i-1] - g[i+1:end]'v[i+1:end])
            x = (1 / g[i]) * (c - (s - g[i] * v[i])) # ⇔ x = (1 / g[i]) (c - g[1:i-1]ᵀv[1:i-1] - g[i+1:]ᵀv[i+1:])
            if g[i] > 0
                u[i] = min(u[i], x)
            else
                l[i] = max(l[i], x)
            end
        end
    end

    return bounds
end

function geometric_distance(x̂::Vector{Float64}, g::Vector{Float64}, c::Float64)
    return abs(g'x̂ + c) / sqrt(g'g)
end

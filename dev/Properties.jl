function get_epsilon_property_with_neuron_splitting(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    VeryDiff.EQUIVALENCE_PROPERTY[] = VeryDiff.EpsilonEquivalence

    return (N₁::Network, N₂::Network, prop_state::PropState) -> begin
        Zin = prop_state.zono_storage.zonotopes[1].zonotope
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        mask = prop_state.task.branch.undetermined
        split_nodes = prop_state.task.branch.split_nodes
        bounds_cache = prop_state.task_bounds.bounds_cache
        box = nothing

        prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; mask=mask)
        if prop_satisfied || !isnothing(cex) || isempty(split_nodes)
            return prop_satisfied, cex, nothing, nothing, distance_bound, nothing
        end

        if VeryDiff.USE_ZONO_CONTRACT[]
            sort_split_nodes!(split_nodes, prop_state)
        end

        if VeryDiff.USE_LP[] || !VeryDiff.USE_VERTICAL_SPLITTING[] && prop_state.num_instable == 0
            model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            set_time_limit_sec(model, 10)
            
            xs = [@variable(model, [1:size(G, 2)], lower_bound=-1.0, upper_bound=1.0) for G in Zout.∂Z.Gs]

            for node in split_nodes
                (;network, diff_layer, neuron, direction, bounds) = node
                Z = VeryDiff.get_split_node_zono(node, prop_state)
                indices = intersect_indices(Zout.∂Z.generator_ids, Z.generator_ids)
                bc = bounds_cache[diff_layer.layer_idx]
                if network == 1
                    lower, upper = bc.lower₁[neuron], bc.upper₁[neuron]
                else
                    lower, upper = bc.lower₂[neuron], bc.upper₂[neuron]
                end
                expr = sum(G[neuron, :]'xs[i][1:size(G, 2)] for (G, i) in zip(Z.Gs, indices)) + Z.c[neuron]
                @constraint(model, lower <= expr <= upper)
                # @constraint(model, direction * (sum(G[neuron, :]'xs[i][1:size(G, 2)] for (G, i) in zip(Z.Gs, indices)) + Z.c[neuron]) >= 0.0)
            end

            _distance_bound = 0.0
            for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]
                    @objective(model, Max, σ * (sum(G[i, :]'xs[k] for (k, G) in enumerate(Zout.∂Z.Gs)) + Zout.∂Z.c[i]))
                    optimize!(model)

                    numeric_foucs_opt = prop_state.num_instable == 0 && has_values(model) && abs(objective_value(model)) > epsilon
                    if numeric_foucs_opt
                        set_optimizer_attribute(model, "NumericFocus", 3)
                        optimize!(model)
                    end
                    
                    if is_solved_and_feasible(model)
                        val = value.(xs[1])
                        cex_input = Zin.Z₁.Gs[1] * val + Zin.Z₁.c
                        sample_distance = get_sample_distance(N₁, N₂, cex_input)

                        if prop_state.num_instable == 0 && any(mask)
                            @info "[LP Solution] x = $(val)"
                            @info "Zin(x) = $cex_input"
                            @info "sample distance: $sample_distance"
                            @info "obj. value: $(abs(objective_value(model)))"
                        end

                        if sample_distance > epsilon
                            return false, (cex_input, (N₁(cex_input), N₂(cex_input), sample_distance)), nothing, nothing, distance_bound, nothing
                        end
                    end
                    if has_values(model)
                        δ = abs(objective_value(model))
                        mask[i, j] &= δ > epsilon
                        _distance_bound = max(_distance_bound, δ)
                    end
                    mask[i, j] &= termination_status(model) != MOI.INFEASIBLE
                end
            end
            @assert !(prop_state.num_instable == 0 && any(mask))
            distance_bound = min(distance_bound, _distance_bound)
            
        elseif VeryDiff.POST_CONTRACT[]
            box = contract_zono_all!(InputBox(Zout), split_nodes, prop_state) 
            if isnothing(box)
                return true, nothing, nothing, nothing, distance_bound, nothing
            end
            if !is_unit_hypercube(box)
                Zout = transform_offset_diff_zono!(box, Zout)
                prop_satisfied, cex, _, _, _distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; mask=mask)
                if prop_satisfied || !isnothing(cex)
                    return prop_satisfied, cex, nothing, nothing, distance_bound, box
                end
                distance_bound = min(distance_bound, _distance_bound)
            end
        end

        return !any(mask), nothing, nothing, nothing, distance_bound, box
    end
end

function get_top1_property_with_neuron_splitting(delta::Float64)
    property_check = get_top1_property(;delta=delta)

    EQUIVALENCE_PROPERTY[] = DeltaTop1Equivalence

    approach = NEURON_SPLITTING_APPROACH[]
    contract = ZONO_CONTRACT_MODE[]
    use_zono_contract = approach == ZonoContraction
    post_contract = use_zono_contract && (contract == ZonoContract || contract == ZonoContractPost)

    return (N₁::Network, N₂::Network, Zin::DiffZonotope, Zout::DiffZonotope, prop_state::PropState) -> begin

        constraints = prop_state.split_constraints
        input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
        N̂ = size(Zout.∂Z.G, 2)
        distance_bound = maximum(abs, zono_bounds(Zout.∂Z))
        
        input_bounds = nothing
        
        if !isempty(constraints)
            if post_contract
                sort_constraints!(constraints, zeros(N̂))
                input_bounds = contract_zono_all!([-ones(N̂) ones(N̂)], constraints)
                    
                if isnothing(input_bounds)
                    return true, nothing, nothing, nothing, task.distance_bound, nothing
                end
                    
                if !is_unit_hypercube(input_bounds)
                    Zout = transform_offset_diff_zono!(input_bounds, Zout)
                    constraints = transform_constraints!(input_bounds, constraints)
                    input_bounds = nothing
                end
            end
        end
        
        prop_satisfied, cex, heuristics_info, verification_status, p_distance_bound = property_check(N₁, N₂, Zin, Zout, task.verification_status; constraints=constraints)
        distance_bound = min(distance_bound, p_distance_bound)

        if prop_state.num_instables == 0
            if !prop_satisfied && isnothing(cex)
                for _ in 1:1000
                    x = rand(Float64, input_dim)
                    input = Zin.Z₁.G * x + Zin.Z₁.c
                    y₁ = N₁(input)
                    y₂ = N₂(input)

                    argmax_N₁ = argmax(y₁)
                    argmax_N₂ = argmax(y₂)

                    if argmax_N₁ != argmax_N₂
                        softmax_N₁ = exp.(y₁) / sum(exp.(y₁))
                        if iszero(delta) || softmax_N₁[argmax_N₁] >= delta
                            println("Found cex")
                            second_most = sort(softmax_N₁, rev=true)[2]
                            println("N1: $(softmax_N₁[argmax_N₁]) (vs. $second_most)")
                            softmax_N₂ = exp.(y₂)/sum(exp.(y₂))
                            println("N2: $(softmax_N₂[argmax_N₂])")
                            println("N1 Probability: $(softmax_N₁[argmax_N₁]) >= $delta")
                            return false, (input, (argmax_N₁, argmax_N₂)), nothing, nothing, distance_bound, input_bounds
                        end
                    end
                end
            end
        end

        return prop_satisfied, cex, heuristics_info, verification_status, distance_bound, input_bounds
    end
end

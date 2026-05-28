function get_epsilon_property_with_neuron_splitting(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    VeryDiff.EQUIVALENCE_PROPERTY[] = VeryDiff.EpsilonEquivalence

    return (N₁::Network, N₂::Network, prop_state::PropState) -> begin
        Zin = prop_state.zono_storage.zonotopes[1].zonotope
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        safe_out_dim = prop_state.task.branch.safe_out_dim
        split_nodes = prop_state.task.branch.split_nodes
        box = nothing

        prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; safe_out_dim=safe_out_dim)
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
                (;network, neuron, diff_layer, direction, bounds) = node
                Z = VeryDiff.get_split_node_zono(node, prop_state)
                indices = intersect_indices(Zout.∂Z.generator_ids, Z.generator_ids)
                affine_repr = AffExpr(Z.c[neuron])
                for (G, i) in zip(Z.Gs, indices)
                    add_to_expression!(affine_repr, G[neuron, :]'xs[i][1:size(G, 2)])
                end
                @constraint(model, direction * affine_repr >= 0.0)
            end

            @objective(model, Max, 0.0)
            optimize!(model)

            if termination_status(model) == MOI.INFEASIBLE
                safe_out_dim .= true
            else
                _distance_bound = 0.0
                for dim in findall(!, safe_out_dim)
                    dim_num, dim_bound = Tuple(dim)
                    σ = ifelse(dim_bound == 1, -1, 1)
                    @objective(model, Max, σ * (sum(G[dim_num, :]'xs[k] for (k, G) in enumerate(Zout.∂Z.Gs)) + Zout.∂Z.c[dim_num]))
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
                        δ = abs(objective_value(model))
    
                        if δ > epsilon && prop_state.num_instable == 0 && any(!, safe_out_dim)
                            @info "---------------------------------------------"
                            @info "[LP Solution] x = $(val)"
                            @info "Zin(x) = $cex_input"
                            @info "sample distance: $sample_distance"
                            @info "obj. value: $(abs(objective_value(model)))"
                            for node in split_nodes
                                (;network, layer, neuron, diff_layer, direction, bounds) = node
                                Z = VeryDiff.get_split_node_zono(node, prop_state)
                                indices = intersect_indices(Zout.∂Z.generator_ids, Z.generator_ids)
                                v = Z.c[neuron]
                                for (G, i) in zip(Z.Gs, indices)
                                    v += G[neuron, :]'value(xs[i])[1:size(G, 2)]
                                end
                                v *= direction
                                if v < 0.0
                                    @info "split node: $((network, layer, neuron, direction)), $v"
                                end
                            end
                        end
    
                        if sample_distance > epsilon
                            return false, (cex_input, (N₁(cex_input), N₂(cex_input), sample_distance)), nothing, nothing, distance_bound, nothing
                        end
                    end
                    if has_values(model)
                        δ = abs(objective_value(model))
                        safe_out_dim[dim] |= δ <= epsilon
                        _distance_bound = max(_distance_bound, δ)
                    end
                    safe_out_dim[dim] |= termination_status(model) == MOI.INFEASIBLE
                end

                if prop_state.num_instable == 0 && any(!, safe_out_dim)
                    (;middle, distance, distance_indices) = prop_state.task
                    Z = Zonotope([G₁ - G₂ for (G₁, G₂) in zip(Zout.Z₁.Gs, Zout.Z₂.Gs)], Zout.Z₁.c - Zout.Z₂.c, nothing, Zout.Z₁.generator_ids, nothing)
                    @info "bounds(Zin): $(zono_bounds(Zin.Z₁))"
                    @info "bounds(task): $([(middle[distance_indices] .- distance) (middle[distance_indices] .+ distance)])"
                    @info "bounds(Z₁ - Z₂): $(zono_bounds(Z))"
                    @info "bounds(∂Z): $(zono_bounds(Zout.∂Z))"
                    @info "split nodes: $(map(n -> (n.network, n.layer, n.neuron, n.direction), prop_state.task.branch.split_nodes))"
                end

                @assert !(prop_state.num_instable == 0 && any(!, safe_out_dim))
                distance_bound = min(distance_bound, _distance_bound)
            end
            
        elseif VeryDiff.POST_CONTRACT[]
            box = contract_zono_all!(InputBox(Zout), split_nodes, prop_state) 
            if isnothing(box)
                return true, nothing, nothing, nothing, distance_bound, nothing
            end
            if !is_unit_hypercube(box)
                transform_offset_diff_zono!(box, Zout)
                prop_satisfied, cex, _, _, _distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; safe_out_dim=safe_out_dim)
                if prop_satisfied || !isnothing(cex)
                    return prop_satisfied, cex, nothing, nothing, distance_bound, nothing
                end
                distance_bound = min(distance_bound, _distance_bound)
            end
        end

        return all(safe_out_dim), nothing, nothing, nothing, distance_bound, box
    end
end

function get_top1_property_with_neuron_splitting(delta::Float64)
    property_check = get_top1_property(;delta=delta)
    VeryDiff.EQUIVALENCE_PROPERTY[] = VeryDiff.DeltaTop1Equivalence

    return (N₁::Network, N₂::Network, prop_state::PropState) -> begin
        Zin = prop_state.zono_storage.zonotopes[1].zonotope
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        split_nodes = prop_state.task.branch.split_nodes
        distance_bound = maximum(abs, zono_bounds(Zout.∂Z))
        distance_bound = min(distance_bound, prop_state.task.distance_bound)
        box = nothing

        if VeryDiff.USE_ZONO_CONTRACT[]
            sort_split_nodes!(split_nodes, prop_state)
        end
        
        if !isempty(split_nodes) && VeryDiff.POST_CONTRACT[]
            box = contract_zono_all!(InputBox(Zout), split_nodes, prop_state)
            if isnothing(box)
                return true, nothing, nothing, nothing, distance_bound, nothing
            end
            if !is_unit_hypercube(box)
                transform_offset_diff_zono!(box, Zout)
                transform_verification_task!(box, prop_state.task)
                transform_constraints!(box, split_nodes, prop_state)
                box = nothing # To ensure no contraction is done with this box anymore
            end
        end
        
        verification_status = prop_state.task.verification_status
        prop_satisfied, cex, heuristics_info, verification_status, _distance_bound = property_check(N₁, N₂, Zin, Zout, verification_status; prop_state=prop_state)
        distance_bound = min(distance_bound, _distance_bound)

        if !prop_satisfied && isnothing(cex) && prop_state.num_instable == 0
            for _ in 1:1000
                xs = [rand(size(G, 2)) for G in Zin.Z₁.Gs]
                input = sum(G * x for (G, x) in zip(Zin.Z₁.Gs, xs)) + Zin.Z₁.c
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
                        return false, (input, (argmax_N₁, argmax_N₂)), nothing, nothing, distance_bound, box
                    end
                end
            end
        end

        return prop_satisfied, cex, heuristics_info, verification_status, distance_bound, box
    end
end

function get_epsilon_property_with_neuron_splitting(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    VeryDiff.EQUIVALENCE_PROPERTY[] = VeryDiff.EpsilonEquivalence

    return (N₁::Network, N₂::Network, prop_state::PropState) -> begin
        zonotopes = get_zonos_at_pos(:, prop_state)
        Zin = get_zonotope(zonotopes[1])
        Zout = get_zonotope(zonotopes[end])
        ∂out_ids = Zout.∂Z.generator_ids
        ∂out_gens = Zout.∂Z.Gs
        mask = prop_state.task.branch.undetermined
        split_nodes = prop_state.task.branch.split_nodes
        num_instable = prop_state.num_instable
        box = nothing

        prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; mask=mask)
        if prop_satisfied || !isnothing(cex) || isempty(split_nodes)
            return prop_satisfied, cex, nothing, nothing, distance_bound, nothing
        end

        if VeryDiff.USE_ZONO_CONTRACT[]
            split_nodes = sort_split_nodes!(split_nodes, zonotopes)
        end

        if VeryDiff.USE_LP[] || prop_state.num_instable == 0
            model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            set_time_limit_sec(model, 10)
            
            xs = [@variable(model, [1:size(∂out_gens[find_index_position(∂out_ids, i)], 2)], lower_bound=-1.0, upper_bound=1.0) for i in ∂out_ids]

            for (;network, layer, neuron, direction) in split_nodes
                Z = zonotopes[layer].zonotope |> (z -> ifelse(network == 1, z.Z₁, z.Z₂))
                @constraint(model, direction * (sum(G[neuron, :]'xs[find_index_position(∂out_ids, i)][1:size(G, 2)] for (G, i) in zip(Z.Gs, Z.generator_ids)) + Z.c[neuron]) >= 0.0)
            end

            _distance_bound = 0.0
            for i in (1:size(mask, 1))[mask[:, 1] .|| mask[:, 2]]
                for (j, σ) in [(1, -1), (2, 1)][mask[i, :]]
                    @objective(model, Max, σ * (sum(G[i, :]'xs[k] for (k, G) in enumerate(∂out_gens)) + Zout.∂Z.c[i]))
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
            if prop_state.num_instable == 0 && any(mask)
                @info "$mask, $(findall(mask))"
                count_instable = 0
                for Z in get_zonotope.(zonotopes)
                    num_approx₁ = length(Z.Z₁.Gs) <= 1 ? 0 : sum(size(G, 2) for G in Z.Z₁.Gs) - size(Z.Z₁.Gs[1], 2)
                    num_approx₂ = length(Z.Z₂.Gs) <= 1 ? 0 : sum(size(G, 2) for G in Z.Z₂.Gs) - size(Z.Z₂.Gs[1], 2)
                    ∂num_approx = length(Z.∂Z.Gs) <= 1 ? 0 : sum(size(G, 2) for G in Z.∂Z.Gs) - size(Z.∂Z.Gs[1], 2)
                    @info "num_approx₁: $(num_approx₁), num_approx₂: $(num_approx₂), ∂num_approx: $(∂num_approx)"
                    count_instable += num_approx₁ + num_approx₂
                end
                @info "count_instable: $count_instable"
            end
            @assert !(prop_state.num_instable == 0 && any(mask))
            distance_bound = min(distance_bound, _distance_bound)
            
        elseif VeryDiff.POST_CONTRACT[]
            box = InputBox(Zout)
            box = contract_zono_all!(box, split_nodes, zonotopes) 
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

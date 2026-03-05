function get_epsilon_property_with_neuron_splitting(epsilon::Float64)
    property_check = get_epsilon_property(epsilon)

    approach = NEURON_SPLITTING_APPROACH[]
    contract = ZONO_CONTRACT_MODE[]
    use_lp = approach == LP
    use_zono_contract = approach == ZonoContraction
    use_lp_zc = use_zono_contract && contract == LPZonoContract
    post_contract = use_zono_contract && (contract == ZonoContract || contract == ZonoContractPost)

    return (N₁::Network, N₂::Network, Zin::DiffZonotope, Zout::DiffZonotope, task::VerificationTask, prop_state::PropState) -> begin
        
        mask = task.branch.undetermined
        prop_satisfied, cex, _, _, distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; mask=mask)
    
        constraints = prop_state.split_constraints
        num_instables = prop_state.num_instables
        if prop_satisfied || !isnothing(cex) || (isempty(constraints) && num_instables > 0)
            return prop_satisfied, cex, nothing, nothing, distance_bound, nothing
        end

        input_bounds = nothing
        input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
        N̂ = size(Zout.∂Z.G, 2)

        @assert !(prop_state.num_instables == 0 && N̂ != input_dim)

        # Append zeros to the constraints vectors so that the match the output dimension
        @timeit to "Align Constraints" begin
            for constraint in constraints
                offset = ifelse(constraint.node.network == 1, 0, Zout.num_approx₁)
                constraint.g = align_vector(constraint.g, N̂, input_dim, offset)
            end
        end

        if use_lp || use_lp_zc || num_instables == 0
            
            @timeit to "Initialize LP-solver" begin
                # Initialize the LP solver
                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
                set_time_limit_sec(model, 10)
                
                # Add input variables
                @variable(model, -1.0 <= x[1:N̂] <= 1.0)

                # Add split constraints
                for (;node, g, c) in constraints
                    @constraint(model, node.direction * (g' * x + c) >= 0.0)
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
                            cex_input = Zin.Z₁.G * value.(x[1:input_dim]) + Zin.Z₁.c
                            sample_distance = get_sample_distance(N₁, N₂, cex_input)

                            if sample_distance > epsilon
                                @timeit to "LP Found Cex" begin
                                    distance_bound = min(distance_bound, lp_distance_bound)
                                    return false, (cex_input, (N₁(cex_input), N₂(cex_input), sample_distance)), nothing, nothing, distance_bound, nothing
                                end
                            end
                        end

                        if has_values(model)
                            δ = abs(objective_value(model))
                            mask[i, j] &= δ > epsilon
                            lp_distance_bound = max(lp_distance_bound, δ)
                        end

                        if prop_state.num_instables == 0
                            println("Termination Status: $(termination_status(model))")
                        end
                        
                        mask[i, j] &= termination_status(model) != MOI.INFEASIBLE
                    end
                end
            end

            @assert !(prop_state.num_instables == 0 && any(mask))

            distance_bound = min(distance_bound, lp_distance_bound)

        elseif post_contract
            @timeit to "Post-Contract Zono" begin
                
                @timeit to "Sort Constraints" begin
                    sort_constraints!(constraints, zeros(N̂))
                end
                
                @timeit to "Contract Zono All" begin
                    input_bounds = contract_zono_all!([-ones(N̂) ones(N̂)], constraints)
                end
                
                if isnothing(input_bounds)
                    @timeit to "Unsatisfiable" begin
                        return true, nothing, nothing, nothing, distance_bound, nothing
                    end
                end
                
                if !is_unit_hypercube(input_bounds)
                    @timeit to "Transform Zono" begin
                        Zout = transform_offset_diff_zono!(input_bounds, Zout)
                    end
                    prop_satisfied, cex, _, _, p_distance_bound = property_check(N₁, N₂, Zin, Zout, nothing; mask=mask)
                    distance_bound = min(p_distance_bound, distance_bound)

                    if prop_satisfied || !isnothing(cex)
                        return prop_satisfied, cex, nothing, nothing, distance_bound, input_bounds
                    end
                end
            end
        end

        return !any(mask), nothing, nothing, nothing, distance_bound, input_bounds
    end
end

function get_top1_property_with_neuron_splitting(delta::Float64)
    property_check = get_top1_property(;delta=delta)

    approach = NEURON_SPLITTING_APPROACH[]
    contract = ZONO_CONTRACT_MODE[]
    use_lp = approach == LP
    use_zono_contract = approach == ZonoContraction
    use_lp_zc = use_zono_contract && contract == LPZonoContract
    post_contract = use_zono_contract && (contract == ZonoContract || contract == ZonoContractPost)

    return (N₁::Network, N₂::Network, Zin::DiffZonotope, Zout::DiffZonotope, task::VerificationTask, prop_state::PropState) -> begin

        constraints = prop_state.split_constraints
        input_dim = size(Zout.Z₁.G, 2) - Zout.num_approx₁
        N̂ = size(Zout.∂Z.G, 2)
        
        input_bounds = nothing

        # Append zeros to the constraints vectors so that the match the output dimension
        @timeit to "Align Constraints" begin
            for constraint in constraints
                offset = ifelse(constraint.node.network == 1, 0, Zout.num_approx₁)
                constraint.g = align_vector(constraint.g, N̂, input_dim, offset)
            end
        end
        
        if !isempty(constraints)
            if post_contract
                @timeit to "Post-Contract Zono" begin
    
                    @timeit to "Sort Constraints" begin
                        sort_constraints!(constraints, zeros(N̂))
                    end
                    
                    @timeit to "Contract Zono All" begin
                        input_bounds = contract_zono_all!([-ones(N̂) ones(N̂)], constraints)
                    end
                    
                    if isnothing(input_bounds)
                        @timeit to "Unsatisfiable" begin
                            return true, nothing, nothing, nothing, task.distance_bound, nothing
                        end
                    end
                    
                    if !is_unit_hypercube(input_bounds)
                        @timeit to "Transform Zono" begin
                            Zout = transform_offset_diff_zono!(input_bounds, Zout)
                        end
                        @timeit to "Transform Constraints" begin
                            constraints = transform_constraints!(input_bounds, constraints)
                        end
                    end
                end
            end
        end
        
        prop_satisfied, cex, _, verification_status, distance_bound = property_check(N₁, N₂, Zin, Zout, task.verification_status; constraints=constraints)

        @assert !(prop_state.num_instables == 0 && N̂ != input_dim)

        return prop_satisfied, cex, nothing, verification_status, distance_bound, input_bounds
    end
end

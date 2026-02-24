using VeryDiff
using LinearAlgebra
using Random

Random.seed!(4242)

COUNT_FUZZ_TESTING = 0

function fuzz_testing(N₁::Network, N₂::Network, task::VeryDiff.VerificationTask, distance_bound::Float64, queue::VeryDiff.Queue)
    # global COUNT_FUZZ_TESTING += 1
    # Zin = VeryDiff.to_diff_zono(task)
    # δ = distance_bound
    # input_dim = size(Zin.Z₁.G, 2)
    # if !isempty(queue)
    #     _, next_task = first(queue)
    #     δ = max(next_task.distance_bound, distance_bound)
    # end
    # for _ in 1:100
    #     x = Zin.Z₁.G * rand(Float64, input_dim) + Zin.Z₁.c
    #     sample_distance = VeryDiff.get_sample_distance(N₁, N₂, x)
    #     @assert sample_distance <= δ "Input x = $x has a difference distance of $sample_distance which is not within the $δ-bound, seems like a bug."
    # end
end

function get_weighted_random_vector(values, weights::Vector{Int}, size)
    total = sum(weights)
    rand_values = rand(1:total, size)
    decision = cumsum(weights)
    result = map(x -> values[findfirst(decision .>= x)], rand_values)
    return result
end

function create_random_nets()
    layers₁ = Layer[]
    layers₂ = Layer[]
    input_dim = rand(1:50)
    output_dim = 10
    cur_dim = input_dim
    L = rand(2:10)
    networks = Tuple{Network,Network,Int,Int}[]
    for i in 1:L
        if i == L
            new_dim = output_dim
        else
            new_dim = rand(50:100)
        end
        W₁ = rand(Float64, new_dim, cur_dim)
        b₁ = rand(Float64, new_dim)
        # net_choice = get_weighted_random_vector([1, 2, 3], [1, 1, 1], 1)[1]

        # println("NET CHOICE: $(net_choice)")
        # if net_choice == 1 # Independent weights
        #     W₂ = randn(Float64, (new_dim,cur_dim))
        #     b₂ = randn(Float64, new_dim)
        #     zero_one_rows1 = get_weighted_random_vector([0.0, 1.0], [1, 9], new_dim)
        #     simple_rows = get_weighted_random_vector([0.0, 1.0], [4, 6], new_dim) .* rand([-1.0, 0.0, 1.0],(new_dim, cur_dim))
        #     W₁ = W₁ .* zero_one_rows1 .+ (1 .- zero_one_rows1) .* simple_rows
        #     zero_one_rows2 = get_weighted_random_vector([0.0, 1.0], [1, 9], new_dim)
        #     simple_rows = get_weighted_random_vector([0.0, 1.0],[1, 9], new_dim) .* rand([-1.0, 0.0, 1.0], (new_dim, cur_dim))
        #     W₂ = W₂ .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
        #     b₁ = b₁ .* zero_one_rows1
        #     b₂ = b₂ .* zero_one_rows2
        # elseif net_choice == 2 # Pruned weights
        #     W₂ = copy(W₁)
        #     b₂ = copy(b₁)
        #     zero_one_rows2 = get_weighted_random_vector([0.0, 1.0], [1, 9], new_dim)
        #     simple_rows = get_weighted_random_vector([0.0, 1.0], [1, 9], new_dim) .* rand([-1.0, 0.0, 1.0], (new_dim, cur_dim))
        #     W₂ = W₂ .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
        #     b₂ = b₂ .* zero_one_rows2
        # else
        W₂ = W₁ + 1e-20 * randn(Float64, (new_dim, cur_dim))
        b₂ = b₁ + 1e-20 * randn(Float64, new_dim)
        # end

        push!(layers₁, Dense(W₁, b₁))
        push!(layers₂, Dense(W₂, b₂))
        push!(layers₁, ReLU())
        push!(layers₂, ReLU())

        push!(networks, (Network(deepcopy(layers₁)), Network(deepcopy(layers₂)), input_dim, new_dim))

        cur_dim = new_dim
    end

    return networks
end

function run_tests()
    global COUNT_FUZZ_TESTING = 0
    VeryDiff.set_neuron_splitting_config((true, false, false, true); mode=VeryDiff.DeepSplitUnbiased, approach=VeryDiff.LP, contract=VeryDiff.LPZonoContract)
    println("\nUsing $(VeryDiff.get_config())...\n")

    net_count = 0
    while true
        next_seed = rand(1:99999)
        Random.seed!(next_seed)
        Random.seed!(9064)
        # Random.seed!(1147)
        # Random.seed!(101020)
        # Random.seed!(2917)
        # Random.seed!(70832)
        # Random.seed!(92369)
        # Random.seed!(70976)
        # Random.seed!(65447)
        println("[FUZZER] SEED: $(next_seed)")
        networks = create_random_nets()
        net_count += 1
        println("[FUZZER] NET COUNT: $(net_count)")

        for (N₁, N₂, input_dim, output_dim) in networks
            VeryDiff.reset_timer!(VeryDiff.to)
            N = GeminiNetwork(N₁, N₂)

            range = rand([0.1, 1.0, 2.0, 2.5, 3.0, 4.0])
            # range = rand([0.1, 4.0])
            offset = rand(input_dim)
            bounds = [(offset .- range) (offset .+ range)]
            lower = @view bounds[:, 1]
            upper = @view bounds[:, 2]
            mid = (upper .+ lower) ./ 2
            distance = mid .- lower
            non_zero_indices = findall((!).(iszero.(distance)))
            distance = distance[non_zero_indices]

            ∂Z = Zonotope(Matrix(0.0I, input_dim, size(non_zero_indices, 1)), zeros(Float64, input_dim), nothing)
            initial_task = VeryDiff.VerificationTask(mid, distance, non_zero_indices, ∂Z, nothing, Inf64, Branch())

            ϵ = 1.0e-10
            # ϵ = 1.0e-0
            property_check = get_epsilon_property(ϵ)

            verify_func = deepsplit_verify_network(ϵ; fuzz_testing=fuzz_testing)
            status, cex, δ_bounds = verify_func(N, N₁, N₂, initial_task, deepsplit_heuristic; timeout=30)
            # status, δ_bounds = verify_network(N₁, N₂, bounds, property_check, VeryDiff.epsilon_split_heuristic; timeout=30)

            show(VeryDiff.to)
            println("\nStatus: $status")
            println("\nδ_bounds: $(δ_bounds)")
            if !isnothing(cex)
                println("\nFound Counterexample: $(cex)")
            end
            # break
        end
        break
    end
    println("[FUZZER] TESTING COUNT: $(COUNT_FUZZ_TESTING)")
end


# function get_top1_property(; delta=zero(Float64), naive=false)
#     if !iszero(delta)
#         @assert 0.5 <= delta && delta <= 1.0
#         dist = log(delta / (1 - delta))
#     else
#         dist = 0.0
#     end
#     return (N1, N2, Zin, Zout, verification_status) -> begin
#         global FIRST_ROUND
#         global TOP1_FOUND_CONCRETE_DELTA
#         if FIRST_ROUND
#             TOP1_FOUND_CONCRETE_DELTA = false
#         end
#         if isnothing(verification_status)
#             verification_status = Dict{Tuple{Int,Int},Bool}()
#         end
#         input_dim = size(Zout.Z₂, 2) - Zout.num_approx₂
#         top_dimension_violation = zeros(input_dim) #size(Zout.Z₁.G,1))
#         res1 = N1(Zin.Z₁.c)
#         argmax_N1 = argmax(res1)
#         argmax_N2 = argmax(N2(Zin.Z₂.c))
#         softmax_N1 = exp.(res1) / sum(exp.(res1))
#         if argmax_N1 != argmax_N2
#             if iszero(delta) || softmax_N1[argmax_N1] >= delta
#                 println("Found cex")
#                 println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
#                 return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
#             else
#                 second_largest = sort(res1, rev=true)[2]
#                 if !iszero(delta) && res1[argmax_N1] - second_largest >= dist
#                     println("Found spurious cex")
#                     println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
#                     println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
#                 end
#             end
#         end
#         property_satisfied = true
#         distance_bound = 0.0
#         # Formulation of "Description of Z₁-∂Z = Z₂" in LP:
#         #
#         # n2_G = -Zout.∂Z.G
#         # n2_G[:,1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G
#         # n2_G[:,1:input_dim] .-= Zout.Z₂.G[:,1:input_dim]
#         # z2_approx_start = (size(Zout.Z₁.G,2)+1)
#         # z2_approx_end = z2_approx_start + Zout.num_approx₂ - 1
#         # n2_G[:,z2_approx_start:z2_approx_end] .-= Zout.Z₂.G[:,(input_dim+1):end]
#         # n2_c = -(Zout.Z₁.c .- Zout.∂Z.c) + Zout.Z₂.c

#         # δ-Top-1 property
#         any_feasible = false
#         for top_index in 1:size(Zout.Z₁, 1)
#             # TODO: Construct LP that ensures that top_index is maximal in N1
#             G1 = Zout.Z₁.G .- Zout.Z₁.G[top_index:top_index, :]
#             c1 = Zout.Z₁.c[top_index] .- Zout.Z₁.c

#             model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))

#             set_time_limit_sec(model, 10)
#             var_num = size(Zin.Z₁.G, 2) + Zout.num_approx₁ + Zout.num_approx₂ + Zout.∂num_approx
#             @variable(model, -1.0 <= x[1:var_num] <= 1.0)

#             # Additional (but not that helpful) constraints
#             #@constraint(model,G2*x .<= c2)
#             #@constraint(model, n2_G*x == n2_c)
#             # Constraint Formulation for:
#             # Z₁ = Z₂ + ∂Z <-> G₁*x+c₁ = (G₂+∂G)*x+(c₂+∂c)
#             # <-> c₁ - c₂ - ∂c = (G₂+∂G-G₁)*x

#             if !naive
#                 G2 = copy(Zout.∂Z.G)
#                 offset = 1
#                 G2[:, offset:input_dim] .+= Zout.Z₂.G[:, 1:input_dim] .- Zout.Z₁.G[:, 1:input_dim]
#                 offset += input_dim
#                 G2[:, offset:(offset+Zout.num_approx₁-1)] .-= Zout.Z₁.G[:, (input_dim+1):end]
#                 offset += Zout.num_approx₁
#                 G2[:, offset:(offset+Zout.num_approx₂-1)] .+= Zout.Z₂.G[:, (input_dim+1):end]
#                 @constraint(model,
#                     G2 * x .== (Zout.Z₁.c .- Zout.∂Z.c .- Zout.Z₂.c)
#                 )
#             end

#             @constraint(model, G1[1:end.!=top_index, :] * x[1:size(Zout.Z₁.G, 2)] .<= (c1[1:end.!=top_index] .- dist))
#             @objective(model, Max, 0)
#             optimize!(model)

#             if termination_status(model) == MOI.INFEASIBLE
#                 for other_index in 1:size(Zout.Z₁, 1)
#                     verification_status[(top_index, other_index)] = true
#                 end
#             else

#                 if !iszero(delta) && !TOP1_FOUND_CONCRETE_DELTA
#                     input = Zin.Z₁.G * value.(x[1:input_dim]) + Zin.Z₁.c
#                     res1 = N1(input)
#                     argmax_N1 = argmax(res1)
#                     softmax_N1 = exp.(res1) / sum(exp.(res1))
#                     if softmax_N1[argmax_N1] >= delta
#                         println("[TOP-1] required confidence ($(softmax_N1[argmax_N1])≥$delta) is feasible for index $argmax_N1")
#                         TOP1_FOUND_CONCRETE_DELTA = true
#                     else
#                         println("[TOP-1] did not find required confidence yet.")
#                     end
#                 end
#                 any_feasible = true
#                 for other_index in 1:size(Zout.Z₁, 1)
#                     if other_index != top_index && !haskey(verification_status, (top_index, other_index))
#                         a = zeros(var_num)
#                         input_dim = size(Zin.Z₂.G, 2)
#                         a[1:input_dim] .= Zout.Z₂.G[other_index, 1:input_dim] .- Zout.Z₂.G[top_index, 1:input_dim]
#                         offset = input_dim + Zout.num_approx₁ + 1
#                         a[offset:(offset+Zout.num_approx₂-1)] .= Zout.Z₂.G[other_index, (input_dim+1):end] .- Zout.Z₂.G[top_index, (input_dim+1):end]
#                         @objective(model, Max, a' * x)
#                         violation_difference = a

#                         threshold = Zout.Z₂.c[top_index] - Zout.Z₂.c[other_index]
#                         # If the optimal value is < threshold, then the property is satisfied
#                         # otherwise (optimal >= threshold) we may have found a counterexample
#                         if USE_GUROBI # we are using GUROBI -> set objective/bound thresholds
#                             set_optimizer_attribute(model, "Cutoff", threshold - 1e-6)
#                         end
#                         optimize!(model)

#                         model_status = termination_status(model)
#                         # Model must be feasible since we did not add any constraints
#                         @assert model_status != MOI.INFEASIBLE
#                         # Model should be optimal or have reached the objective limit
#                         # any other status -> split and retry
#                         if model_status != MOI.OPTIMAL && model_status != MOI.OBJECTIVE_LIMIT
#                             println("[GUROBI] Irregular model status: $model_status")
#                             top_dimension_violation .+= abs.(violation_difference[1:input_dim])
#                             property_satisfied = false
#                             if has_values(model)
#                                 distance_bound = max(distance_bound, objective_value(model))
#                             end
#                             continue
#                         end
#                         if model_status == MOI.OBJECTIVE_LIMIT || objective_value(model) < threshold
#                             verification_status[(top_index, other_index)] = true
#                         else
#                             distance_bound = max(distance_bound, objective_value(model))
#                             input = Zin.Z₁.G * value.(x[1:input_dim]) + Zin.Z₁.c
#                             res1 = N1(input)
#                             res2 = N2(input)
#                             argmax_N1 = argmax(res1)
#                             argmax_N2 = argmax(res2)
#                             softmax_N1 = exp.(res1) / sum(exp.(res1))
#                             if argmax_N1 != argmax_N2
#                                 if iszero(delta) || softmax_N1[argmax_N1] >= delta
#                                     println("Found cex")
#                                     second_most = sort(softmax_N1, rev=true)[2]
#                                     println("N1: $(softmax_N1[argmax_N1]) (vs. $second_most)")
#                                     softmax_N2 = exp.(res2) / sum(exp.(res2))
#                                     println("N2: $(softmax_N2[argmax_N2])")
#                                     println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
#                                     return false, (input, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
#                                 else
#                                     second_largest = sort(res1, rev=true)[2]
#                                     if !iszero(delta) && res1[argmax_N1] - second_largest >= dist
#                                         println("Found spurious cex")
#                                         println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
#                                         println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
#                                     end
#                                     top_dimension_violation .+= abs.(violation_difference[1:input_dim])
#                                     property_satisfied = false
#                                 end
#                             else
#                                 top_dimension_violation .+= abs.(violation_difference[1:input_dim])
#                                 property_satisfied = false
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#         @assert !iszero(delta) || any_feasible "One output must be maximal, but our analysis says there is no maximum -- this smells like a bug!"
#         return property_satisfied, nothing, top_dimension_violation, verification_status, distance_bound
#     end
# end

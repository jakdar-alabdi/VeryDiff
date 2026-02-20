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
    networks = Tuple{Network, Network, Int, Int}[]
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
    VeryDiff.set_neuron_splitting_config((true, false, true, true); mode=VeryDiff.DeepSplitUnbiased, approach=VeryDiff.LP, contract=VeryDiff.LPZonoContract)
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

            # verify_func = deepsplit_verify_network(ϵ; fuzz_testing=fuzz_testing)
            # status, cex, δ_bounds = verify_func(N, N₁, N₂, initial_task, deepsplit_heuristic; timeout=30)
            status, δ_bounds = verify_network(N₁, N₂, bounds, property_check, VeryDiff.epsilon_split_heuristic; timeout=30)

            show(VeryDiff.to)
            println("\nStatus: $status")
            println("\nδ_bounds: $(δ_bounds)")
            # if !isnothing(cex)
            #     println("\nFound Counterexample: $(cex)")
            # end
            # break
        end
        break
    end
    println("[FUZZER] TESTING COUNT: $(COUNT_FUZZ_TESTING)")
end

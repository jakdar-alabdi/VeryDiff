using VeryDiff
using LinearAlgebra
using JuMP, Gurobi
using Random

VeryDiff.NEW_HEURISTIC = false

function get_weighted_random_vector1(values, weights::Vector{Int},size)
    total = sum(weights)
    rand_values = rand(1:total,size)
    decision = cumsum(weights)
    result = map(x -> values[findfirst(decision .>= x)],rand_values)
    return result
end

NET_COUNT = 0
verydiff_status = VeryDiff.UNKNOWN
deepsplit_status = VeryDiff.UNKNOWN

while verydiff_status == deepsplit_status
    global NET_COUNT
    next_seed = rand(1:99999999999)
    Random.seed!(next_seed);
    Random.seed!(93277004881)
    println("[FUZZER] SEED: $(next_seed)")
    NET_COUNT += 1
    println("[FUZZER] NET COUNT: $(NET_COUNT)")
    
    ϵ = 1.0e-1
    num_layers = rand(2:50)
    input_dim = rand(1:50)
    output_dim = 10
    cur_dim = input_dim
    layers1 = Layer[]
    layers2 = Layer[]

    for l in 1:num_layers
        if l == num_layers
            new_dim = output_dim
        else
            new_dim = rand(50:100)
        end

        W1 = randn(Float64, new_dim, cur_dim)
        b1 = randn(Float64, new_dim)
        # net_choice = get_weighted_random_vector1([1,2,3],[1,1,1],1)[1]
        # # println("NET CHOICE: $(net_choice)")
        # if net_choice == 1 # Independent weights
        #     W2 = randn(Float64,(new_dim,cur_dim))
        #     b2 = randn(Float64,new_dim)
        #     zero_one_rows1 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
        #     simple_rows = get_weighted_random_vector1([0.0,1.0],[4,6],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
        #     W1 = W1 .* zero_one_rows1 .+ (1 .- zero_one_rows1) .* simple_rows
        #     zero_one_rows2 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
        #     simple_rows = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
        #     W2 = W2 .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
        #     b1 = b1 .* zero_one_rows1
        #     b2 = b2 .* zero_one_rows2
        # elseif net_choice == 2 # Pruned weights
        #     W2 = copy(W1)
        #     b2 = copy(b1)
        #     zero_one_rows2 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
        #     simple_rows = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
        #     W2 = W2 .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
        #     b2 = b2 .* zero_one_rows2
        # else
            W2 = W1 + 1e-20*randn(Float64,(new_dim,cur_dim))
            b2 = b1 + 1e-20*randn(Float64,new_dim)
        # end

        push!(layers1, Dense(W1, b1))
        push!(layers1, ReLU())
        push!(layers2, Dense(W2, b2))
        push!(layers2, ReLU())

        cur_dim = new_dim
    end

    input_range = rand([0.1, 4.0])
    offset = rand(input_dim)

    property_check = get_epsilon_property(ϵ)

    Z = Zonotope(Matrix(input_range * I, input_dim, input_dim), offset, nothing)
    bounds = zono_bounds(Z)
    N₁ = Network(layers1)
    N₂ = Network(layers2)

    # verydiff_status = verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic)

    # println("\nVerDiff status: $verydiff_status")

    println("\n###############################################################################################")

    deepsplit_status = deepsplit_lp_search_epsilon(N₁, N₂, bounds, ϵ)

    println("DeepSplit status: $deepsplit_status")
end

# sysimage_dir = @__DIR__

# run_cmd([
#     "--neuron-splitting", 
#     "--epsilon", "0.005",
#     "$sysimage_dir\\..\\test\\examples\\nets\\ACASXU_run2a_1_1_batch_2000.onnx",
#     "$sysimage_dir\\..\\test\\examples\\nets\\ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
#     "$sysimage_dir\\..\\test\\examples\\specs\\prop_1.vnnlib"
# ])

# println("\n###############################################################################################")

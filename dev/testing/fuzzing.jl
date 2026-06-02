# using VeryDiff
using Random
using Gurobi, JuMP

include("random_networks.jl")

function fuzz_testing(N₁::Network, N₂::Network, task::VeryDiff.VerificationTask, distance_bound::Float64; distance_metric=nothing, num_samples=100)
    (;distance, middle, distance_indices) = task
    lower = middle[distance_indices] - distance
    upper = middle[distance_indices] + distance
    width = upper - lower
    input_dim = length(lower)
    @info "Distance Bound: $distance_bound"
    next_seed = rand(1:999999)
    # @info "Next seed: $(next_seed)"
    Random.seed!(next_seed)
    x = zeros(input_dim)
    for _ in 1:num_samples
        x .= clamp.(lower .+ width .* rand(input_dim), lower, upper)
        # @info "Random input: $x"
        sample_distance = distance_metric(N₁, N₂, x)
        # @info "Sample distance: $sample_distance"
        # @assert sample_distance <= distance_bound || isapprox(sample_distance, distance_bound; atol=1e-6) "Found counterexample $(x) with sample distance $(sample_distance) > $distance_bound."
        @assert sample_distance <= distance_bound "Found counterexample $(x) with sample distance $(sample_distance) > $distance_bound."
    end
end

function start_fuzz_testing()
    VeryDiff.NEW_HEURISTIC[] = false
    # Random.seed!(42)
    # Random.seed!(169370)
    # Random.seed!(986905)
    # Random.seed!(299535)
    # Random.seed!(250891)
    Random.seed!(1234)
    timeout = 30
    num_iter = 0
    max_iters = 100

    fuzz_testing_func = nothing
    if VeryDiff.NEW_HEURISTIC[]
        VeryDiff.set_neuron_splitting_config((false, false, false))
        property_check_func = VeryDiff.get_epsilon_property
        verifier = verify_network
    else
        fuzz_testing_func = fuzz_testing
        VeryDiff.set_neuron_splitting_config((true, false, false); mode=VeryDiff.DeepSplitUnbiased, approach=VeryDiff.VerticalSplitting, contract=VeryDiff.ZonoContractInter)
        property_check_func = VeryDiff.get_epsilon_property_with_neuron_splitting
        verifier = deepsplit_verify_network
    end

    if !isdir(out_dir)
        mkdir(out_dir)
    end

    println(VeryDiff.get_config())
    while num_iter < max_iters
        num_iter += 1
        println("Iteration: $num_iter")
        next_seed = rand(1:999999)
        @info "Next seed: $(next_seed)"
        Random.seed!(next_seed)
        # Random.seed!(777864)

        num_layers = 3 * rand(2:10)
        @info "Num layers: $num_layers"
        input_dim = rand(2:50)
        @info "Input dimension: $input_dim"
        output_dim = rand(2:50)
        @info "Output dimension: $output_dim"

        N₁, N₂ = create_random_networks(num_layers, input_dim, output_dim)

        input_center = randn(input_dim)
        input_radius = rand(input_dim)
        bounds = [(input_center - input_radius) (input_center + input_radius)]

        epsilon = abs(randn())
        property_check = property_check_func(epsilon)

        if VeryDiff.NEW_HEURISTIC[]
            veri_result = verifier(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=timeout)
        else
            veri_result = verifier(N₁, N₂, bounds, property_check; timeout=timeout, fuzz_testing=fuzz_testing)
        end

        println(veri_result)
        @info "ϵ: $epsilon"
        @info "Input bounds: $bounds"
        println("\n-------------------------------------------------------------")
        flush(stdout)
        flush(stderr)
        GC.gc()
    end
end

    
original_stdout = stdout
original_stderr = stderr
out_dir = "$(@__DIR__)/fuzzing_logs/"
open(joinpath(out_dir, "fuzzing.log"), "w") do f
    redirect_stdout(f)
    redirect_stderr(f)
    try
        start_fuzz_testing()
    catch e
        showerror(stdout, e, catch_backtrace())
    end
    redirect_stdout(original_stdout)
    redirect_stderr(original_stderr)
end

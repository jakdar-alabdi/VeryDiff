using VeryDiff
using Random
using Gurobi, JuMP

include("random_networks.jl")


Random.seed!(1042)

function fuzz_testing(N₁::Network, N₂::Network, Zin::VeryDiff.Zonotope, distance_bound::Float64; distance_metric=nothing, num_samples=100)
    bounds = VeryDiff.zono_bounds(Zin)
    lower = @view bounds[:, 1]
    upper = @view bounds[:, 2]
    width = upper - lower
    input_dim = length(lower)
    num_samples = 10_000#rand(1000:100_000)
    x = zeros(input_dim)
    for _ in 1:num_samples
        x .= clamp.(lower .+ width .* rand(input_dim), lower, upper)
        sample_distance = distance_metric(N₁, N₂, x)
        @assert sample_distance <= distance_bound "Found counterexample $(x) with sample distance $(sample_distance) > $distance_bound."
    end
end

function start_fuzz_testing()
    VeryDiff.NEW_HEURISTIC[] = false
    timeout = 120
    num_iter = 0
    max_iters = 100_000

    fuzz_testing_func = nothing
    if VeryDiff.NEW_HEURISTIC[]
        VeryDiff.set_neuron_splitting_config(
            (false, false, false), 
            (false, false, false),
        )
        property_check_func = VeryDiff.get_epsilon_property
        verifier = verify_network
    else
        fuzz_testing_func = fuzz_testing
        VeryDiff.set_neuron_splitting_config(
            (true, false, false), 
            (false, false, false), 
            VeryDiff.DeepSplitUnbiased, 
            VeryDiff.ZonoContraction, 
            VeryDiff.LPZonoContract
        )
        property_check_func = VeryDiff.get_epsilon_property_with_neuron_splitting
        verifier = deepsplit_verify_network
    end

    original_stdout = stdout
    original_stderr = stderr
    out_dir = "$(@__DIR__)/fuzzing_logs/"

    if !isdir(out_dir)
        mkdir(out_dir)
    end

    open(joinpath(out_dir, "fuzzing.log"), "w") do f
        redirect_stdout(f)
        redirect_stderr(f)
        println(VeryDiff.get_config())
        while num_iter < max_iters
            num_iter += 1
            println("Iteration: $num_iter")
            next_seed = rand(1:999999)
            @info "Next seed: $(next_seed)"
            Random.seed!(next_seed)

            num_layers = 3 * rand(2:10)
            @info "Num layers: $num_layers"
            input_dim = rand(5:150)
            @info "Input dimension: $input_dim"
            output_dim = rand(5:50)
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
        flush(stdout)
        flush(stderr)
        GC.gc()
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
    end
end

# start_fuzz_testing()

# using VeryDiff
using Random
using Gurobi, JuMP

include("random_networks.jl")

function fuzz_testing(N₁::Network, N₂::Network, prop_state::PropState, provable_distance_bound::Float64; distance_metric=nothing)
    Zin = prop_state.zono_storage.zonotopes[1].zonotope
    Zout = prop_state.zono_storage.zonotopes[end].zonotope
    bounds = zono_bounds(Zin.Z₁)
    lower = @view bounds[:, 1]
    upper = @view bounds[:, 2]
    width = upper - lower
    input_dim = length(lower)
    # @info "Provable distance bound: $provable_distance_bound"
    next_seed = rand(1:999999)
    # @info "Next seed: $(next_seed)"
    Random.seed!(next_seed)
    x = zeros(input_dim)
    for _ in 1:100
        x .= lower .+ width .* rand(input_dim)
        # @info "Random input: $x"
        sample_distance = distance_metric(N₁, N₂, x)
        # @info "Sample distance: $sample_distance"
        @assert sample_distance <= provable_distance_bound
    end
    
    if VeryDiff.USE_VERTICAL_SPLITTING[]
        split_nodes = filter(node -> node.direction == -1 && !isnothing(node.bounds), prop_state.task.branch.split_nodes)
        if !isempty(split_nodes)
            model = Model(() -> Gurobi.Optimizer(VeryDiff.Properties.GRB_ENV[]))
            set_time_limit_sec(model, 10)
            
            xs = [@variable(model, [1:size(G, 2)], lower_bound=-1.0, upper_bound=1.0) for G in Zout.∂Z.Gs]

            for node in split_nodes
                (;network, neuron, diff_layer, direction, bounds) = node
                Z = VeryDiff.get_split_node_zono(node, prop_state)
                indices = VeryDiff.intersect_indices(Zout.∂Z.generator_ids, Z.generator_ids)
                affine_repr = AffExpr(Z.c[neuron])
                for (G, i) in zip(Z.Gs, indices)
                    add_to_expression!(affine_repr, G[neuron, :]'xs[i][1:size(G, 2)])
                end
                @constraint(model, affine_repr >= bounds[2, 1])
            end

            @objective(model, Max, 0)
            optimize!(model)

            if termination_status(model) != MOI.INFEASIBLE
                for i in 1:length(Zout.∂Z.c)
                    for σ in [-1, 1]
                        @objective(model, Max, σ * (sum(G[i, :]'xs[k] for (k, G) in enumerate(Zout.∂Z.Gs)) + Zout.∂Z.c[i]))
                        optimize!(model)                
                        if is_solved_and_feasible(model)
                            val = value.(xs[1])
                            cex_input = Zin.Z₁.Gs[1] * val + Zin.Z₁.c
                            sample_distance = distance_metric(N₁, N₂, cex_input)                
                            @assert sample_distance <= provable_distance_bound "Found counterexample $(cex_input) with sample distance $(sample_distance) and LP value $(val), this seems like a bug."
                        end
                    end
                end
            end
        end
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

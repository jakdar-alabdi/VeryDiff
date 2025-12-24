function parse_networks(nn_file₁::String, nn_file₂::String)
    println("Parsing $(basename(nn_file₁))...")
    N₁ = parse_network(load_network(nn_file₁))
    println("Parsing $(basename(nn_file₂))...")
    N₂ = parse_network(load_network(nn_file₂))
    return N₁, N₂
end

function save_results(out_dir::String, spec::SpecificationEpsilon, status::VeryDiff.VerificationStatus, δ_bounds::Tuple{Float64, Float64}, to::TimerOutputs.TimerOutput)
    net_name = replace(basename(spec.nn_file₂), ".onnx" => "")
    spec_name = replace(basename(spec.spec_file), ".vnnlib" => "")
    runtime, num_propagations, num_input_splits, num_neuron_splits = 0, 0, 0, 0
    if "Verify" in keys(to.inner_timers)
        inner = to["Verify"]
        runtime = inner.accumulated_data.time
        if "Zonotope Loop" in keys(inner.inner_timers)
            inner = inner["Zonotope Loop"]
            if "Zonotope Propagate" in keys(inner.inner_timers)
                num_propagations = inner["Zonotope Propagate"].accumulated_data.ncalls
            end
            if "Compute Split" in keys(inner.inner_timers)
                inner = inner["Compute Split"]
                if "Split Input" in keys(inner.inner_timers)
                    num_input_splits = inner["Split Input"].accumulated_data.ncalls
                end
                if "Split Neuron" in keys(inner.inner_timers)
                    num_neuron_splits = inner["Split Neuron"].accumulated_data.ncalls
                end
                if num_input_splits == 0 && num_neuron_splits == 0
                    num_input_splits = inner.accumulated_data.ncalls
                end
            end
        end
    end
    open(out_dir, "a") do c
        redirect_stdout(c) do
            redirect_stderr(c) do
                println("$net_name, $spec_name, $status, $runtime, $num_propagations, $num_input_splits, $num_neuron_splits, $(δ_bounds[1]), $(δ_bounds[2])")
            end
        end
    end
end

function verydiff(spec::SpecificationEpsilon, result_out_dir::String)
    (;nn_file₁, nn_file₂, spec_file, epsilon, timeout) = spec
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = get_epsilon_property(epsilon)
    println("Using VeryDiff...")
    set_deepsplit_config((false, false, false, false))
    for (bounds, _, _, _) in f
        status, δ_bounds = verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=1)
        save_results(result_out_dir, spec, status, δ_bounds, VeryDiff.to)
    end
end

function deepsplit(config::Tuple{Bool, Bool, Bool, Bool})
    return (spec::SpecificationEpsilon, result_out_dir::String) -> begin
        (;nn_file₁, nn_file₂, spec_file, epsilon, timeout) = spec
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        println("Using DeepSplit...")
        set_deepsplit_config(config)
        for (bounds, _, _, _) in f
            status, δ_bounds = deepsplit_lp_search_epsilon(N₁, N₂, bounds, epsilon; timeout=1)
            save_results(result_out_dir, spec, status, δ_bounds, VeryDiff.to)
        end
    end
end

function run_experiments()
    println("Running acas all...")
    
    # println("\nRunning on VeryDiff")
    # run_acas_all(verydiff, "VeryDiff")
    # println("\nRunning on DeepSplit-Base")
    # run_acas_all(deepsplit((true, false, false, false)), "DeepSplit-Base")
    # println("\nRunning on DeepSplit-Alt")
    # run_acas_all(deepsplit((true, false, false, false)), "DeepSplit-Alt")
    println("\nRunning on DeepSplit-Base-Input")
    run_acas_all(deepsplit((true, false, false, true)), "DeepSplit-Base-Input")
    # println("\nRunning on DeepSplit-Base-DiffZono")
    # run_acas_all(deepsplit((true, false, true, false)), "DeepSplit-Base-DiffZono")
    println("\nRunning on DeepSplit-Base-Input-DiffZono")
    run_acas_all(deepsplit((true, false, true, true)), "DeepSplit-Base-Input-DiffZono")
    # println("\nRunning on DeepSplit-Alt-Input")
    # run_acas_all(deepsplit((true, false, false, true)), "DeepSplit-Alt-Input")
    # println("\nRunning on DeepSplit-Alt-DiffZono")
    # run_acas_all(deepsplit((true, false, true, false)), "DeepSplit-Alt-DiffZono")
    # println("\nRunning on DeepSplit-Alt-Input-DiffZono")
    # run_acas_all(deepsplit((true, false, true, true)), "DeepSplit-Alt-Input-DiffZono")

    println("Running mnist all...")
    # println("\nRunning on VeryDiff")
    # run_mnist_all(verydiff, "VeryDiff")
    # println("\nRunning on DeepSplit-Base")
    # run_mnist_all(deepsplit((true, false, false, false)), "DeepSplit-Base")
    # println("\nRunning on DeepSplit-Alt")
    # run_mnist_all(deepsplit((true, false, false, false)), "DeepSplit-Alt")
    println("\nRunning on DeepSplit-Base-Input")
    run_mnist_all(deepsplit((true, false, false, true)), "DeepSplit-Base-Input")
    # println("\nRunning on DeepSplit-Base-DiffZono")
    # run_mnist_all(deepsplit((true, false, true, false)), "DeepSplit-Base-DiffZono")
    println("\nRunning on DeepSplit-Base-Input-DiffZono")
    run_mnist_all(deepsplit((true, false, true, true)), "DeepSplit-Base-Input-DiffZono")
    # println("\nRunning on DeepSplit-Alt-Input")
    # run_mnist_all(deepsplit((true, false, false, true)), "DeepSplit-Alt-Input")
    # println("\nRunning on DeepSplit-Alt-DiffZono")
    # run_mnist_all(deepsplit((true, false, true, false)), "DeepSplit-Alt-DiffZono")
    # println("\nRunning on DeepSplit-Alt-Input-DiffZono")
    # run_mnist_all(deepsplit((true, false, true, true)), "DeepSplit-Alt-Input-DiffZono")
end

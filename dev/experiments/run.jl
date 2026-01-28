function parse_networks(nn_file₁::String, nn_file₂::String)
    println("Parsing $(basename(nn_file₁))...")
    N₁ = parse_network(load_network(nn_file₁))
    println("Parsing $(basename(nn_file₂))...")
    N₂ = parse_network(load_network(nn_file₂))
    return N₁, N₂
end

function save_results(out_dir::String, net_name::String, spec_name::String, status::VeryDiff.VerificationStatus, δ_bounds::Tuple{Float64, Float64}, to::TimerOutputs.TimerOutput)
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

    original_stdout = stdout
    original_stderr = stderr
    open(out_dir, "a") do c
        redirect_stdout(c)
        redirect_stderr(c)
        flush(stdout)
        flush(stderr)
        println("$net_name, $spec_name, $status, $runtime, $num_propagations, $num_input_splits, $num_neuron_splits, $(δ_bounds[1]), $(δ_bounds[2])")
        flush(stdout)
        flush(stderr)
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
    end
end

function verydiff(nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true)
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = get_epsilon_property(epsilon)
    set_neuron_splitting_config((false, false, false, false))
    VeryDiff.NEW_HEURISTIC = true
    println("Using $(VeryDiff.get_config())...")
    for (bounds, _, _, _) in f
        status, δ_bounds = verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, status, δ_bounds, VeryDiff.to)
        end
    end
end

function deepsplit(config::Tuple{Bool, Bool, Bool, Bool}; mode=ZonoBiased, approach=LP, contract=ZonoContract)
    return (nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        set_neuron_splitting_config(config; mode=mode, approach=approach, contract=contract)
        println("Using $(VeryDiff.get_config())...")
        for (bounds, _, _, _) in f
            status, δ_bounds = deepsplit_lp_search_epsilon(N₁, N₂, bounds, epsilon; timeout=timeout)
            net_name = replace(basename(nn_file₂), ".onnx" => "")
            spec_name = replace(basename(spec_file), ".vnnlib" => "")
            if save
                save_results(result_out_dir, net_name, spec_name, status, δ_bounds, VeryDiff.to)
            end
        end
    end
end

function run_experiments()
    println("\nRunning MNIST all...")
    run_mnist_all(deepsplit((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Base")
    run_mnist_all(deepsplit((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Input-DiffZono")
    
    run_mnist_all(deepsplit((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Base")
    run_mnist_all(deepsplit((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Input-DiffZono")

    run_mnist_all(deepsplit((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPre), "ZonoContractPre-DU-Base")
    run_mnist_all(deepsplit((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPre), "ZonoContractPre-DU-Input-DiffZono")

    run_mnist_all(deepsplit((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Base")
    run_mnist_all(deepsplit((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Input-DiffZono")
end

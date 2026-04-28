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

function verydiff_epsilon(nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true)
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

function verydiff_top1(nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true)
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = get_top1_property(;delta=delta)
    set_neuron_splitting_config((false, false, false, false))
    VeryDiff.NEW_HEURISTIC = true
    global FIRST_ROUND = true
    global TOP1_FOUND_CONCRETE_DELTA = false
    println("Using $(VeryDiff.get_config())...")
    for (bounds, _, _, _) in f
        status, δ_bounds = verify_network(N₁, N₂, bounds, property_check, top1_configure_split_heuristic(1); timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, status, δ_bounds, VeryDiff.to)
        end
    end
end

function deepsplit_epsilon(config::Tuple{Bool, Bool, Bool, Bool}; mode=ZonoBiased, approach=LP, contract=ZonoContract)
    return (nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        set_neuron_splitting_config(config; mode=mode, approach=approach, contract=contract)
        property_check = get_epsilon_property_with_neuron_splitting(epsilon)
        println("Using $(VeryDiff.get_config())...")
        for (bounds, _, _, _) in f
            status, δ_bounds = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout)
            net_name = replace(basename(nn_file₂), ".onnx" => "")
            spec_name = replace(basename(spec_file), ".vnnlib" => "")
            if save
                save_results(result_out_dir, net_name, spec_name, status, δ_bounds, VeryDiff.to)
            end
        end
    end
end

function deepsplit_top1(config::Tuple{Bool, Bool, Bool, Bool}; mode=ZonoBiased, approach=LP, contract=ZonoContract)
    return (nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        set_neuron_splitting_config(config; mode=mode, approach=approach, contract=contract)
        property_check = get_top1_property_with_neuron_splitting(delta)
        println("Using $(VeryDiff.get_config())...")
        global FIRST_ROUND = true
        global TOP1_FOUND_CONCRETE_DELTA = false
        for (bounds, _, _, _) in f
            status, δ_bounds = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout)
            net_name = replace(basename(nn_file₂), ".onnx" => "")
            spec_name = replace(basename(spec_file), ".vnnlib" => "")
            if save
                save_results(result_out_dir, net_name, spec_name, status, δ_bounds, VeryDiff.to)
            end
        end
    end
end

function run_experiments_pool4()    
    # println("\nRunning ACAS all...")

    # run_acas_all_epsilon(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Base")
    # run_acas_all_epsilon(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Input-DiffZono")

    # run_acas_all_epsilon(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Base")
    # run_acas_all_epsilon(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Input-DiffZono")

    # run_acas_all_epsilon(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Base")
    # run_acas_all_epsilon(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Input-DiffZono")
    
    # run_acas_all_epsilon(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Base")
    # run_acas_all_epsilon(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Input-DiffZono")

    println("\nRunning MNIST all...")

    run_func = run_mnist_all_epsilon("mnist-prune.csv", "experiments_final/better_bounds")
    
    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=LP), "DeepSplit-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=LP), "DeepSplit-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=VerticalSplitting), "VerticalSplitting-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=VerticalSplitting), "VerticalSplitting-DU-Input-DiffZono")
end

function run_experiments_pool5()
    println("\nRunning MNIST all...")

    run_func = run_mnist_all_epsilon("mnist-prune-global_4-5.csv", "experiments_final/better_bounds")
    
    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=LPZonoContract), "LP-ZC-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractInter), "ZonoContractInter-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContractPost), "ZonoContractPost-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=ZonoContraction, contract=ZonoContract), "ZonoContract-DU-Input-DiffZono")

    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=LP), "DeepSplit-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=LP), "DeepSplit-DU-Input-DiffZono")
    
    run_func(deepsplit_epsilon((true, false, false, false); mode=DeepSplitUnbiased, approach=VerticalSplitting), "VerticalSplitting-DU-Base")
    run_func(deepsplit_epsilon((true, false, true, true); mode=DeepSplitUnbiased, approach=VerticalSplitting), "VerticalSplitting-DU-Input-DiffZono")
end

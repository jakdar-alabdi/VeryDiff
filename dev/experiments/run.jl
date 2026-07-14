function parse_networks(nn_file₁::String, nn_file₂::String)
    println("Parsing $(basename(nn_file₁))...")
    N₁ = load_onnx_model(nn_file₁)
    println("Parsing $(basename(nn_file₂))...")
    N₂ = load_onnx_model(nn_file₂)
    return N₁, N₂
end

function save_results(out_dir::String, net_name::String, spec_name::String, veri_result::VerificationResult)
    original_stdout = stdout
    original_stderr = stderr
    (;status, initial_δ_bound, final_δ_bound, num_propagations, num_input_splits, num_neuron_splits, verification_time) = veri_result
    open(out_dir, "a") do c
        redirect_stdout(c)
        redirect_stderr(c)
        flush(stdout)
        flush(stderr)
        println("$net_name, $spec_name, $(status), $(verification_time), $(num_propagations), $num_input_splits, $num_neuron_splits, $(initial_δ_bound), $(final_δ_bound)")
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
    println("Using $(get_config())...")
    for (bounds, _, _, _) in f
        veri_result = verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, veri_result)
        end
        println("\n$veri_result")
    end
end

function verydiff_top1(nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true)
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = get_top1_property(;delta=delta)
    println("Using $(get_config())...")
    for (bounds, _, _, _) in f
        veri_result = verify_network(N₁, N₂, bounds, property_check, top1_configure_split_heuristic(1); timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, veri_result)
        end
        println("\n$veri_result")
    end
end

function deepsplit_epsilon(nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true)
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = Properties.get_epsilon_property_with_neuron_splitting(epsilon)
    println("Using $(get_config())...")
    for (bounds, _, _, _) in f
        veri_result = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, veri_result)
        end
        println("\n$veri_result")
    end
end

function deepsplit_top1(nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true)
    N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
    f, n_inputs, _ = get_ast(spec_file)
    property_check = Properties.get_top1_property_with_neuron_splitting(delta)
    println("Using $(get_config())...")
    for (bounds, _, _, _) in f
        veri_result = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout)
        net_name = replace(basename(nn_file₂), ".onnx" => "")
        spec_name = replace(basename(spec_file), ".vnnlib" => "")
        if save
            save_results(result_out_dir, net_name, spec_name, veri_result)
        end
        println("\n$veri_result")
    end
end

function run_experiments_acas_epsilon(specs_file::String; heuristic_config=(true, false, false), run_verydiff=false)
    println("\nRunning ACAS all...")
    run_func = run_acas_all_epsilon(specs_file, "experiments_final/")
    
    set_config(heuristic_config, (false, false, false, false), false, DeepSplitUnbiased, ZonoContraction, LPZonoContract)
    run_func(deepsplit_epsilon, get_config())
    set_config(heuristic_config, (true, true, true), false, DeepSplitUnbiased, ZonoContraction, LPZonoContract)
    run_func(deepsplit_epsilon, get_config())
    
    set_config(heuristic_config, (false, false, false), false, DeepSplitUnbiased, LP)
    run_func(deepsplit_epsilon, get_config())
    set_config(heuristic_config, (true, true, true), false, DeepSplitUnbiased, LP)
    run_func(deepsplit_epsilon, get_config())

    if run_verydiff
        set_config((false, false, false), (false, false, false), false)
        run_func(verydiff_epsilon, get_config())
        set_config((false, false, false), (true, true, true), false)
        run_func(verydiff_epsilon, get_config())
    end
end

function run_experiments_mnist_epsilon(specs_file::String; heuristic_config=(true, false, false), run_verydiff=false)
    println("\nRunning MNIST all...")
    run_func = run_mnist_all_epsilon(specs_file, "experiments_final/")
    
    set_config(heuristic_config, (false, true, false, false), true, DeepSplitUnbiased, LP)
    run_func(deepsplit_epsilon, get_config())

    if run_verydiff
        set_config((false, false, false), (false, false, false, false), false)
        run_func(verydiff_epsilon, get_config())
    end
end

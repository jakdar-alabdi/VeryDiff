# using Pkg
# Pkg.activate("./")
using VeryDiff
using VNNLib

include("fuzzing.jl")

function parse_networks(nn_file₁::String, nn_file₂::String)
    println("Parsing $(basename(nn_file₁))...")
    N₁ = load_onnx_model(nn_file₁)
    println("Parsing $(basename(nn_file₂))...")
    N₂ = load_onnx_model(nn_file₂)
    return N₁, N₂
end

function verydiff_epsilon(split_bounds_config::Tuple{Bool, Bool, Bool})
    return (nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        property_check = VeryDiff.get_epsilon_property(epsilon)
        VeryDiff.set_neuron_splitting_config((false, false, false), split_bounds_config)
        println("\nUsing $(VeryDiff.get_config()) as verifier\n")
        for (bounds, _, _, _) in f
            veri_result = verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=timeout, fuzz_testing=nothing)
            println(veri_result)
        end
    end
end

function verydiff_top1(split_bounds_config::Tuple{Bool, Bool, Bool})
    return (nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        property_check = VeryDiff.get_top1_property(;delta=delta)
        VeryDiff.set_neuron_splitting_config((false, false, false), split_bounds_config)
        VeryDiff.NEW_HEURISTIC[] = true
        println("\nUsing $(VeryDiff.get_config()) as verifier\n")
        for (bounds, _, _, _) in f
            veri_result = verify_network(N₁, N₂, bounds, property_check, top1_configure_split_heuristic(1); timeout=timeout)
            println(veri_result)
        end
    end
end

function deepsplit_epsilon(
    heuristic_config::Tuple{Bool, Bool, Bool},
    split_bounds_config::Tuple{Bool, Bool, Bool}; 
    mode=VeryDiff.ZonoBiased, 
    approach=VeryDiff.LP, 
    contract=VeryDiff.ZonoContract)
    return (nn_file₁::String, nn_file₂::String, spec_file::String, epsilon::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        VeryDiff.set_neuron_splitting_config(heuristic_config, split_bounds_config; mode=mode, approach=approach, contract=contract)
        println("\nUsing $(VeryDiff.get_config())...\n")
        property_check = VeryDiff.get_epsilon_property_with_neuron_splitting(epsilon)
        for (bounds, _, _, _) in f
            veri_result = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout, fuzz_testing=nothing)
            println(veri_result)
        end
    end
end

function deepsplit_top1(
    heuristic_config::Tuple{Bool, Bool, Bool},
    split_bounds_config::Tuple{Bool, Bool, Bool}; 
    mode=VeryDiff.ZonoBiased, 
    approach=VeryDiff.LP, 
    contract=VeryDiff.ZonoContract)
    return (nn_file₁::String, nn_file₂::String, spec_file::String, delta::Float64, timeout::Int64, result_out_dir::String; save=true) -> begin
        N₁, N₂ = parse_networks(nn_file₁, nn_file₂)
        f, n_inputs, _ = get_ast(spec_file)
        VeryDiff.set_neuron_splitting_config(heuristic_config, split_bounds_config; mode=mode, approach=approach, contract=contract)
        println("\nUsing $(VeryDiff.get_config())...\n")
        property_check = VeryDiff.get_top1_property_with_neuron_splitting(delta)
        for (bounds, _, _, _) in f
            veri_result = deepsplit_verify_network(N₁, N₂, bounds, property_check; timeout=timeout, fuzz_testing=nothing)
            println(veri_result)
        end
    end
end


function run_tests_epsilon(benchmarks_dir::String, specs_csv_file::String, run_name::String, eval_func)
    open(specs_csv_file, "r") do f
        while !eof(f)
            spec = split(readline(f), ",")
            if !isempty(spec)
                nn_file₁ = "$benchmarks_dir/$(spec[1])"
                nn_file₂ = "$benchmarks_dir/$(spec[2])"
                spec_file = "$benchmarks_dir/$(spec[3])"
                epsilon = parse(Float64, string(spec[4]))
                timeout = parse(Int64, string(spec[5]))
                eval_func(nn_file₁, nn_file₂, spec_file, epsilon, timeout, ""; save=false)
            end
        end
    end
end

function run_tests_top1(benchmarks_dir::String, specs_csv_file::String, run_name::String, eval_func)
    open(specs_csv_file, "r") do f
        while !eof(f)
            spec = split(readline(f), ",")
            if !isempty(spec)
                nn_file₁ = "$benchmarks_dir/$(spec[1])"
                nn_file₂ = "$benchmarks_dir/$(spec[2])"
                spec_file = "$benchmarks_dir/$(spec[3])"
                timeout = parse(Int64, string(spec[end]))
                for delta in spec[4:end-1]
                    delta = parse(Float64, string(delta))
                    eval_func(nn_file₁, nn_file₂, spec_file, delta, timeout, ""; save=false)
                end
            end
        end
    end
end

cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"
acas_csv_dir = joinpath(cur_dir, "acas-prune.csv")
mnist_csv_dir = joinpath(cur_dir, "mnist-prune.csv")
lhc_csv_dir = joinpath(cur_dir, "lhc.csv")

# verifier = deepsplit_top1(
#     (true, true, true),
#     (false, true, true);
#     mode=VeryDiff.DeepSplitUnbiased,
#     approach=VeryDiff.ZonoContraction,
#     contract=VeryDiff.ZonoContractInter
#     )
# verifier = verydiff_top1((false, true, true))
verifier = deepsplit_epsilon(
    (true, true, true),
    (false, true, true);
    mode=VeryDiff.DeepSplitUnbiased,
    approach=VeryDiff.ZonoContraction,
    contract=VeryDiff.LPZonoContract
    )
# verifier = verydiff_epsilon((false, true, true))

run_tests_epsilon(benchmarks_dir, mnist_csv_dir, "", verifier)
# run_tests_top1(benchmarks_dir, lhc_csv_dir, "", verifier)
# run_tests_epsilon(benchmarks_dir, mnist_csv_dir, "", verifier)

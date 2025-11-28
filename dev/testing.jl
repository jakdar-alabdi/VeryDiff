using VeryDiff
using LinearAlgebra
using JuMP, Gurobi
using Random
using VNNLib
import VNNLib.NNLoader: load_network


sysimage_dir = @__DIR__

function VeryDiff.deepsplit_lp_search_epsilon(nn_file, spec_file, epsilon)
    nn_file₁ = nn_file
    println("Parsing $(basename(nn_file₁))...")
    N₁ = parse_network(VNNLib.load_network(nn_file₁))
    nn_file₂ = replace(nn_file, "nets" => "nets_pruned", ".onnx" => "-0.0.onnx")
    
    f, n_inputs, _ = get_ast(spec_file)

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]
        nn_file₂ = replace(nn_file₂, "$(i - 0.1).onnx" => "$i.onnx")
        println("Parsing $(basename(nn_file₂))...")
        N₂ = parse_network(VNNLib.load_network(nn_file₂))
        for (bounds, _, _, _) in f
            println("\nExecuting DeepSplit LP-based Search")
            runtime_deepsplit = @elapsed deepsplit_lp_search_epsilon(N₁, N₂, bounds, ϵ)
            println("\n######################################################################################")
            println("\nExecuting VeryDiff")
            runtime_verydif = @elapsed run_cmd(["--epsilon", "$ϵ", nn_file₁, nn_file₂, spec_file])
                
            println("\n")
            println("DeepSplit Time: $runtime_deepsplit")
            println("VeryDiff Time: $runtime_verydif")        
        end
    end
end

lhc_path = "$sysimage_dir\\..\\..\\verydiff-experiments\\new_benchmarks\\lhc"
log_dir = "$sysimage_dir\\testing_log"
ϵ = 1.0

original_stdout = stdout
original_stderr = stderr

nn_file = "$lhc_path\\nets\\4_20-1.onnx"
log_dir = joinpath(log_dir, "lhc-$ϵ", replace(basename(nn_file), ".onnx" => ""))
mkpath(log_dir)

spec_file = "$lhc_path\\specs\\sigma_0.1.vnnlib"
spec_name = basename(spec_file)
log_file_name = joinpath(log_dir, "$spec_name.log")

open(log_file_name, "w") do f
    redirect_stdout(f) do
        redirect_stderr(f) do
            deepsplit_lp_search_epsilon(nn_file, spec_file, ϵ)
        end
    end
end

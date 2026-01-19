using VeryDiff
using VNNLib
import VNNLib.NNLoader: load_network

sysimage_dir = @__DIR__

function run(nn_file₁, nn_file₂, spec_file, epsilon)
    println("Parsing $(basename(nn_file₁))...")
    N₁ = parse_network(load_network(nn_file₁))
    println("Parsing $(basename(nn_file₂))...")
    N₂ = parse_network(load_network(nn_file₂))

    f, n_inputs, _ = get_ast(spec_file)

    property_check = get_epsilon_property(ϵ)

    for (bounds, _, _, _) in f
        println("\nExecuting DeepSplit LP-based Search")
        runtime_deepsplit = @elapsed deepsplit_lp_search_epsilon(N₁, N₂, bounds, ϵ; timeout=120)
        
        println("\nExecuting VeryDiff")
        VeryDiff.set_neuron_splitting_config((false, false, false, false))
        runtime_verydif = @elapsed verify_network(N₁, N₂, bounds, property_check, epsilon_split_heuristic; timeout=120)

        println("\nDeepSplit Time: $runtime_deepsplit")
        println("VeryDiff Time: $runtime_verydif")
    end
end

benchmarks_dir = "$sysimage_dir\\..\\..\\verydiff-experiments"
mnist_path = "$benchmarks_dir\\benchmarks\\mnist-prune"
acas_path = "$benchmarks_dir\\benchmarks\\acas-prune"
lhc_path = "$benchmarks_dir\\new_benchmarks\\lhc"
examples_path = "$sysimage_dir\\..\\test\\examples"

# benchmark_name = "mnist"
benchmark_name = "acas"

VeryDiff.set_neuron_splitting_config((true, false, false, false); mode=VeryDiff.ZonoUnbiased, approach=VeryDiff.ZonoContraction)
if VeryDiff.DEEPSPLIT_HUERISTIC_ALTERNATIVE[]
    log_dir = "$sysimage_dir\\testing\\DeepSplit-Alt"
else
    log_dir = "$sysimage_dir\\testing\\DeepSplit-Base"
end
# ϵ = 55.0
ϵ = 0.5
# ϵ = 1.0

# nn_file₁ = "$mnist_path\\nets\\mnist_relu_3_100.onnx"
nn_file₁ = "$acas_path\\nets\\ACASXU_run2a_1_1_batch_2000.onnx"
# nn_file₁ = "$examples_path\\nets\\ACASXU_run2a_1_1_batch_2000.onnx"
# nn_file₁ = "$lhc_path\\nets\\2_40-0.1.onnx"
nn_file₁_name = basename(replace(nn_file₁, ".onnx" => ""))
# nn_file₂ = "$mnist_path\\nets_pruned\\mnist_relu_3_100_pruned5.onnx"
nn_file₂ = "$acas_path\\nets_pruned\\ACASXU_run2a_1_1_batch_2000_pruned5.onnx"
# nn_file₂ = replace(nn_file₁, "nets" => "nets_pruned", ".onnx" => "-0.1.onnx")
# nn_file₂ = "$examples_path\\nets\\ACASXU_run2a_1_1_batch_2000_pruned5.onnx"
nn_file₂_name = basename(replace(nn_file₂, ".onnx" => ""))

# spec_file = "$mnist_path\\specs\\mnist_0_global_3.vnnlib"
spec_file = "$acas_path\\specs\\prop_1.vnnlib"
# spec_file = "$lhc_path\\specs\\sigma_1.0.vnnlib"
# spec_file = "$examples_path\\specs\\prop_1.vnnlib"
spec_name = replace(basename(spec_file), ".vnnlib" => "")

log_dir = joinpath(log_dir, "$benchmark_name-$ϵ", nn_file₂_name)
# log_dir = joinpath(log_dir, "acas", nn_file₁_name, nn_file₂_name, spec_name)
# # log_dir = joinpath(log_dir, "lhc", nn_file₁_name, nn_file₂_name, spec_name)
# log_dir = joinpath(log_dir, "acas", nn_file₁_name, nn_file₂_name, spec_name)
mkpath(log_dir)

log_file_name = joinpath(log_dir, "$spec_name.log")

# open(log_file_name, "w") do f
#     redirect_stdout(f) do
#         redirect_stderr(f) do
            run(nn_file₁, nn_file₂, spec_file, ϵ)
#         end
#     end
# end

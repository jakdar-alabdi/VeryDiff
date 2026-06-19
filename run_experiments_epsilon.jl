using Pkg
Pkg.activate("./")
using VeryDiff

benchmark = ARGS[1]
config = ARGS[2]
specs_file = ARGS[3]
run_verydiff = length(ARGS) > 3 && ARGS[4] == "run VeryDiff"

if benchmark == "MNIST"
    run_func = VeryDiff.run_experiments_mnist_epsilon
elseif benchmark == "ACAS"
    run_func = VeryDiff.run_experiments_acas_epsilon
else
    throw("Benchmark $(benchmark) not recognized.")
end

if config == "NEURON"
    heuristic_config = (true, false, false)
elseif config == "INPUT"
    heuristic_config = (true, false, true)
elseif config == "DIFF"
    heuristic_config = (true, true, false)
elseif config == "INPUT-DIFF"
    heuristic_config = (true, true, true)
else
    throw("Heuristic configuration $(config) not recognized.")
end

run_func(specs_file; heuristic_config=heuristic_config, run_verydiff=run_verydiff)

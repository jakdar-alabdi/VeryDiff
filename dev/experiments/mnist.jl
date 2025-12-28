cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"
# benchmarks_dir = "$cur_dir\\..\\..\\..\\verydiff-experiments"

mnist_specs = SpecificationEpsilon[]

# bench_name = "mnist"
csv_dir = "$cur_dir/mnist-prune-verydiff.csv"
# csv_dir = "$cur_dir/mnist-prune.csv"
# csv_dir = "$cur_dir\\mnist-prune.csv"

open(csv_dir, "r") do f
    while !eof(f)
        spec = split(readline(f), ",")
        nn_file₁ = "$benchmarks_dir/$(spec[1])"
        nn_file₂ = "$benchmarks_dir/$(spec[2])"
        spec_file = "$benchmarks_dir/$(spec[3])"
        # nn_file₁ = "$benchmarks_dir\\$(replace(spec[1], "/" => "\\"))"
        # nn_file₂ = "$benchmarks_dir\\$(replace(spec[2], "/" => "\\"))"
        # spec_file = "$benchmarks_dir\\$(replace(spec[3], "/" => "\\"))"
        epsilon = parse(Float64, string(spec[4]))
        timeout = parse(Int64, string(spec[5]))
        push!(mnist_specs, SpecificationEpsilon(nn_file₁, nn_file₂, spec_file, epsilon, timeout))
    end
end

function _run_mnist_all(specs::Vector{SpecificationEpsilon}, log_dir::String, run_name::String, eval_func)
    for spec in specs
        (; nn_file₁, nn_file₂, spec_file, epsilon, timeout) = spec
        net_name = replace(basename(nn_file₂), ".onnx" => "", "mnist_relu_" => "")
        out_dir = joinpath(log_dir, run_name, "mnist-$epsilon", net_name)
        mkpath(out_dir)
        spec_file_name = replace(basename(spec_file), ".vnnlib" => "")
        log_file_name = joinpath(out_dir, "$spec_file_name.log")
        csv_file_name = joinpath(log_dir, run_name, "mnist-$epsilon", "results.csv")
        
        original_stdout = stdout
        original_stderr = stderr
        open(log_file_name, "w") do f
            redirect_stdout(f)
            redirect_stderr(f)
            flush(stdout)
            flush(stderr)
            eval_func(spec, csv_file_name)
            flush(stdout)
            flush(stderr)
            GC.gc()
            redirect_stdout(original_stdout)
            redirect_stderr(original_stderr)
        end
    end
end

function run_mnist_all(eval_func, run_name::String)
    # _run_mnist_all(mnist_specs, "$cur_dir\\experiments_final", run_name, eval_func)
    _run_mnist_all(mnist_specs, "$cur_dir/experiments_final", run_name, eval_func)
end

cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"
# benchmarks_dir = "$cur_dir\\..\\..\\..\\verydiff-experiments"

acas_specs = SpecificationEpsilon[]

# bench_name = "acas"
csv_dir = "$cur_dir/acas-prune.csv"
# csv_dir = "$cur_dir\\acas-prune.csv"

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
        push!(acas_specs, SpecificationEpsilon(nn_file₁, nn_file₂, spec_file, epsilon, timeout))
    end
end

function _run_acas_all(specs::Vector{SpecificationEpsilon}, log_dir::String, run_name::String, eval_func)
    for spec in specs
        (; nn_file₁, nn_file₂, spec_file, epsilon, timeout) = spec
        net_name = replace(basename(nn_file₂), ".onnx" => "", "ACASXU_run2a_" => "", "batch_2000_" => "")
        out_dir = joinpath(log_dir, run_name, "acas-$epsilon", net_name)
        mkpath(out_dir)
        spec_file_name = replace(basename(spec_file), ".vnnlib" => "")
        log_file_name = joinpath(out_dir, "$spec_file_name.log")
        csv_file_name = joinpath(log_dir, run_name, "acas-$epsilon", "results.csv")
        open(log_file_name, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    eval_func(spec, csv_file_name)
                end
            end
        end
    end
end

function run_acas_all(eval_func, run_name::String)
    # _run_acas_all(acas_specs, "$cur_dir\\experiments_final", run_name, eval_func)
    _run_acas_all(acas_specs, "$cur_dir/experiments_final", run_name, eval_func)
end

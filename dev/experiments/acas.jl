cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"

function _run_acas_all(specs_csv_file::String, warmup_specs_csv_file::String, log_dir::String, run_name::String, eval_func)
    
    println("\nWarmup...")
    println("\nConfiguration: $run_name\n")

    open(warmup_specs_csv_file, "r") do f
        while !eof(f)
            spec = split(readline(f), ",")
            nn_file₁ = "$benchmarks_dir/$(spec[1])"
            nn_file₂ = "$benchmarks_dir/$(spec[2])"
            spec_file = "$benchmarks_dir/$(spec[3])"
            epsilon = parse(Float64, string(spec[4]))
            timeout = parse(Int64, string(spec[5]))
            eval_func(nn_file₁, nn_file₂, spec_file, epsilon, timeout, ""; save=false)
        end
    end

    println("\nConfiguration: $run_name\n")

    open(specs_csv_file, "r") do f
        while !eof(f)
            spec = split(readline(f), ",")
            nn_file₁ = "$benchmarks_dir/$(spec[1])"
            nn_file₂ = "$benchmarks_dir/$(spec[2])"
            spec_file = "$benchmarks_dir/$(spec[3])"
            epsilon = parse(Float64, string(spec[4]))
            timeout = parse(Int64, string(spec[5]))

            net_name = replace(basename(nn_file₂), ".onnx" => "", "ACASXU_run2a_" => "", "batch_2000_" => "")
            out_dir = joinpath(log_dir, run_name, "acas-$epsilon", net_name)
            mkpath(out_dir)
            spec_file_name = replace(basename(spec_file), ".vnnlib" => "")
            log_file_name = joinpath(out_dir, "$spec_file_name.log")
            csv_file_name = joinpath(log_dir, run_name, "acas-$epsilon", "results.csv")
            
            # print statement bench, prop
            original_stdout = stdout
            original_stderr = stderr
            open(log_file_name, "w") do f
                redirect_stdout(f)
                redirect_stderr(f)
                flush(stdout)
                flush(stderr)
                eval_func(nn_file₁, nn_file₂, spec_file, epsilon, timeout, csv_file_name; save=true)
                flush(stdout)
                flush(stderr)
                GC.gc()
                redirect_stdout(original_stdout)
                redirect_stderr(original_stderr)
            end
        end
    end
end

function run_acas_all(eval_func, run_name::String)
    _run_acas_all("$cur_dir/acas-prune.csv", "$cur_dir/acas-prune_warmup.csv", "$cur_dir/experiments_final", run_name, eval_func)
end

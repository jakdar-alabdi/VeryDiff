cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"

function _run_mnist_all_epsilon(specs_file::String, warmup_specs_file::String, out_dir::String, run_name::String, eval_func)
    
    println("Configuration: $run_name")
    println("\nWarmup...")
    
    open(warmup_specs_file, "r") do f
        while !eof(f)
            spec = split(readline(f), ",")
            nn_file₁ = "$benchmarks_dir/$(spec[1])"
            nn_file₂ = "$benchmarks_dir/$(spec[2])"
            spec_file = "$benchmarks_dir/$(spec[3])"
            epsilon = parse(Float64, string(spec[4]))
            timeout = parse(Int64, string(spec[5]))

            println("\nNN₁: $(basename(nn_file₁))")
            println("NN₂: $(basename(nn_file₂))")
            println("Prop: $(basename(spec_file))")
            
            original_stdout = stdout
            original_stderr = stderr
            redirect_stdout(devnull)
            redirect_stderr(devnull)
            flush(stdout)
            flush(stderr)
            eval_func(nn_file₁, nn_file₂, spec_file, epsilon, timeout, ""; save=false)
            flush(stdout)
            flush(stderr)
            GC.gc()
            redirect_stdout(original_stdout)
            redirect_stderr(original_stderr)
        end
    end

    println("\nWarmup End")
    
    config_dir = joinpath(out_dir, run_name)
    if !isdir(config_dir)
        mkpath(config_dir)
    end
    first_line = true
    specs_name = replace(basename(specs_file), "-prune" => "", ".csv" => "")
    open(specs_file, "r") do f
        while !eof(f)
	        spec = split(readline(f), ",")
            nn_file₁ = "$benchmarks_dir/$(spec[1])"
            nn_file₂ = "$benchmarks_dir/$(spec[2])"
            spec_file = "$benchmarks_dir/$(spec[3])"
            epsilon = parse(Float64, string(spec[4]))
            timeout = parse(Int64, string(spec[5]))

            mnist_name = "$specs_name-$epsilon-$timeout"
            spec_file_name = replace(basename(spec_file), ".vnnlib" => "")
            log_file_dir = joinpath(config_dir, mnist_name)
            if first_line
                rm(log_file_dir, force=true)
                first_line = false
            end
            
            net_name = replace(basename(nn_file₂), ".onnx" => "", "mnist_relu_" => "")
            log_file = joinpath(log_file_dir, net_name, "$spec_file_name.log")
            results_file = joinpath(config_dir, mnist_name, "results.csv")
            
            println("\nNN₁: $(basename(nn_file₁))")
            println("NN₂: $(basename(nn_file₂))")
            println("Prop: $(basename(spec_file))")

            original_stdout = stdout
            original_stderr = stderr
            open(log_file, "w") do f
                redirect_stdout(f)
                redirect_stderr(f)
                flush(stdout)
                flush(stderr)
                eval_func(nn_file₁, nn_file₂, spec_file, epsilon, timeout, results_file; save=true)
                flush(stdout)
                flush(stderr)
                GC.gc()
                redirect_stdout(original_stdout)
                redirect_stderr(original_stderr)
            end
        end
    end
end

function run_mnist_all_epsilon(specs_file::String, out_dir::String)
    return (eval_func, run_name::String) -> begin
        _run_mnist_all_epsilon("$cur_dir/specs/$specs_file", "$cur_dir/specs/mnist-prune_warmup.csv", "$cur_dir/$out_dir", run_name, eval_func)
    end
end

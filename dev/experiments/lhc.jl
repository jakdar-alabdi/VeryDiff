cur_dir = @__DIR__
benchmarks_dir = "$cur_dir/../../../verydiff-experiments"

function _run_lhc_all_top1(specs_file::String, warmup_specs_file::String, out_dir::String, run_name::String, eval_func)
    
    println("Configuration: $run_name")
    println("\nWarmup...")
    
    open(warmup_specs_file, "r") do f
        while !eof(f)
	        query = split(readline(f), ",")
            nn_file₁ = joinpath(benchmarks_dir, query[1])
            nn_file₂ = joinpath(benchmarks_dir, query[2])
            spec_file = joinpath(benchmarks_dir, query[3])
            delta = parse(Float64, string(query[4]))
            timeout = parse(Int64, string(query[5]))

            println("\nNN₁: $(basename(nn_file₁))")
            println("NN₂: $(basename(nn_file₂))")
            println("Prop: $(basename(spec_file))")
            
            original_stdout = stdout
            original_stderr = stderr
            redirect_stdout(devnull)
            redirect_stderr(devnull)
            flush(stdout)
            flush(stderr)
            eval_func(nn_file₁, nn_file₂, spec_file, delta, timeout, ""; save=false)
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
    specs_file_name = replace(basename(specs_file), "-prune" => "", ".csv" => "")
    
    open(specs_file, "r") do f
        while !eof(f)
	        query = split(readline(f), ",")
            nn_file₁ = joinpath(benchmarks_dir, query[1])
            nn_file₂ = joinpath(benchmarks_dir, query[2])
            spec_file = joinpath(benchmarks_dir, query[3])
            timeout = parse(Int64, string(query[end]))

            for delta in query[4:end-1]
                delta = parse(Float64, string(delta))
                
                lhc_name = "$specs_file_name-top1-$delta-$timeout"
                net_name = replace(basename(nn_file₂), ".onnx" => "")
                spec_file_name = replace(basename(spec_file), ".vnnlib" => "")

                lhc_out_dir = joinpath(config_dir, lhc_name)
                if first_line
                    rm(lhc_out_dir, force=true, recursive=true)
                    mkpath(lhc_out_dir)
                end
                
                net_out_dir = joinpath(lhc_out_dir, net_name)
                if !isdir(net_out_dir)
                    mkdir(net_out_dir)
                end

                log_file = joinpath(lhc_out_dir, net_name, "$spec_file_name.log")
                touch(log_file)
                results_file = joinpath(lhc_out_dir, "results.csv")
                touch(results_file)
                
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
                    eval_func(nn_file₁, nn_file₂, spec_file, delta, timeout, results_file; save=true)
                    flush(stdout)
                    flush(stderr)
                    GC.gc()
                    redirect_stdout(original_stdout)
                    redirect_stderr(original_stderr)
                end
            end

            first_line = false
        end
    end
end

function run_lhc_all_top1(eval_func, run_name::String)
    _run_lhc_all_top1("$cur_dir/specs/lhc.csv", "$cur_dir/specs/lhc_warmup.csv", "$cur_dir/experiments_final", run_name, eval_func)
end

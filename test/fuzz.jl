if "VERYDIFF_FUZZ" in keys(ENV)
    iterations = parse(Int, ENV["VERYDIFF_FUZZ"])
    @testset "Fuzzer" begin
        i = 1
        @info "Starting fuzzing with $(iterations > 0 ? iterations : "infinite") iterations"
        while true
            ENV["VERYDIFF_TEST_SEED"] = string(rand(1:999999))
            cur_ts = @testset "Fuzzing iteration $i" begin
                @info "Fuzzing iteration $i"
                include("unit/propagation/dense.jl")
                include("unit/propagation/dense_zero_diff.jl")
                include("unit/propagation/relu.jl")
            end
            Test.print_test_results(cur_ts)
            if iterations > 0 && i >= iterations
                break
            end
            i += 1
            # Flush stdout and stderr to ensure logs are written in case of crash
            flush(stdout)
            flush(stderr)
        end
    end
end

using Pkg
Pkg.activate("./")
using VeryDiff

cur_dir = @__DIR__
log_dir = joinpath(cur_dir, "runlim2.log")

#original_stdout = stdout
#original_stderr = stderr
#open(log_dir, "w") do f
#    redirect_stdout(f)
#    redirect_stderr(f)
#    flush(stdout)
#    flush(stderr)
    run_experiments()
#    flush(stdout)
#    flush(stderr)
#    GC.gc()
#    redirect_stdout(original_stdout)
#    redirect_stderr(original_stderr)
#end

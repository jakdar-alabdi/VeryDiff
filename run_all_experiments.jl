using Pkg
Pkg.activate("./")
using VeryDiff

original_stdout = stdout
original_stderr = stderr
open("runlim.log", "w") do f
    redirect_stdout(f)
    redirect_stderr(f)
    flush(stdout)
    flush(stderr)
    run_experiments()
    flush(stdout)
    flush(stderr)
    GC.gc()
    redirect_stdout(original_stdout)
    redirect_stderr(original_stderr)
end

using Pkg
Pkg.activate("./")
using VeryDiff

include("dev/testing/fuzzing.jl")

start_fuzz_testing()

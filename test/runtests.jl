using Test

using VeryDiff

VeryDiff.USE_DIFFZONO = true
VeryDiff.NEW_HEURISTIC = true
VeryDiff.Transformers.RELU_PROP_MODE = :compare

include("unit/util/simd_bool.jl")

@testset "VeryDiff Tests" begin
    @testset "Propagation Tests" begin
        include("unit/propagation/main.jl")
    end
end

include("fuzz.jl")

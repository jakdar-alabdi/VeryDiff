using Test

using VeryDiff

VeryDiff.USE_DIFFZONO = true
VeryDiff.NEW_HEURISTIC = true

include("unit/util/simd_bool.jl")

@testset "VeryDiff Tests" begin
    @testset "Propagation Tests" begin
        include("unit/propagation/main.jl")
    end
    @testset "Properties" begin
        include("unit/properties/main.jl")
    end
end

include("fuzz.jl")

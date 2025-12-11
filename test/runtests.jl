using Test

using VeryDiff

VeryDiff.USE_DIFFZONO = true
VeryDiff.NEW_HEURISTIC = true

@testset "VeryDiff Tests" begin
    @testset "Propagation Tests" begin
        include("unit/propagation/main.jl")
    end
end

include("fuzz.jl")

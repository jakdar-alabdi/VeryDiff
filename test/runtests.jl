using Test

@testset "VeryDiff Tests" begin
    @testset "Propagation Tests" begin
        include("unit/propagation/dense.jl")
    end
end

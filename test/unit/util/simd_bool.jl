using Random

@testset "simd_bool_expr macro tests" begin
    # Test correctness of simd_bool_expr macro
    for i in 1:10_000
        # Seed randomness
        Random.seed!(i)
        @testset "Test iteration $i" begin
            DIM = rand(1000:5000)
            lower1 = randn(DIM)
            upper1 = lower1 .+ abs.(randn(DIM))
            lower2 = randn(DIM)
            upper2 = lower2 .+ abs.(randn(DIM))
            check = falses(DIM)
            check .= rand(Bool, DIM)

            expected = (lower1 .< 0.0) .&& (upper1 .> 0.0) .&& (lower2 .< 0.0) .&& (upper2 .> 0.0) .&& .!check
            macro_result = @VeryDiff.simd_bool_expr DIM ((lower1 < 0.0) & (upper1 > 0.0) & (lower2 < 0.0) & (upper2 > 0.0) & !check)
            @test expected == macro_result
        end
    end
end
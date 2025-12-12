include("../src/Util/simd_bool.jl")

function compute!(lower1::AbstractVector{<:Real},
                  upper1::AbstractVector{<:Real},
                  lower2::AbstractVector{<:Real},
                  upper2::AbstractVector{<:Real},
                  check::BitVector)
    any_any = similar(check)
    @inbounds @simd for i in eachindex(any_any)
        any_any[i] = (lower1[i] < 0.0) &
                     (upper1[i] > 0.0) &
                     (lower2[i] < 0.0) &
                     (upper2[i] > 0.0) &
                     .!check[i]
    end
    return any_any
end

using BenchmarkTools

DIM = 1000

lower1 = randn(DIM)
upper1 = lower1 .+ abs.(randn(DIM))
lower2 = randn(DIM)
upper2 = lower2 .+ abs.(randn(DIM))
check = falses(DIM)
check .= rand(Bool, DIM)

@info "Original expression:"
@btime any_any = ($lower1 .< 0.0) .&& ($upper1 .> 0.0) .&& ($lower2 .< 0.0) .&& ($upper2 .> 0.0) .&& .!$check

@info "Single &:"
@btime any_any = @.(($lower1 < 0.0) & ($upper1 > 0.0) & ($lower2 < 0.0) & ($upper2 > 0.0) & !$check)

@info "Compute function:"
@btime compute!($lower1, $upper1, $lower2, $upper2, $check)

@info "Macro:"
@btime any_any = @simd_bool_expr $DIM (($lower1 < 0.0) & ($upper1 > 0.0) & ($lower2 < 0.0) & ($upper2 > 0.0) & !$check)

# [ Info: Original expression:
#   39.464 μs (6300 allocations: 169.88 KiB)
# [ Info: Single &:
#   2.401 μs (3 allocations: 4.41 KiB)
# [ Info: Compute function:
#   1.927 μs (2 allocations: 224 bytes)
# [ Info: Macro:
#   1.940 μs (2 allocations: 224 bytes)

any_any = falses(DIM)
any_any .= rand(Bool, DIM)

@info "Retrieve true indices:"
@btime findall($any_any)
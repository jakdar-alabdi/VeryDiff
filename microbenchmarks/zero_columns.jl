# Question: Is it cheaper to use view of matrix with only first 50% of columns and perform
# matrix multiplications on that view or to set the remaining columns to zero and perform
# matrix multiplications on the full matrix?

using BenchmarkTools
using LinearAlgebra

A = rand(1000, 1000)
B = rand(1000, 1000)
C = rand(1000, 1000)

B1 = @view B[:, 1:500]
C1 = @view C[:, 1:500]

@btime mul!($C1, $A, $B1)

B2 = copy(B)
B2[:, 501:1000] .= 0.0
@btime mul!($C, $A, $B2)

#  7.477 ms (0 allocations: 0 bytes)
#  10.202 ms (0 allocations: 0 bytes)

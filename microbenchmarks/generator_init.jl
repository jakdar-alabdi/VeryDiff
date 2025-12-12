using BenchmarkTools
using LinearAlgebra
using LoopVectorization

# Dummy data similar to your scenario
N = 50_000
task_distance = rand(N)
task_distance_indices = rand(1:1000, N)  # random row indices

generator1_common = zeros(1000, N)
generator2_common = zeros(1000, N)

# --- Method 1: Original loops ---
function method_loops!(g1, g2, task_distance, task_distance_indices)
    @inbounds for (i, idx) in enumerate(task_distance_indices)
        g1[idx, i] = task_distance[i]
    end
    @inbounds for (i, idx) in enumerate(task_distance_indices)
        g2[idx, i] = task_distance[i]
    end
    return nothing
end

# --- Method 2: Vectorized / broadcast indexing ---
function method_broadcast!(g1, g2, task_distance, task_distance_indices)
    rows = task_distance_indices
    cols = eachindex(task_distance)
    g1[rows, cols] .= task_distance
    g2[rows, cols] .= task_distance
    return nothing
end

# --- Method 3: Loop over generators ---
function method_loop_generators!(g1, g2, task_distance, task_distance_indices)
    rows = task_distance_indices
    cols = eachindex(task_distance)
    for g in (g1, g2)
        g[rows, cols] .= task_distance
    end
    return nothing
end

# --- Method 4: SIMD version ---
function method_simd!(g1, g2, task_distance, task_distance_indices)
    @inbounds @simd for i in eachindex(task_distance)
        idx = task_distance_indices[i]
        val = task_distance[i]
        g1[idx, i] = val
        g2[idx, i] = val
    end
    return nothing
end

# -- Method 5: LoopVectorization turbo --
function method_turbo!(g1, g2, task_distance, task_distance_indices)
    LoopVectorization.@avx for i in eachindex(task_distance)
        idx = task_distance_indices[i]
        val = task_distance[i]
        g1[idx, i] = val
        g2[idx, i] = val
    end
    return nothing
end

# --- Assignment as multiplication with selected columns from identity matrix ---
function method_assignment_identity!(g1, g2, task_distance, task_distance_indices)
    g1 .= Matrix{Float64}(I, length(task_distance), length(task_distance))[:, task_distance_indices] * task_distance'
    g2 .= Matrix{Float64}(I, length(task_distance), length(task_distance))[:, task_distance_indices] * task_distance'
    return nothing
end
println("Running benchmarks…")

println("\n--- Original loops ---")
@btime method_loops!($generator1_common, $generator2_common, $task_distance, $task_distance_indices)

println("\n--- SIMD version ---")
@btime method_simd!($generator1_common, $generator2_common,
                    $task_distance, $task_distance_indices)

println("\n--- Turbo version ---")
@btime method_turbo!($generator1_common, $generator2_common,
                    $task_distance, $task_distance_indices)

# println("\n--- Assignment via identity matrix ---")
# This one fills up a lot of memory
# @btime method_assignment_identity!($generator1_common, $generator2_common,
#                                   $task_distance, $task_distance_indices)

println("\n--- Broadcast vectorized indexing ---")
@btime method_broadcast!($generator1_common, $generator2_common, $task_distance, $task_distance_indices)

println("\n--- Loop over generators (broadcast) ---")
@btime method_loop_generators!($generator1_common, $generator2_common, $task_distance, $task_distance_indices)


# --- Original loops ---
#   178.543 μs (0 allocations: 0 bytes)

# --- SIMD version ---
#   190.396 μs (0 allocations: 0 bytes)

# --- Broadcast vectorized indexing ---
#   2.334 s (0 allocations: 0 bytes)

# --- Loop over generators (broadcast) ---
#   2.764 s (0 allocations: 0 bytes)
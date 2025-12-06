using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, Zonotope, zono_bounds, reset_ps!
using Test
using LinearAlgebra
using Random

"""
    create_random_dense_network(input_dim::Int, num_layers::Int, layer_dims::Vector{Int})
    
Create a randomized neural network with only Dense layers with fixed dimensions.
"""
function create_random_dense_network(input_dim::Int, num_layers::Int, layer_dims::Vector{Int})
    layers = Layer[]
    cur_dim = input_dim
    
    for i in 1:num_layers
        new_dim = layer_dims[i]
        
        W = randn(Float64, (new_dim, cur_dim))
        b = randn(Float64, new_dim)
        
        push!(layers, Dense(W, b))
        cur_dim = new_dim
    end
    
    return Network(layers)
end

"""
    sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int)
    
Sample uniformly from the hypercube defined by low and high bounds.
"""
function sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int)
    dim = length(low)
    samples = zeros(Float64, dim, num_samples)
    
    for i in 1:num_samples
        samples[:, i] = low .+ (high .- low) .* rand(Float64, dim)
    end
    
    return samples
end

"""
    create_zonotope_from_bounds(low::Vector, high::Vector)
    
Create a Zonotope representing the hypercube from low to high.
"""
function create_zonotope_from_bounds(low::Vector, high::Vector)
    dim = length(low)
    center = (low .+ high) ./ 2
    radius = (high .- low) ./ 2
    
    # Generator matrix is diagonal with radii on the diagonal
    G = Diagonal(radius)
    
    return Zonotope(Matrix(G), center, nothing)
end

@testset "Dense Layer Propagation Tests" begin
    
    test_seed = rand(1:999999)
    Random.seed!(test_seed)
    @info "Test seed: $(test_seed)"
    
    @testset "Basic Gemini Network Propagation" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(20:50) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create two randomized networks with same structure
        N1 = create_random_dense_network(input_dim, num_layers, layer_dims)
        N2 = create_random_dense_network(input_dim, num_layers, layer_dims)
        
        # Create Gemini Network
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds [-1, 1]^input_dim
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask to encode the property
        mid = (high .+ low) ./ 2
        distance = mid .- low
        non_zero_indices = collect(1:input_dim)  # All dimensions are non-zero
        
        verification_task = VerificationTask(
            mid, distance, non_zero_indices,
            nothing, nothing,  # distance1_secondary, middle1_secondary
            nothing, nothing,  # distance2_secondary, middle2_secondary
            nothing,           # verification_status
            1.0,              # distance_bound
            1.0               # work_share
        )
        
        # Create PropState and initialize with verification task
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        
        # Propagate through Gemini Network
        prop_state = propagate!(N_gemini, prop_state)
        
        # Extract output zonotopes
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        
        # Basic checks
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (10,)  # Output dimension should be 10
        @test size(Zout.Z₂.c) == (10,)
    end
    
    @testset "Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(400:600) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create networks with same structure
        N1 = create_random_dense_network(input_dim, num_layers, layer_dims)
        N2 = create_random_dense_network(input_dim, num_layers, layer_dims)
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask
        mid = (high .+ low) ./ 2
        distance = mid .- low
        non_zero_indices = collect(1:input_dim)
        
        verification_task = VerificationTask(
            mid, distance, non_zero_indices,
            nothing, nothing,
            nothing, nothing,
            nothing,
            1.0,
            1.0
        )
        
        # First propagation - counts allocations
        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)
        
        # Second propagation - should allocate less
        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)
        
        @info "First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "First run time: $time_1 s, Second run time: $time_2 s"
        
        # Check that second run allocates less (allowing for small variance)
        @test alloc_2 < alloc_1 * 0.4
    end
    
    @testset "Sampled Points Within Output Bounds" begin
        input_dim = 10
        num_layers = rand(5:15)
        num_samples = 20_000
        tolerance = 1e-4
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(20:100) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create networks with same structure
        N1 = create_random_dense_network(input_dim, num_layers, layer_dims)
        N2 = create_random_dense_network(input_dim, num_layers, layer_dims)
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds [-1, 1]^input_dim
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask
        mid = (high .+ low) ./ 2
        distance = mid .- low
        non_zero_indices = collect(1:input_dim)
        
        verification_task = VerificationTask(
            mid, distance, non_zero_indices,
            nothing, nothing,
            nothing, nothing,
            nothing,
            1.0,
            1.0
        )
        
        # Propagate zonotope through network
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        
        # Get output bounds from zonotope
        bounds_z1 = zono_bounds(Zout.Z₁)
        bounds_z2 = zono_bounds(Zout.Z₂)
        bounds_z∂ = zono_bounds(Zout.∂Z)
        # @info "Output bounds for Network 1: $bounds_z1"
        # @info "Output bounds for Network 2: $bounds_z2"
        # @info "Output bounds for Difference Zonotope: $bounds_z∂"
        
        # Sample points from input space and propagate through both networks
        input_samples = sample_points_in_hypercube(low, high, num_samples)
        
        violations = 0
        
        for i in 1:num_samples
            x = input_samples[:, i]
            
            # Propagate through N1
            y1 = N1(x)
            
            # Propagate through N2
            y2 = N2(x)
            
            # Check if outputs are within bounds
            for d in 1:10
                if y1[d] < bounds_z1[d, 1] + tolerance || y1[d] > bounds_z1[d, 2] - tolerance
                    violations += 1
                end
                
                if y2[d] < bounds_z2[d, 1] + tolerance || y2[d] > bounds_z2[d, 2] - tolerance
                    violations += 1
                end
                if (y1[d] - y2[d]) < bounds_z∂[d, 1] + tolerance || (y1[d] - y2[d]) > bounds_z∂[d, 2] - tolerance
                    violations += 1
                end
            end
        end
        
        @info "Found $violations bound violations out of $(num_samples * 30) total checks"
        
        # All sampled points should be within bounds
        @test violations == 0
    end
    
end

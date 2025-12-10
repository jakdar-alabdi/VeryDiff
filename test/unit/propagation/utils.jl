using VeryDiff
using LinearAlgebra
using Random

"""
    create_random_dense_network(input_dim::Int, layer_dims::Vector{Int})

Create a randomized neural network with only Dense layers using the provided layer dimensions.
"""
function create_random_dense_network(input_dim::Int, layer_dims::Vector{Int}; relu=false)
    layers = Layer[]
    cur_dim = input_dim
    for new_dim in layer_dims
        W = 0.1*randn(Float64, (new_dim, cur_dim))
        b = 0.1*randn(Float64, new_dim)
        # Set some rows to zero
        zero_one_rows = randn(new_dim) .< -2.5
        W[zero_one_rows, :] .= 0.0
        b[zero_one_rows] .= 0.0
        # Set some components to zero
        zero_one_components = randn(size(W)) .< -3.0
        W[zero_one_components] .= 0.0
        push!(layers, Dense(W, b))
        if relu
            push!(layers, ReLU())
        end
        cur_dim = new_dim
    end
    return Network(layers)
end

"""
    create_randon_network_mutant(network :: Network)

Create a mutated version of the provided network layers by slightly perturbing weights and biases according to different strategies.
"""
function create_random_network_mutant(network :: Network)
    new_layers = Layer[]
    for layer in network.layers
        push!(new_layers, create_random_layer_mutant(layer))
    end
    return Network(new_layers)
end

"""
    create_random_layer_mutant(layer :: Layer)
Create a mutated version of the provided layer by slightly perturbing weights and biases according to different strategies.
"""
function create_random_layer_mutant(layer :: Dense)
    mutation_type = rand(1:4)
    if mutation_type == 1
        @debug "Independent layer"
        # New random weights and biases
        W_new = 0.1*randn(Float64, size(layer.W))
        b_new = 0.1*randn(Float64, size(layer.b))
    elseif mutation_type == 2
        @debug "Zeroed components"
        # Set some weights / biases to zero
        W_new = deepcopy(layer.W)
        b_new = deepcopy(layer.b)
        W_mask = randn(size(W_new)) .< -2.0
        b_mask = randn(size(b_new)) .< -2.0
        W_new[W_mask] .= 0.0
        b_new[b_mask] .= 0.0
    elseif mutation_type == 3
        @debug "Pruned rows"
        # Prune some rows
        W_new = deepcopy(layer.W)
        b_new = deepcopy(layer.b)
        row_mask = randn(size(W_new, 1)) .< -2.0
        W_new[row_mask, :] .= 0.0
        b_new[row_mask] .= 0.0
    elseif mutation_type == 4
        @debug "Small perturbation"
        # Small random perturbation
        W_new = layer.W .+ 0.01*randn(Float64, size(layer.W))
        b_new = layer.b .+ 0.01*randn(Float64, size(layer.b))
    else
        error("Unknown mutation type")
    end
    return Dense(W_new, b_new)
end

function create_random_layer_mutant(layer :: ReLU)
    return deepcopy(ReLU())
end

"""
    make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false)

Create two networks sharing the same architecture. When `identical` is true, the weights/biases are identical.
"""
function make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false, relu=false)
    N1 = create_random_dense_network(input_dim, layer_dims; relu=relu)
    N2 = identical ? deepcopy(N1) : create_random_network_mutant(N1)
    return N1, N2
end

"""
    sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int)

Sample uniformly from the hypercube defined by `low` and `high` bounds.
"""
function sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int; secondary_low=nothing, secondary_high=nothing)
    dim = length(low)
    total_dim = dim
    use_secondary = !isnothing(secondary_low) && !isnothing(secondary_high)
    if use_secondary
        @assert length(secondary_low) == length(secondary_high)
        total_dim += length(secondary_low)
    end
    samples = zeros(Float64, total_dim, num_samples)
    for i in 1:num_samples
        samples[1:dim, i] = low .+ (high .- low) .* rand(Float64, dim)
        if use_secondary
            samples[dim+1:end, i] = secondary_low .+ (secondary_high .- secondary_low) .* rand(Float64, length(secondary_low))
        end
    end
    return samples
end

"""
    create_verification_task(low::Vector, high::Vector; with_secondary::Bool=false, secondary_scale::Float64=1.0)

Helper to build a `VerificationTask` for bounds `low`..`high`. When `with_secondary` is true, secondary distances mirror the primary span (scaled by `secondary_scale`).
"""
function create_verification_task(low::Vector, high::Vector; with_secondary::Bool=false, secondary_scale::Float64=1.0)
    mid = (high .+ low) ./ 2
    distance = mid .- low
    non_zero_indices = collect(1:length(low))
    if with_secondary
        dist1 = ones(size(low)) .* secondary_scale
        mid1 = zeros(Float64, length(low))
        dist2 = ones(size(low)) .* secondary_scale
        mid2 = zeros(Float64, length(low))
    else
        dist1 = nothing
        mid1 = nothing
        dist2 = nothing
        mid2 = nothing
    end
    return VerificationTask(
        mid, distance, non_zero_indices,
        dist1, mid1,
        dist2, mid2,
        nothing,
        1.0,
        1.0
    )
end

using VeryDiff
using LinearAlgebra
using Random

"""
    create_random_dense_network(input_dim::Int, layer_dims::Vector{Int})

Create a randomized neural network with only Dense layers using the provided layer dimensions.
Returns both a Network and an OnnxNet representation.
"""
function create_random_dense_network(input_dim::Int, layer_dims::Vector{Int}; relu=false, add_const=false)
    layers = Vector{Node{String}}()
    layer_metadata = Vector{Tuple{String, Int}}()  # Track (layer_name, mutation_type) for synchronization
    cur_dim = input_dim
    layer_count = 0
    prev_output_id = "network_input"
    
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
        layer_count += 1
        input_id = prev_output_id
        output_id = "output_$layer_count"
        push!(layers, ONNXLinear([input_id], [output_id], "dense_$layer_count", W, b))
        prev_output_id = output_id
        
        if add_const
            layer_count += 1
            addconst_input_id = output_id
            addconst_output_id = "output_$layer_count"
            # Create a non-zero constant to add (random values in range [-0.5, 0.5])
            const_vec = 0.1 .* randn(Float64, new_dim)
            # Ensure at least one non-zero element
            if all(c ≈ 0 for c in const_vec)
                const_vec[1] = 0.1
            end
            push!(layers, ONNXAddConst([addconst_input_id], [addconst_output_id], "addconst_$layer_count", const_vec))
            prev_output_id = addconst_output_id
        end
        
        if relu
            layer_count += 1
            relu_input_id = prev_output_id
            relu_output_id = "output_$layer_count"
            push!(layers, ONNXRelu([relu_input_id], [relu_output_id], "relu_$layer_count"))
            prev_output_id = relu_output_id
        end
        cur_dim = new_dim
    end
    
    # Create OnnxNet with appropriate input/output shapes and node structure
    node_dict = Dict{String, Node{String}}()
    for layer in layers
        node_dict[layer.name] = layer
    end
    
    start_nodes = ["dense_1"]
    final_nodes = [layers[end].name]
    input_shapes = Dict("network_input" => (input_dim,))
    output_shapes = Dict(layers[end].outputs[1] => (layer_dims[end],))
    
    onnx_net = OnnxNet(layers, start_nodes, final_nodes, input_shapes, output_shapes)
    
    return onnx_net
end

"""
    create_random_network_mutant(onnx_net :: OnnxNet)

Create a mutated version of the provided ONNX network by mutating every ONNXLinear layer.
Synchronizes mutations between ONNXLinear and corresponding ONNXAddConst layers.
"""
function create_random_network_mutant(onnx_net :: OnnxNet)
    new_layers = Vector{Node{String}}()
    # Store mutation types for each dense layer to sync with addconst
    mutation_map = Dict{String, Int}()
    
    # First pass: determine mutation types for dense layers
    for (node_name, node) in onnx_net.nodes
        if node isa ONNXLinear{String} && startswith(node_name, "dense_")
            mutation_map[node_name] = rand(1:4)
        end
    end
    
    # Second pass: apply mutations
    for (node_name, node) in onnx_net.nodes
        if node isa ONNXLinear{String}
            mutation_type = get(mutation_map, node_name, rand(1:4))
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        elseif node isa ONNXAddConst{String}
            # Find corresponding dense layer via node_prevs topology
            mutation_type = rand(1:4)  # default
            if haskey(onnx_net.node_prevs, node_name)
                for prev_name in onnx_net.node_prevs[node_name]
                    if haskey(mutation_map, prev_name)
                        mutation_type = mutation_map[prev_name]
                        break
                    end
                end
            end
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        else
            push!(new_layers, create_random_layer_mutant(node))
        end
    end
    
    # Reconstruct OnnxNet with mutated layers
    start_nodes = onnx_net.start_nodes
    final_nodes = onnx_net.final_nodes
    input_shapes = onnx_net.input_shapes
    output_shapes = onnx_net.output_shapes
    
    return OnnxNet(new_layers, start_nodes, final_nodes, input_shapes, output_shapes)
end

"""
    create_random_layer_mutant(layer :: ONNXLinear, mutation_type :: Int)
Create a mutated version of the provided layer using the specified mutation type.
"""
function create_random_layer_mutant(layer :: ONNXLinear{String}, mutation_type :: Int)
    if mutation_type == 1
        @debug "Independent layer"
        # New random weights and biases
        W_new = 0.1*randn(Float64, size(layer.dense.weight))
        b_new = 0.1*randn(Float64, size(layer.dense.bias))
    elseif mutation_type == 2
        @debug "Zeroed components"
        # Set some weights / biases to zero
        W_new = deepcopy(layer.dense.weight)
        b_new = deepcopy(layer.dense.bias)
        W_mask = randn(size(W_new)) .< -2.0
        b_mask = randn(size(b_new)) .< -2.0
        W_new[W_mask] .= 0.0
        b_new[b_mask] .= 0.0
    elseif mutation_type == 3
        @debug "Pruned rows"
        # Prune some rows
        W_new = deepcopy(layer.dense.weight)
        b_new = deepcopy(layer.dense.bias)
        row_mask = randn(size(W_new, 1)) .< -2.0
        W_new[row_mask, :] .= 0.0
        b_new[row_mask] .= 0.0
    elseif mutation_type == 4
        @debug "Small perturbation"
        # Small random perturbation
        W_new = layer.dense.weight .+ 0.01*randn(Float64, size(layer.dense.weight))
        b_new = layer.dense.bias .+ 0.01*randn(Float64, size(layer.dense.bias))
    else
        error("Unknown mutation type")
    end
    return ONNXLinear(layer.inputs, layer.outputs, layer.name, W_new, b_new, transpose=layer.transpose)
end

"""
    create_random_layer_mutant(layer :: ONNXAddConst, mutation_type :: Int)
Create a mutated version of the AddConst layer using the specified mutation type (synchronized with linear layer).
"""
function create_random_layer_mutant(layer :: ONNXAddConst{String}, mutation_type :: Int)
    if mutation_type == 1
        @debug "Independent AddConst layer"
        # New random constant
        c_new = 0.1 .* randn(Float64, size(layer.c))
        if all(c ≈ 0 for c in c_new)
            c_new[1] = 0.1
        end
    elseif mutation_type == 2
        @debug "Zeroed components in AddConst"
        # Set some components to zero
        c_new = deepcopy(layer.c)
        c_mask = randn(size(c_new)) .< -2.0
        c_new[c_mask] .= 0.0
    elseif mutation_type == 3
        @debug "Zeroed all components in AddConst (pruned)"
        # Zero out the entire constant
        c_new = zeros(size(layer.c))
    elseif mutation_type == 4
        @debug "Small perturbation in AddConst"
        # Small random perturbation
        c_new = layer.c .+ 0.01*randn(Float64, size(layer.c))
    else
        error("Unknown mutation type")
    end
    return ONNXAddConst(layer.inputs, layer.outputs, layer.name, c_new)
end

"""
    create_random_layer_mutant(layer :: Layer)
Create a mutated version of the provided layer by slightly perturbing weights and biases according to different strategies.
"""
function create_random_layer_mutant(layer :: ONNXLinear{String})
    mutation_type = rand(1:4)
    return create_random_layer_mutant(layer, mutation_type)
end

function create_random_layer_mutant(layer :: ONNXRelu{String})
    return deepcopy(layer)
end

function create_random_layer_mutant(layer :: ONNXAddConst{String})
    mutation_type = rand(1:4)
    return create_random_layer_mutant(layer, mutation_type)
end

function create_random_layer_mutant(layer :: ONNXAdd{String})
    return deepcopy(layer)
end

"""
    make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false)

Create two networks sharing the same architecture. When `identical` is true, the weights/biases are identical.
Returns tuples of (Network, OnnxNet) pairs.
"""
function make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false, relu=false, add_const=false)
    N1_onnx = create_random_dense_network(input_dim, layer_dims; relu=relu, add_const=add_const)
    if identical
        N2_onnx = deepcopy(N1_onnx)
    else
        N2_onnx = create_random_network_mutant(N1_onnx)
    end
    return N1_onnx, N2_onnx
end

"""
    create_mutant_layers(layers::Vector{Node{String}})

Helper function to create mutated versions of layers.
"""
function create_mutant_layers(layers::Vector{Node{String}})
    new_layers = Vector{Node{String}}()
    for layer in layers
        push!(new_layers, create_random_layer_mutant(layer))
    end
    return new_layers
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


"""
    create_random_add_network(input_dim::Int, layer_dims::Vector{Int}; topology=:fork_join, relu=false, add_const=false)

Create a randomized neural network with ONNXAdd nodes forming a DAG structure.
Supported topologies:
- `:fork_join` — Input → Dense₁ → (BranchA: Dense₂, BranchB: Dense₃) → ONNXAdd → Dense₄ → Output
- `:fork_join_relu` — Same as fork_join but with ReLU and optionally AddConst after each Dense
- `:resnet` — ResNet-style skip connections: multiple residual blocks each adding the block output to its input
"""
function create_random_add_network(input_dim::Int, layer_dims::Vector{Int}; topology::Symbol=:fork_join, relu::Bool=false, add_const::Bool=false)
    if topology == :fork_join
        return _create_fork_join_network(input_dim, layer_dims; relu=relu, add_const=add_const)
    elseif topology == :resnet
        return _create_resnet_network(input_dim, layer_dims; relu=relu, add_const=add_const)
    else
        error("Unknown topology: $topology")
    end
end

"""
Helper: Add optional AddConst and ReLU layers after a Dense layer.
Returns the updated (layers, layer_count, prev_output_id).
"""
function _add_post_dense_layers!(layers, layer_count, prev_output_id, new_dim; relu=false, add_const=false)
    if add_const
        layer_count += 1
        addconst_input_id = prev_output_id
        addconst_output_id = "output_$layer_count"
        const_vec = 0.1 .* randn(Float64, new_dim)
        if all(c ≈ 0 for c in const_vec)
            const_vec[1] = 0.1
        end
        push!(layers, ONNXAddConst([addconst_input_id], [addconst_output_id], "addconst_$layer_count", const_vec))
        prev_output_id = addconst_output_id
    end

    if relu
        layer_count += 1
        relu_input_id = prev_output_id
        relu_output_id = "output_$layer_count"
        push!(layers, ONNXRelu([relu_input_id], [relu_output_id], "relu_$layer_count"))
        prev_output_id = relu_output_id
    end

    return layers, layer_count, prev_output_id
end

"""
Helper: Create a Dense layer with optional AddConst and ReLU.
Returns the updated (layers, layer_count, prev_output_id).
"""
function _add_dense_block!(layers, layer_count, prev_output_id, cur_dim, new_dim; relu=false, add_const=false, name_prefix="dense")
    W = 0.1*randn(Float64, (new_dim, cur_dim))
    b = 0.1*randn(Float64, new_dim)
    # Set some rows to zero
    zero_one_rows = randn(new_dim) .< -2.5
    W[zero_one_rows, :] .= 0.0
    b[zero_one_rows] .= 0.0
    # Set some components to zero
    zero_one_components = randn(size(W)) .< -3.0
    W[zero_one_components] .= 0.0

    layer_count += 1
    input_id = prev_output_id
    output_id = "output_$layer_count"
    push!(layers, ONNXLinear([input_id], [output_id], "$(name_prefix)_$layer_count", W, b))
    prev_output_id = output_id

    layers, layer_count, prev_output_id = _add_post_dense_layers!(layers, layer_count, prev_output_id, new_dim; relu=relu, add_const=add_const)

    return layers, layer_count, prev_output_id
end

"""
Fork-Join topology:
  Input → Dense_stem → [BranchA: Dense_a, BranchB: Dense_b] → ONNXAdd → Dense_out₁ → ... → Dense_outN → Output

layer_dims specifies: [stem_dim, branch_dim, out_dim₁, ..., out_dimN]
(minimum 3 elements)
"""
function _create_fork_join_network(input_dim::Int, layer_dims::Vector{Int}; relu::Bool=false, add_const::Bool=false)
    @assert length(layer_dims) >= 3 "Fork-join requires at least 3 layer dimensions: [stem, branch, output...]"
    layers = Vector{Node{String}}()
    layer_count = 0

    stem_dim = layer_dims[1]
    branch_dim = layer_dims[2]

    # Stem: input → Dense_stem
    layers, layer_count, stem_out = _add_dense_block!(layers, layer_count, "network_input", input_dim, stem_dim; relu=relu, add_const=add_const, name_prefix="dense_stem")

    # Branch A: stem → Dense_a
    layers, layer_count, branch_a_out = _add_dense_block!(layers, layer_count, stem_out, stem_dim, branch_dim; relu=relu, add_const=add_const, name_prefix="dense_branch_a")

    # Branch B: stem → Dense_b
    layers, layer_count, branch_b_out = _add_dense_block!(layers, layer_count, stem_out, stem_dim, branch_dim; relu=relu, add_const=add_const, name_prefix="dense_branch_b")

    # ONNXAdd: branch_a + branch_b
    layer_count += 1
    add_output_id = "output_$layer_count"
    push!(layers, ONNXAdd([branch_a_out, branch_b_out], [add_output_id], "add_$layer_count"))
    prev_output_id = add_output_id
    cur_dim = branch_dim

    # Output layers
    for i in 3:length(layer_dims)
        new_dim = layer_dims[i]
        layers, layer_count, prev_output_id = _add_dense_block!(layers, layer_count, prev_output_id, cur_dim, new_dim; relu=(relu && i < length(layer_dims)), add_const=(add_const && i < length(layer_dims)), name_prefix="dense_out")
        cur_dim = new_dim
    end

    # Build OnnxNet
    start_nodes = [layers[1].name]
    final_nodes = [layers[end].name]
    input_shapes = Dict("network_input" => (input_dim,))
    output_shapes = Dict(layers[end].outputs[1] => (layer_dims[end],))

    return OnnxNet(layers, start_nodes, final_nodes, input_shapes, output_shapes)
end

"""
ResNet-style topology with multiple residual blocks:
  Input → Dense₁ → [residual block₁: Dense_res + ONNXAdd(skip)] → ... → [residual blockN: Dense_res + ONNXAdd(skip)] → Dense_out → Output

Each residual block: x → Dense(x) → (optional ReLU/AddConst) → Add(result, x)

layer_dims specifies: [hidden_dim₁, hidden_dim₂, ..., output_dim]
All hidden dimensions must be equal (since skip connections require matching dimensions).
The last dimension is the output dimension (with a projection dense layer).
"""
function _create_resnet_network(input_dim::Int, layer_dims::Vector{Int}; relu::Bool=false, add_const::Bool=false)
    @assert length(layer_dims) >= 2 "ResNet requires at least 2 layer dimensions: [hidden_dim..., output_dim]"
    hidden_dim = layer_dims[1]
    for i in 1:(length(layer_dims)-1)
        @assert layer_dims[i] == hidden_dim "All hidden dimensions in ResNet must be equal, got $(layer_dims[i]) != $hidden_dim at position $i"
    end

    layers = Vector{Node{String}}()
    layer_count = 0

    # Projection: input → hidden_dim
    layers, layer_count, prev_output_id = _add_dense_block!(layers, layer_count, "network_input", input_dim, hidden_dim; relu=relu, add_const=add_const, name_prefix="dense_proj")
    cur_dim = hidden_dim

    # Residual blocks
    num_res_blocks = length(layer_dims) - 1
    for block_idx in 1:num_res_blocks
        skip_input = prev_output_id

        # Dense inside residual block
        layers, layer_count, block_out = _add_dense_block!(layers, layer_count, prev_output_id, cur_dim, hidden_dim; relu=relu, add_const=add_const, name_prefix="dense_res$(block_idx)")

        # ONNXAdd: block_out + skip_input
        layer_count += 1
        add_output_id = "output_$layer_count"
        push!(layers, ONNXAdd([block_out, skip_input], [add_output_id], "add_res$(block_idx)_$layer_count"))
        prev_output_id = add_output_id
    end

    # Final output projection (no relu/addconst on last layer)
    output_dim = layer_dims[end]
    layers, layer_count, prev_output_id = _add_dense_block!(layers, layer_count, prev_output_id, cur_dim, output_dim; relu=false, add_const=false, name_prefix="dense_final")

    # Build OnnxNet
    start_nodes = [layers[1].name]
    final_nodes = [layers[end].name]
    input_shapes = Dict("network_input" => (input_dim,))
    output_shapes = Dict(layers[end].outputs[1] => (output_dim,))

    return OnnxNet(layers, start_nodes, final_nodes, input_shapes, output_shapes)
end

"""
    create_random_add_network_mutant(onnx_net::OnnxNet)

Create a mutated version of an OnnxNet that may contain ONNXAdd nodes.
Mutates ONNXLinear and ONNXAddConst layers using the same strategy as create_random_network_mutant.
ONNXAdd and ONNXRelu layers are kept identical.
"""
function create_random_add_network_mutant(onnx_net::OnnxNet)
    new_layers = Vector{Node{String}}()
    # Determine mutation types for dense layers
    mutation_map = Dict{String, Int}()
    for (node_name, node) in onnx_net.nodes
        if node isa ONNXLinear{String} && occursin("dense", node_name)
            mutation_map[node_name] = rand(1:4)
        end
    end

    # Iterate in topological order (use the original layer list order)
    # We need to iterate the nodes in a consistent order
    # OnnxNet stores nodes as a Dict, but we can use topological order from node_prevs/node_nexts
    visited = Set{String}()
    queue = copy(onnx_net.start_nodes)
    ordered_nodes = String[]
    visit_count = Dict{String, Int}()
    for name in keys(onnx_net.nodes)
        visit_count[name] = 0
    end

    while !isempty(queue)
        node_name = popfirst!(queue)
        if node_name in visited
            continue
        end
        push!(visited, node_name)
        push!(ordered_nodes, node_name)
        if haskey(onnx_net.node_nexts, node_name)
            for next_name in onnx_net.node_nexts[node_name]
                visit_count[next_name] += 1
                if visit_count[next_name] >= length(onnx_net.node_prevs[next_name])
                    push!(queue, next_name)
                end
            end
        end
    end

    for node_name in ordered_nodes
        node = onnx_net.nodes[node_name]
        if node isa ONNXLinear{String}
            mutation_type = get(mutation_map, node_name, rand(1:4))
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        elseif node isa ONNXAddConst{String}
            # Find corresponding dense layer via node_prevs topology
            mutation_type = rand(1:4)  # default
            if haskey(onnx_net.node_prevs, node_name)
                for prev_name in onnx_net.node_prevs[node_name]
                    if haskey(mutation_map, prev_name)
                        mutation_type = mutation_map[prev_name]
                        break
                    end
                end
            end
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        else
            push!(new_layers, create_random_layer_mutant(node))
        end
    end

    start_nodes = onnx_net.start_nodes
    final_nodes = onnx_net.final_nodes
    input_shapes = onnx_net.input_shapes
    output_shapes = onnx_net.output_shapes

    return OnnxNet(new_layers, start_nodes, final_nodes, input_shapes, output_shapes)
end

"""
    make_add_pair(input_dim::Int, layer_dims::Vector{Int}; topology=:fork_join, identical=false, relu=false, add_const=false)

Create two networks with ONNXAdd nodes sharing the same architecture.
When `identical` is true, the weights/biases are identical.
"""
function make_add_pair(input_dim::Int, layer_dims::Vector{Int}; topology::Symbol=:fork_join, identical::Bool=false, relu::Bool=false, add_const::Bool=false)
    N1_onnx = create_random_add_network(input_dim, layer_dims; topology=topology, relu=relu, add_const=add_const)
    if identical
        N2_onnx = deepcopy(N1_onnx)
    else
        N2_onnx = create_random_add_network_mutant(N1_onnx)
    end
    return N1_onnx, N2_onnx
end

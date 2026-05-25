using VeryDiff
using VNNLib
using VNNLib.OnnxParser: Node, ONNXLinear, ONNXRelu, ONNXAddConst

function create_random_dense_layer(input_id::String, output_id::String, layer_name::String, rows::Int, columns::Int)
    W₁ = 0.1 * randn(rows, columns)
    b₁ = 0.1 * randn(rows)
    # Set some rows to zero
    zero_one_rows = randn(rows) .< -2.5
    W₁[zero_one_rows, :] .= 0.0
    b₁[zero_one_rows] .= 0.0
    # Set some components to zero
    zero_one_components = randn(rows, columns) .< -3.0
    W₁[zero_one_components] .= 0.0

    layer_type = rand(2:4)
    if layer_type == 1
        @info "Independent layer"
        # New random weights and biases
        W₂ = 0.1 * randn(rows, columns)
        b₂ = 0.1 * randn(rows)
    elseif layer_type == 2
        @info "Zeroed components"
        # Set some weights / biases to zero
        W₂ = deepcopy(W₁)
        b₂ = deepcopy(b₁)
        W_mask = randn(rows, columns) .< -2.0
        b_mask = randn(rows) .< -2.0
        W₂[W_mask] .= 0.0
        b₂[b_mask] .= 0.0
    elseif layer_type == 3
        @info "Pruned rows"
        # Prune some rows
        W₂ = deepcopy(W₁)
        b₂ = deepcopy(b₁)
        row_mask = randn(size(W₂, 1)) .< -2.0
        W₂[row_mask, :] .= 0.0
        b₂[row_mask] .= 0.0
    else
        @info "Small perturbation"
        # Small random perturbation
        W₂ = W₁ .+ 0.01 * randn(Float64, size(W₁))
        b₂ = b₁ .+ 0.01 * randn(Float64, size(b₁))
    end

    layer₁ = ONNXLinear([input_id], [output_id], layer_name, W₁, b₁)
    layer₂ = ONNXLinear([input_id], [output_id], layer_name, W₂, b₂)
    return layer₁, layer₂
end

function create_random_addconst_layer(input_id::String, output_id::String, layer_name::String, dim::Int)
    # Create a non-zero constant to add (random values in range [-0.5, 0.5])
    c₁ = 0.1 .* randn(dim)
    # Ensure at least one non-zero element
    if all(c ≈ 0 for c in c₁)
        c₁[1] = 0.1
    end

    layer_type = rand(1:4)
    if layer_type == 1
        @info "Independent layer"
        # New random weights and biases
        c₂ = 0.1 .* randn(dim)
        if all(c ≈ 0 for c in c₂)
            c₂[1] = 0.1
        end
    elseif layer_type == 2
        @info "Zeroed components"
        # Set some weights / biases to zero
        c₂ = deepcopy(c₁)
        c_mask = randn(dim) .< -2.0
        c₂[c_mask] .= 0.0
    elseif layer_type == 3
        @info "Pruned rows"
        # Prune some rows
        c₂ = zeros(dim)
    else
        @info "Small perturbation"
        # Small random perturbation
        c₂ = c₁ .+ 0.01 * randn(dim)
    end

    layer₁ = ONNXAddConst([input_id], [output_id], layer_name, c₁)
    layer₂ = ONNXAddConst([input_id], [output_id], layer_name, c₂)
    return layer₁, layer₂
end

function create_random_networks(num_layers::Int, input_dim::Int, output_dim::Int)
    cur_dim = input_dim

    layers₁ = Vector{Node{String}}()
    layers₂ = Vector{Node{String}}()
    prev_output_id = "network_input"
    
    for i in 1:3:num_layers
        if i == num_layers
            new_dim = output_dim
        else
            new_dim = rand(2:100)
        end

        input_id = prev_output_id
        output_id = "output_$i"
        next_dense_layer₁, next_dense_layer₂ = create_random_dense_layer(input_id, output_id, "dense_$(i)", new_dim, cur_dim)
        
        input_id = output_id
        output_id = "output_$(i + 1)"
        next_addconst_layer₁, next_addconst_layer₂ = create_random_addconst_layer(input_id, output_id, "addconst_$(i + 1)", new_dim)
        
        input_id = output_id
        output_id = "output_$(i + 2)"
        next_relu_layer₁ = ONNXRelu([input_id], [output_id], "relu_$(i + 2)")
        next_relu_layer₂ = ONNXRelu([input_id], [output_id], "relu_$(i + 2)")

        push!(layers₁, next_dense_layer₁, next_addconst_layer₁, next_relu_layer₁)
        push!(layers₂, next_dense_layer₂, next_addconst_layer₂, next_relu_layer₂)
        prev_output_id = output_id
        cur_dim = new_dim
    end
    
    start_nodes₁, start_nodes₂ = ["dense_1"], ["dense_1"]
    final_nodes₁, final_nodes₂ = [layers₁[end].name], [layers₂[end].name]
    input_shapes₁, input_shapes₂ = Dict("network_input" => (input_dim,)), Dict("network_input" => (input_dim,))
    output_shapes₁, output_shapes₂ = Dict(layers₁[end].outputs[1] => (output_dim,)), Dict(layers₂[end].outputs[1] => (output_dim,))
    
    onnx_net₁ = OnnxNet(layers₁, start_nodes₁, final_nodes₁, input_shapes₁, output_shapes₁)
    onnx_net₂ = OnnxNet(layers₂, start_nodes₂, final_nodes₂, input_shapes₂, output_shapes₂)
    
    return onnx_net₁, onnx_net₂
end

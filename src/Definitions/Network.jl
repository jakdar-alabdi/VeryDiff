

struct OnnxLayer{LayerIdT} <: Node{LayerIdT}
    layer_id :: LayerIdT
    node :: Node{LayerIdT}
    input_ids :: Vector{Int64}
end

function cleanup_network(layers :: Vector{OnnxLayer{LayerIdT}}) :: Vector{OnnxLayer{LayerIdT}} where {LayerIdT}
    valid_layers = []
    for i in 1:length(layers)
        if layers[i].node isa ONNXLinear{LayerIdT}
            W = layers[i].node.dense.weight
            b = layers[i].node.dense.bias
            f = layers[i].node.dense.σ
            if all(isone.(diag(W))) && all([all(iszero.(diag(W, k))) && all(iszero.(diag(W, -k))) for k in 1:size(W,1)-1]) && all(iszero.(b)) && f == identity
                @info "Removing identity Dense layer at index $i"
                continue
            end
        end
        push!(valid_layers, i)
    end
    print(valid_layers)
    return layers[valid_layers]
end

struct DiffLayer{L1<:Node, Ld<:Node, L2<:Node}
    layer_idx :: Int64
    inputs :: Vector{Int64}
    layer1 :: L1
    diff_layer :: Ld
    layer2 :: L2
    function DiffLayer(layer_idx :: Int64, inputs :: Vector{Int64}, layer1 :: L1, diff_layer :: Ld, layer2 :: L2) where {L1<:Node, Ld<:Node, L2<:Node}
        return new{L1,Ld,L2}(layer_idx, inputs, layer1, diff_layer, layer2)
    end
end

struct ZeroDense{LayerIdT} <: Node{LayerIdT}
end

function topological_sort!(network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}, node_id :: LayerIdT, visited :: Set{LayerIdT}, sorting :: Dict{LayerIdT, Int64}) where {LayerIdT,NShapeIn,NShapeOut}
    if node_id in visited
        return
    end
    push!(visited, node_id)
    for next_node in network.node_nexts[node_id]
        topological_sort!(network, next_node, visited, sorting)
    end
    sorting[node_id] = length(sorting) + 1
end

function sort_network(network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) :: Vector{OnnxLayer{LayerIdT}} where {LayerIdT,NShapeIn,NShapeOut}
    @assert length(network.input_shapes) == 1 "Currently only single input networks are supported"
    @assert length(network.output_shapes) == 1 "Currently only single output networks are supported"
    sorting = Dict{LayerIdT, Int64}()
    visited = Set{LayerIdT}()
    for input_node in network.start_nodes
        topological_sort!(network, input_node, visited, sorting)
    end
    num_nodes = length(sorting)
    # Map i to num_nodes - i + 1
    for (k,v) in sorting
        sorting[k] = num_nodes - v + 1
    end
    layers = OnnxLayer{LayerIdT}[]
    for (node_id, node) in network.nodes
        input_ids = Int64[]
        for input_id in node.inputs
            if haskey(network.output_dict, input_id)
                push!(input_ids, sorting[network.output_dict[input_id]])
            else 
                # Input node (currently only one input supported)
                @assert haskey(network.input_shapes, input_id) "Input node $input_id not found in network inputs"
                push!(input_ids, 0)
            end
        end
        push!(layers, OnnxLayer{LayerIdT}(node_id, node, input_ids))
    end
    sorted_layers = sort(layers, by = x -> sorting[x.layer_id])
    return sorted_layers
end

function sort_mirror_network(layers :: Vector{OnnxLayer{LayerIdT}}, network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) :: Vector{Node{LayerIdT}} where {LayerIdT,NShapeIn,NShapeOut}
    sorted_layers = Node{LayerIdT}[]
    for cur_layer in layers
        cur_node = network.nodes[cur_layer.layer_id]
        inputs = Int64[]
        for input_node in cur_node.inputs
            if haskey(network.output_dict, input_node)
                push!(inputs, findfirst(x -> x.layer_id == network.output_dict[input_node], layers))
            else
                # Input node (currently only one input supported)
                @assert haskey(network.input_shapes, input_node) "Input node $input_node not found in network inputs"
                push!(inputs, 0)
            end
        end
        push!(sorted_layers, OnnxLayer{LayerIdT}(cur_layer.layer_id, cur_node, inputs))
    end
    return sorted_layers
end

function mk_dense(W::AbstractMatrix{Float64}, b::AbstractVector{Float64}) 
    return ONNXLinear{Int64}(Dense(W, b, identity))
end


struct GeminiNetwork
    network1 :: OnnxNet
    network2 :: OnnxNet
    diff_layers :: Vector{DiffLayer}
    function GeminiNetwork(network1 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}, network2 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) where {LayerIdT,NShapeIn,NShapeOut}
        n1_layers = sort_network(network1)
        n2_layers = sort_mirror_network(n1_layers, network2)
        diff_layers = DiffLayer[]
        if length(n1_layers) > length(n2_layers)
            n1_layers = cleanup_network(network1)
        elseif length(n2_layers) > length(n1_layers)
            n2_layers = cleanup_network(network2)
        end
        @assert length(n1_layers) == length(n2_layers) "Networks have different number of layers after cleanup: $(length(n1_layers)) vs $(length(n2_layers))"
        for (idx, (l1, l2)) in enumerate(zip(n1_layers, n2_layers))
            @assert typeof(l1.node) == typeof(l2.node) "Mismatch in layer types: $(typeof(l1.node)) vs $(typeof(l2.node))"
            @assert l1.input_ids == l2.input_ids "Mismatch in input ids for layers at index $(l1.layer_id) and $(l2.layer_id)"
            if typeof(l1.node) == ONNXLinear{LayerIdT}
                W1 = l1.node.dense.weight
                b1 = l1.node.dense.bias
                W2 = l2.node.dense.weight
                b2 = l2.node.dense.bias
                f1 = l1.node.dense.σ
                f2 = l2.node.dense.σ
                @assert f1 == identity "Unsupported activation function in network1: $f1"
                @assert f2 == identity "Unsupported activation function in network2: $f2"
                @assert size(W1) == size(W2) "Mismatch in weight matrix size: $(size(W1)) vs $(size(W2))"
                @assert size(b1) == size(b2)
                new_W = W1 .- W2
                new_b = b1 .- b2
                if all(iszero.(new_W)) && all(iszero.(new_b))
                    @info "Detected zero difference in Dense layer, replacing with ZeroDense layer."
                    diff_l = ZeroDense{LayerIdT}()
                else
                    diff_l = mk_dense(new_W, new_b)
                end
                push!(diff_layers, DiffLayer(idx + 1, map(x->x+1, l1.input_ids), l1.node, diff_l, l2.node))
            else
                push!(diff_layers, DiffLayer(idx + 1, map(x->x+1, l1.input_ids), l1.node, l1.node, l2.node))
            end
        end
        return new(network1, network2, diff_layers)
    end
end

function get_inputs(L :: DiffLayer{<:Node,<:Node,<:Node}) :: Vector{Int64}
    return L.inputs
end

function get_layer1(L :: DiffLayer{L1, Ld, L2}) :: L1 where {L1<:Node, Ld<:Node, L2<:Node}
    return L.layer1
end

function get_diff_layer(L :: DiffLayer{L1, Ld, L2}) :: Ld where {L1<:Node, Ld<:Node, L2<:Node}
    return L.diff_layer
end

function get_layer2(L :: DiffLayer{L1, Ld, L2}) :: L2 where {L1<:Node, Ld<:Node, L2<:Node}
    return L.layer2
end

function get_layers(N::GeminiNetwork)
    return N.diff_layers
end
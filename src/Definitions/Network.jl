
function cleanup_network(network1)
    valid_layers = []
    for i in 1:length(network1.layers)
        if network1.layers[i] isa Dense
            if all(isone.(diag(network1.layers[i].W))) && all([all(iszero.(diag(network1.layers[i].W, k))) && all(iszero.(diag(network1.layers[i].W, -k))) for k in 1:size(network1.layers[i].W,1)-1])
                continue
            end
        end
        push!(valid_layers, i)
    end
    print(valid_layers)
    @assert length(valid_layers) == length(network2.layers)
    return Network(network1.layers[valid_layers])
end

function get_input_indices(layer_idx :: Int64, :: Layer)
    # TODO: Needs to be updated for comp graphs
    return Int64[layer_idx - 1]
end

struct DiffLayer{L1<:Layer, Ld<:Layer, L2<:Layer}
    layer_idx :: Int64
    inputs :: Vector{Int64}
    layer1 :: L1
    diff_layer :: Ld
    layer2 :: L2
    function DiffLayer(layer_idx :: Int64, layer1 :: L1, diff_layer :: Ld, layer2 :: L2) where {L1<:Layer, Ld<:Layer, L2<:Layer}
        inputs = union(get_input_indices(layer_idx, layer1), get_input_indices(layer_idx, layer2))
        return new{L1,Ld,L2}(layer_idx, inputs, layer1, diff_layer, layer2)
    end
end

struct ZeroDense <: Layer
end


struct GeminiNetwork
    network1 :: Network
    network2 :: Network
    diff_network :: Network
    function GeminiNetwork(network1 :: Network, network2 :: Network)
        diff_layers = Layer[]
        if length(network1.layers) > length(network2.layers)
            network1 = cleanup_network(network1)
        elseif length(network2.layers) > length(network1.layers)
            network2 = cleanup_network(network2)
        end
        @assert length(network1.layers) == length(network2.layers)
        for (l1, l2) in zip(network1.layers, network2.layers)
            @assert typeof(l1) == typeof(l2)
            if typeof(l1) == Dense
                @assert size(l1.W) == size(l2.W) "Mismatch in weight matrix size: $(size(l1.W)) vs $(size(l2.W))"
                @assert size(l1.b) == size(l2.b)
                new_W = l1.W .- l2.W
                new_b = l1.b .- l2.b
                if all(iszero.(new_W)) && all(iszero.(new_b))
                    push!(diff_layers, ZeroDense())
                else
                    push!(diff_layers, Dense(new_W, new_b))
                    println("Distance: ", sum(abs,diff_layers[end].W))
                end
            elseif typeof(l1) == ReLU
                push!(diff_layers, ReLU())
            else
                error("Unsupported layer type")
            end
        end
        return new(network1, network2, Network(diff_layers))
    end
end

function parse_network(n::Network)
    return n
end

function get_layers(N::Network)
    return N.layers
end

function to_diff_layer(input :: Tuple{Int64,Tuple{L1,Ld,L2}}) :: DiffLayer{L1,Ld,L2} where {L1<:Layer, Ld<:Layer, L2<:Layer}
    (layer_idx,(l1,ld,l2)) = input
    # Index 1 is input => Every subsequent layer is shifted by one
    return DiffLayer(layer_idx + 1,l1,ld,l2)
end

function get_inputs(L :: DiffLayer{<:Layer,<:Layer,<:Layer}) :: Vector{Int64}
    return L.inputs
end

function get_layer1(L :: DiffLayer{L1, Ld, L2}) :: L1 where {L1<:Layer, Ld<:Layer, L2<:Layer}
    return L.layer1
end

function get_diff_layer(L :: DiffLayer{L1, Ld, L2}) :: Ld where {L1<:Layer, Ld<:Layer, L2<:Layer}
    return L.diff_layer
end

function get_layer2(L :: DiffLayer{L1, Ld, L2}) :: L2 where {L1<:Layer, Ld<:Layer, L2<:Layer}
    return L.layer2
end

function get_layers(N::GeminiNetwork)
    return map(to_diff_layer,enumerate(zip(
        get_layers(N.network1),
        get_layers(N.diff_network),
        get_layers(N.network2)
        )))
end
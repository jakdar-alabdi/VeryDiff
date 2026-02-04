mutable struct Zonotope
    G::Matrix{Float64}
    c::Vector{Float64}
    influence::Union{Matrix{Float64},Nothing}
end

mutable struct SplitNode
    network :: Int64
    layer :: Int64
    neuron :: Int64
    direction :: Int64
end

mutable struct SplitConstraint
    node :: SplitNode
    g :: Vector{Float64}
    c :: Float64
end

mutable struct Branch
    split_nodes :: Vector{SplitNode}
    undetermined :: BitMatrix
    function Branch(split_nodes=SplitNode[], undetermined=trues(1, 2))
        new(split_nodes, undetermined)
    end
end

struct VerificationTask
    middle :: Vector{Float64}
    distance :: Vector{Float64}
    distance_indices :: Vector{Int}
    ∂Z::Zonotope
    verification_status
    distance_bound :: Float64
    branch :: Branch
end

# Z₂ = Z₁ - ∂Z

mutable struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
    num_approx₁ :: Int
    num_approx₂ :: Int
    ∂num_approx :: Int
end

mutable struct PropState
    task :: VerificationTask
    contract_inter :: Bool
    isempty_intersection :: Bool
    first_improvement :: Bool
    split_constraints :: Vector{SplitConstraint}
    instable_nodes :: Tuple{Vector{BitVector}, Vector{BitVector}}
    intermediate_zonos :: Tuple{Vector{Zonotope}, Vector{Zonotope}}
    relative_impactes :: Tuple{Vector{Vector{Matrix{Float64}}}, Vector{Vector{Matrix{Float64}}}}
    input_relative_impactes :: Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}}
    function PropState(task::VerificationTask, contract=false)
        return new(task, contract, false, false, SplitConstraint[], (BitVector[], BitVector[]), (Zonotope[], Zonotope[]), (Vector{Matrix{Float64}}[], Vector{Matrix{Float64}}[]), (Matrix{Float64}[], Matrix{Float64}[]))
    end
end

struct PropConfig

end

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
                push!(diff_layers, Dense(l1.W .- l2.W, l1.b .- l2.b))
                # println("Distance: ", sum(abs,diff_layers[end].W))
            elseif typeof(l1) == ReLU
                push!(diff_layers, ReLU())
            else
                error("Unsupported layer type")
            end
        end
        return new(network1, network2, Network(diff_layers))
    end
end
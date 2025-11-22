import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function (N::Network)(Z :: Zonotope, P :: PropState, id :: Int64, split_nodes :: Vector{SplitNode})
    return foldl((Z, (l, L)) -> L(Z, P, l, id, split_nodes), enumerate(N.layers), init=Z)
end

function (L::Dense)(Z :: Zonotope, P :: PropState, l :: Int64, id :: Int64, split_nodes :: Vector{SplitNode})
    return @timeit to "Zonotope_DenseProp" begin
    G = L.W * Z.G
    c = L.W * Z.c .+ L.b
    return Zonotope(G,c, Z.influence)
    end
end

function get_slope(l,u, alpha)
    if u <= 0
        return 0.0
    elseif l >= 0
        return 1.0
    else
        return alpha
    end
end

function (L::ReLU)(Z::Zonotope, P::PropState, l::Int64, id::Int64, split_nodes::Vector{SplitNode}; bounds = nothing, split_candidates = SplitCandidate[])
    return @timeit to "Zonotope_ReLUProp" begin
    @timeit to "Bounds" begin
    row_count = size(Z.G,1)
    if isnothing(bounds)
        bounds = zono_bounds(Z)
    end
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]
    end

    # Get split nodes corresponding to this layer
    layer_split_nodes = filter(node -> (node.network, node.layer) == (id, l), split_nodes)

    # For each node in this layer compute the splitting direction
    # 0 no branching, 1 branch to >= 0, -1 branch to <= 0
    nodes_direction = zeros(Int64, size(lower))
    for node in layer_split_nodes
        nodes_direction[node.neuron] = node.direction

        # Update the generator of split node before propagation
        generator = SplitGenerator(Z.G[node.neuron, :], Z.c[node.neuron])
        current_generator = get!(P.split_generators, to_dict_key(node), generator)
        if (sum(abs.(current_generator.g)) > sum(abs.(generator.g)))
            P.split_generators[to_dict_key(node)] = generator
        end
    end

    @timeit to "Vectors" begin
    α = clamp.(upper ./ (upper .- lower), 0.0, 1.0)
    # Use is_onesided to compute 
    λ = ifelse.(upper .<= 0.0 .|| isone.(-nodes_direction), 0.0, ifelse.(lower .>= 0.0 .|| isone.(nodes_direction), 1.0, α))

    crossing = lower .< 0.0 .&& upper .> 0.0 .&& iszero.(nodes_direction)
    
    γ = 0.5 .* max.(-λ .* lower, 0.0, ((-).(1.0, λ)) .* upper)  # Computed offset (-λl/2)

    ĉ = λ .* Z.c .+ crossing .* γ
    end
    
    @timeit to "Influence Matrix" begin
    if NEW_HEURISTIC
        # TODO(steuber): Can we avoid this reallocation?
        @timeit to "Allocation" begin
        #println(size(Z.influence,1), size(Z.influence,2)+count(crossing))
        influence_new = zeros(Float64, size(Z.influence,1), size(Z.influence,2)+count(crossing))
        end
        @timeit to "Set Matrix" begin
        influence_new[:,1:size(Z.influence,2)] .= Z.influence
        end
        # print("Hello")
        # print(size(influence_new))
        # print(size(Z.influence * Z.G[crossing,:]'))
        @timeit to "Multiply" begin
        influence_new[:,(size(Z.influence,2)+1):end] .=  abs.(Z.influence) * abs.(@view Z.G[crossing,:])'
        end
        # foreach(normalize!, eachcol(@view influence_new[:,(size(Z.influence,2)+1):end]))
    else
        influence_new = Z.influence
    end
    end

    @timeit to "Allocation" begin
    Ĝ = zeros(Float64,row_count, size(Z.G,2)+count(crossing))
    end
    #zeros(row_count, size(Z.G,2)+count(crossing))
    #Z.G .*= λ
    @timeit to "Set Matrix" begin
    Ĝ[:,1:size(Z.G,2)] .= Z.G
    Ĝ[crossing,size(Z.G,2)+1:end] .=  (@view I(row_count)[crossing, crossing])
    end
    @timeit to "Column Multiply" begin
    Ĝ[:,1:size(Z.G,2)] .*= λ
    Ĝ[:,size(Z.G,2)+1:end] .*= abs.(γ)
    end

    # Select a split candidate naively (by only considering each node's generator and their layer) for next branching
    if any(crossing)
        neuron = argmax(i -> sum(abs.(Z.G[i, :])), (1:size(crossing, 1))[crossing])
        split_candidate = SplitCandidate(SplitNode(id, l, neuron, 0), sum(abs.(Z.G[neuron, :])))
        if isempty(split_candidates)
            push!(split_candidates, split_candidate)
        elseif split_candidates[1].err < split_candidate.err
            split_candidates[1] = split_candidate
        end
    end

    return Zonotope(Ĝ, ĉ, influence_new)
    end
end

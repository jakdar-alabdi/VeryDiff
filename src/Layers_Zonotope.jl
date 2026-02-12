import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function (N::Network)(Z :: Zonotope, P :: PropState, network :: Int64)
    return foldl((Z, (layer, L)) -> L(Z, P, network, layer), enumerate(N.layers), init=Z)
end

function (L::Dense)(Z :: Zonotope, P :: PropState, network :: Int64, layer :: Int64)
    if P.is_unsatisfiable
        return Z
    end

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

function (L::ReLU)(Z::Zonotope, P::PropState, network::Int64, layer::Int64; bounds = nothing)
    if P.is_unsatisfiable
        return Z
    end

    return @timeit to "Zonotope_ReLUProp" begin
    @timeit to "Bounds" begin
        row_count = size(Z.G,1)
        if isnothing(bounds)
            if USE_NEURON_SPLITTING[]
                # Get split nodes corresponding to this network and this layer
                indices_mask = map(node -> node.network == network && node.layer == layer, P.task.branch.split_nodes)
                layer_split_nodes = @view P.task.branch.split_nodes[indices_mask]

                if !isempty(layer_split_nodes)
                    N̂ = size(Z.G, 2)

                    if P.inter_contract
                        @timeit to "Inter-Contract Zono" begin
                            layer_constraints = SplitConstraint[]
                            @timeit to "Collect Constraints" begin
                                for node in layer_split_nodes
                                    g = Z.G[node.neuron, :]
                                    c = Z.c[node.neuron]
                                    push!(layer_constraints, SplitConstraint(node, g, c))
                                end
                            end
                            
                            sort_constraints!(layer_constraints, zeros(N̂))
                            input_bounds = [-ones(N̂) ones(N̂)]
                            
                            for (;node, g, c) in layer_constraints
                                @timeit to "Contract Zono" begin
                                    input_bounds = contract_zono(input_bounds, g, c, node.direction)
                                    
                                    P.isempty_intersection |= isnothing(input_bounds)
                                    if P.isempty_intersection
                                        P.instable_nodes = (BitVector[], BitVector[])
                                        P.intermediate_zonos = (Zonotope[], Zonotope[])
                                        return Z
                                    end
                                end
                            end

                            if !all(isone.(abs.(input_bounds)))
                                @timeit to "Transform Zono" begin
                                    transform_offset_zono!(input_bounds, Z)
                                    transform_verification_task!(P.task, input_bounds)
                                end
                            end
                        end
                    else
                        @timeit to "Collect Constraints" begin
                            for node in layer_split_nodes
                                g = zonos[node.network].G[node.neuron, :]
                                c = zonos[node.network].c[node.neuron]
                                push!(P.split_constraints, SplitConstraint(node, g, c))
                            end
                        end
                    end
                end
            end

            bounds = zono_bounds(Z)

            if USE_NEURON_SPLITTING[]
                for node in layer_split_nodes
                    bounds[node.neuron, 1] *= node.direction == -1
                    bounds[node.neuron, 2] *= node.direction == 1
                end
            end
        end
        lower = @view bounds[:,1]
        upper = @view bounds[:,2]
    end

    @timeit to "Vectors" begin
    α = clamp.(upper ./ (upper .- lower), 0.0, 1.0)
    # Use is_onesided to compute 
    λ = ifelse.(upper .<= 0.0, 0.0, ifelse.(lower .>= 0.0, 1.0, α))

    crossing = lower .< 0.0 .&& upper .> 0.0

    
    γ = 0.5 .* max.(-λ .* lower, 0.0, ((-).(1.0, λ)) .* upper)  # Computed offset (-λl/2)
    
    ĉ = λ .* Z.c .+ crossing .* γ
    end

    if USE_NEURON_SPLITTING[]
        if NEURON_SPLITTING_APPROACH[] == VerticalSplitting
            indices_mask = map(node -> (node.network, node.layer, node.direction) == (network, layer, -1), P.task.branch.split_nodes)
            layer_split_nodes = @view P.task.branch.split_nodes[indices_mask]

            # relaxtion = ifelse(VS_RELAXATION == VS_Relaxtion1, relaxtion1, relaxtion2)

            for (;neuron, bounds) in layer_split_nodes
                if crossing[neuron]
                    @timeit to "Relax Upper Part" begin
                        l̅, s₁ = bounds[1, 1], bounds[1, 2]
                        s₂, u̅ = bounds[2, 1], bounds[2, 2]
                        
                        λ[neuron] = u̅ / (u̅ - l̅)
                        γ[neuron] = 0.5 * λ[neuron] * (s₁ - l̅)
                        ĉ[neuron] = λ[neuron] * (Z.c[neuron] - s₁) - γ[neuron]
                        
                        # λ[neuron] = s₂ / (s₂ - s₁)
                        # γ[neuron] = 0.5 * λ[neuron] * (l̅ - s₁)
                        # ĉ[neuron] = λ[neuron] * (Z.c[neuron] - s₁) - γ[neuron]
                    end
                end
            end
        end
        push!(P.instable_nodes[network], crossing)
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
    return Zonotope(Ĝ, ĉ, influence_new)
    end
end

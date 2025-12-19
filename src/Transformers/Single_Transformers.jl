import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function (N::Network)(Z :: Zonotope, P :: PropState)
    return foldl((Z,L) -> L(Z,P),N.layers,init=Z)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: Dense, inputs :: Vector{Zonotope})
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: Dense, Zin :: Zonotope)
    # Zout must have exactly the same ids as Zin
    #@assert all(ZoutRef.generator_ids .== Zin.generator_ids) "Zonotope generator IDs do not match during Dense propagation!"
    for i in 1:length(Zin.Gs)
        mul!(ZoutRef.Gs[i], L.W, Zin.Gs[i])
    end
    mul!(ZoutRef.c, L.W, Zin.c)
    ZoutRef.c .+= L.b
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

function propagate_layer!(ZoutRef :: Zonotope, L :: ReLU, inputs :: Vector{Zonotope}; lower=nothing, upper=nothing)
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin; lower=lower, upper=upper)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: ReLU, Zin :: Zonotope; lower=nothing, upper=nothing)
    @timeit to "Bounds" begin
        if isnothing(lower) || isnothing(upper)
            bounds = zono_bounds(Zin)
            lower = @view bounds[:,1]
            upper = @view bounds[:,2]
        end
    end

    @timeit to "Vectors" begin
        dim = length(lower)
        crossing = @simd_bool_expr dim ((lower < 0.0) & (upper > 0.0))
        α = clamp.(upper./(upper.-lower),0.0,1.0)
        # Use is_onesided to compute 
        λ = ifelse.(crossing, α, ifelse.(lower .>= 0.0, 1.0, 0.0))

        new_gens = count(crossing)
        
        γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

        ZoutRef.c .= λ .* Zin.c .+ crossing.*γ
    end

    indices = intersect_indices(ZoutRef.generator_ids, Zin.generator_ids)
    @timeit to "Influence Matrix" begin
        if NEW_HEURISTIC
            influence_new = ZoutRef.influence
            column_pos = size(influence_new[ZoutRef.owned_generators],2) - new_gens + 1
            # @info "Adding $new_gens new columns at position $column_pos to influence matrix of owned generator ID $(ZoutRef.generator_ids[ZoutRef.owned_generators])"
            # @info "Sizes of influence matrices: $([size(inf) for inf in Zin.influence])"
            @inbounds for (idx, inf) in zip(indices, Zin.influence)
                if idx != ZoutRef.owned_generators
                    influence_new[idx] = inf
                else
                    influence_new[ZoutRef.owned_generators][:, 1:column_pos-1] .= inf
                end
            end
            # @info "Size of owned influence matrix after copy: $(size(influence_new[ZoutRef.owned_generators]))"
            #influence_new[ZoutRef.owned_generators][:,column_pos:end] .= 0.0
            bounds_range = upper[crossing] .- lower[crossing]
            @inbounds for (idx, g) in enumerate(Zin.Gs)
                influence_new[ZoutRef.owned_generators][:,column_pos:end] .= abs.(Zin.influence[idx]) * abs.((@view g[crossing,:]) ./ bounds_range)'
            end
            #influence_new[:,(size(Zin.influence,2)+1):end] .=  abs.(Zin.influence) * abs.(@view Zin.G[crossing,:])'
        else
            influence_new = Zin.influence
        end
    end

    num_new_gens = count(crossing)

    @timeit to "Set Matrix" begin
        updateGeneratorsMul!(ZoutRef.Gs, indices, Zin.Gs, λ, :)
        # for (idx, g) in zip(indices, Zin.Gs)
        #     cols = size(g,2)
        #     ZoutRef.Gs[idx][:, 1:cols] .= λ .* g
        # end
        ZoutRef.Gs[ZoutRef.owned_generators][:,(end-num_new_gens+1):end] .= 0.0
        generator_offset = size(ZoutRef.Gs[ZoutRef.owned_generators],2) - num_new_gens
        A = ZoutRef.Gs[ZoutRef.owned_generators]
        @inbounds for (i, row) in enumerate(findall(crossing))
            A[row, (generator_offset + i)] = abs(γ[row])
        end
    end
end
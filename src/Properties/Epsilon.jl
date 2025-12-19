function get_sample_distance(N1, N2, vector, focus_dim=nothing)
    if !isnothing(focus_dim)
        abs.(N1(vector)[focus_dim]-N2(vector)[focus_dim])
    else
        maximum(abs.(N1(vector)-N2(vector)))
    end
end

function get_epsilon_property(epsilon;focus_dim=nothing)
    @info "Creating epsilon property with ε = $epsilon, focus_dim = $focus_dim"
    return (N1, N2, Zin, Zout, verification_status) -> begin
        #TODO: Use verification status to ignore proven epsilons
        @assert length(Zin.Z₁.generator_ids) == 0 || length(Zout.∂Z.generator_ids) == 0 || Zout.∂Z.generator_ids[1] == Zin.Z₁.generator_ids[1] "Input generator block with ID $(Zin.Z₁.generator_ids[1]) not found in output differential Zonotope!"
        out_bounds = zono_bounds(Zout.∂Z)
        distance_bound, max_dim = if !isnothing(focus_dim)
            maximum(abs.(out_bounds[focus_dim,:])),focus_dim
        else
            maximum(abs.(out_bounds)),argmax(abs.(out_bounds))[1]
        end
        # @debug "Distance Bound: $distance_bound"
        if distance_bound > epsilon
            cex_input = Zin.Z₁.c
            sample_distance = get_sample_distance(N1, N2, cex_input, focus_dim)
            if length(Zout.∂Z.generator_ids) > 0 && length(Zin.Z₁.generator_ids) > 0
                # for i in 1:size(Zout.Z₁.G,1)
                max_vec = zono_get_max_vector(Zout.Z₁,max_dim)[1] # only need first component which should correspond to network inputs
                for c in [-1.0,1.0]
                    cex_input = Zin.Z₁.Gs[1]*(max_vec*c)+Zin.Z₁.c
                    sample_distance = max(
                        sample_distance,
                        get_sample_distance(N1, N2, cex_input, focus_dim)
                    )
                    if sample_distance>epsilon
                        break
                    end
                end
            end
            if sample_distance>epsilon
                return false, (cex_input, (N1(cex_input),N2(cex_input),sample_distance)), nothing, nothing, distance_bound
            end


            return false, nothing, (out_bounds, epsilon, focus_dim), nothing, distance_bound
        else
            return true, nothing, nothing, nothing, distance_bound
        end
    end
end

function get_epsilon_property_naive(epsilon;focus_dim=nothing)
    @info "Creating naive epsilon property with ε = $epsilon, focus_dim = $focus_dim"
    normal_eps = get_epsilon_property(epsilon;focus_dim=focus_dim)
    return (N1, N2, Zin, Zout, verification_status) -> begin
        # Build new output Zonotope with differential Zonotope constructed from Zout.Z₁ - Zout.Z₂
        output_indices = union(Zout.Z₁.generator_ids, Zout.Z₂.generator_ids)
        Gs = Vector{Matrix{Float64}}(undef, length(output_indices))
        for (i, gid) in enumerate(output_indices)
            idx1 = attempt_find_index_position(Zout.Z₁.generator_ids, gid)
            idx2 = attempt_find_index_position(Zout.Z₂.generator_ids, gid)
            if idx1 != -1 && idx2 != -1
                # present in both
                Gs[i] = Zout.Z₁.Gs[idx1] - Zout.Z₂.Gs[idx2]
            elseif idx1 != -1
                # present only in Z1
                Gs[i] = Zout.Z₁.Gs[idx1]
            else
                # present only in Z2
                Gs[i] = -Zout.Z₂.Gs[idx2]
            end
        end
        c = Zout.Z₁.c - Zout.Z₂.c
        gen_ids = output_indices
        # Construct differential Zonotope
        differential_zonotope = Zonotope(Gs, c, nothing, gen_ids, nothing)
        Zout_new = DiffZonotope(Zout.Z₁, Zout.Z₂, differential_zonotope)
        return normal_eps(N1, N2, Zin, Zout_new, verification_status)
    end
end



function epsilon_split_heuristic(Zin,Zout,heuristics_info,verification_task)
    # distance_indices = verification_task.distance_indices

    if NEW_HEURISTIC
        # diff_weights = sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,(abs.(Zout.Z₁.G)*abs.(Zout.Z₁.influence').+abs.(Zout.Z₂.G)*abs.(Zout.Z₂.influence')),dims=1)[1,:]
        diff_weights = zeros(size(Zin.Z₁.Gs[1],2))
        for (g1,inf1) in zip(Zout.Z₁.Gs,Zout.Z₁.influence)
            diff_weights .+= sum(abs, abs.(g1) * abs.(inf1'),dims=1)[1,:]
        end
        for (g2,inf2) in zip(Zout.Z₂.Gs,Zout.Z₂.influence)
            diff_weights .+= sum(abs,abs.(g2) * abs.(inf2'),dims=1)[1,:]
        end
        # # Scale by input generator magnitudes across ALL input generator blocks
        # input_weights = zeros(size(Zin.Z₁.Gs[1],2))
        # for g_in in Zin.Z₁.Gs
        #     input_weights .+= sum(abs, g_in, dims=1)[1,:]
        # end
        # diff_weights .*= input_weights
    else
        throw("Old heuristic no longer implemented")
    end
    #@debug diff_weights
    # @debug length(Zout.Z₁.influence)
    # @debug Zout.Z₁.influence[1][1,:]
    # @debug Zout.Z₁.influence[2][1,:]
    # Output Zout.Z₁.influence[2][1,:] with two digits after decimal point
    #@debug round.(Zout.Z₁.influence[2][1,1:20],digits=3)

    d = argmax(
        # sum(abs,Zin.Z₁.G,dims=1)[1,:].*
        #(∂weights .+ diff_weights)
        diff_weights
    )[1]

    #@debug "Selected: $d"
    
    #return distance_indices[d]
    return d
end
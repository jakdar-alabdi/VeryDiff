

mutable struct PropState
    first :: Bool
    zono_storage :: ZonotopeStorage
    free_generator_id :: Int64
    task_bounds :: TaskBounds
    function PropState(first :: Bool)
        return new(first,
                    ZonotopeStorage(Vector{Zonotope}()),
                    -1,
                    TaskBounds()
                    )
    end
end

function reset_ps!(PS :: PropState)
    PS.first = false
end

function first_pass(PS :: PropState) :: Bool
    return PS.first
end

function get_layer_inputs(idxs :: Vector{Int64}, PS :: PropState) :: Vector{CachedZonotope}
    return @view PS.zono_storage.zonotopes[idxs]
end

function get_zonotope(Z :: CachedZonotope) :: DiffZonotope
    return Z.zonotope
end

function configure_first_usage!(layer_idx :: Int64, Zs :: Vector{CachedZonotope})
    for Z in Zs
        if isnothing(Z.first_usage)
            Z.first_usage = layer_idx
        end
    end
end


function init_zonotope_storage!(PS :: PropState, task :: VerificationTask)
    z1_generators = Vector{Matrix{Float64}}()
    z1_generator_ids = SortedVector{Int64}()
    z2_generators = Vector{Matrix{Float64}}()
    z2_generator_ids = SortedVector{Int64}()
    free_generator_id = 2
    push!(z1_generators, zeros(Float64, size(task.middle,1), length(task.distance_indices)))
    push!(z1_generator_ids, 1)
    z1_center = zeros(Float64, size(task.middle,1))
    push!(z2_generators, zeros(Float64, size(task.middle,1), length(task.distance_indices)))
    push!(z2_generator_ids, 1)
    z2_center = zeros(Float64, size(task.middle,1))
    diff_generators = Vector{Matrix{Float64}}()
    diff_generator_ids = SortedVector{Int64}()
    zd_center = zeros(Float64, size(task.middle,1))
    if !isnothing(task.distance1_secondary)
        push!(z1_generators,zeros(Float64, size(task.middle,1), size(task.distance1_secondary,1)))
        push!(z1_generator_ids, 2)
        push!(diff_generators, zeros(Float64, size(task.middle,1), size(task.distance1_secondary,1)))
        push!(diff_generator_ids, 2)
        free_generator_id += 1
    end
    if !isnothing(task.distance2_secondary)
        push!(z2_generators, zeros(Float64, size(task.middle,1), size(task.distance2_secondary,1)))
        push!(z2_generator_ids, 3)
        push!(diff_generators, zeros(Float64, size(task.middle,1), size(task.distance2_secondary,1)))
        push!(diff_generator_ids, 3)
        free_generator_id += 1
    end
    influence1 = nothing
    influence2 = nothing
    if NEW_HEURISTIC
        influence1 = Vector{Matrix{Float64}}()
        influence2 = Vector{Matrix{Float64}}()
        for g in z1_generators
            push!(influence1, Matrix(1.0I, size(g,2), size(g,2)))
        end
        for g in z2_generators
            push!(influence2, Matrix(1.0I, size(g,2), size(g,2)))
        end
    end
    push!(PS.zono_storage.zonotopes,
        CachedZonotope(
            DiffZonotope(
                Zonotope(z1_generators, z1_center, influence1, z1_generator_ids, nothing),
                Zonotope(z2_generators, z2_center, influence2, z2_generator_ids, nothing),
                Zonotope(diff_generators, zd_center, nothing, diff_generator_ids, nothing)
            ),
            0, # First usage is the input layer itself => Noone is allowed to take over owned generators
        )
    )
    PS.zono_storage.zonotopes[1].zonotope = PS.zono_storage.zonotopes[1].zonotope_proto
    PS.free_generator_id = free_generator_id
end

function prepare_prop_state!(PS :: PropState, task :: VerificationTask)
    if length(PS.zono_storage.zonotopes) == 0
        @assert isone(task.work_share) "Zonotope storage must be initialized by the main task"
        init_zonotope_storage!(PS, task)
    end
    @assert length(PS.zono_storage.zonotopes) > 0 "Zonotope storage should have been initialized"
    Zin = PS.zono_storage.zonotopes[1].zonotope
    # Fill in generators and centers
    generator1_common = Zin.Z₁.Gs[1]
    generator1_common .= 0.0
    generator2_common = Zin.Z₂.Gs[1]
    generator2_common .= 0.0
    #@inbounds 
    for (i, idx) in enumerate(task.distance_indices)
        generator1_common[idx,i] = task.distance[i]
        generator2_common[idx,i] = task.distance[i]
    end
    if !isnothing(task.distance1_secondary)
        generator1_secondary = Zin.Z₁.Gs[2]
        generator1_secondary .= 0.0
        generator1_diff = Zin.∂Z.Gs[1]
        generator1_diff .= 0.0
        #@inbounds
        for (i, dist) in enumerate(task.distance1_secondary)
            generator1_secondary[i,i] = dist
            generator1_diff[i,i] = dist
        end
    end
    if !isnothing(task.distance2_secondary)
        generator2_secondary = Zin.Z₂.Gs[2]
        generator2_secondary .= 0.0
        generator2_diff = Zin.∂Z.Gs[2]
        generator2_diff .= 0.0
        #@inbounds 
        for (i, dist) in enumerate(task.distance2_secondary)
            generator2_secondary[i,i] = dist
            generator2_diff[i,i] = -dist
        end
    end
    Zin.Z₁.c .= task.middle
    if !isnothing(task.distance1_secondary)
        Zin.Z₁.c .+= task.middle1_secondary
    end
    Zin.Z₂.c .= task.middle
    if !isnothing(task.distance2_secondary)
        Zin.Z₂.c .+= task.middle2_secondary
    end
    Zin.∂Z.c .= 0.0
    if !isnothing(task.distance1_secondary)
        Zin.∂Z.c .+= task.middle1_secondary
    end
    if !isnothing(task.distance2_secondary)
        Zin.∂Z.c .-= task.middle2_secondary
    end
    # @debug "Initialized Zonotopes:"
    # @debug "Z₁: $(Zin.Z₁)"
    # @debug "Z₂: $(Zin.Z₂)"
    # @debug "∂Z: $(Zin.∂Z)"
    PS.task_bounds = task.task_bounds
end

function has_layer(PS :: PropState, layer :: DiffLayer) :: Bool
    return length(PS.zono_storage.zonotopes) >= layer.layer_idx
end

function get_layer(PS :: PropState, layer :: DiffLayer) :: CachedZonotope
    @assert has_layer(PS, layer) "Layer $(layer.layer_idx) not initialized in PropState!"
    return PS.zono_storage.zonotopes[layer.layer_idx]
end

function get_free_generator_id!(PS :: PropState) :: Int64
    @assert PS.free_generator_id >= 2 "Free generator ID not initialized"
    id = PS.free_generator_id
    PS.free_generator_id += 1
    return id
end
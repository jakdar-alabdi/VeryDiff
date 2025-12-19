struct TaskBounds
    bounds_cache :: Dict{Int, BoundsCache}
    function TaskBounds()
        return new(Dict{Int, BoundsCache}())
    end
end

struct VerificationTask
    middle :: Vector{Float64}
    distance :: Vector{Float64}
    distance_indices :: Vector{Int}
    distance1_secondary :: Union{Nothing, Vector{Float64}}
    middle1_secondary :: Union{Nothing, Vector{Float64}}
    distance2_secondary :: Union{Nothing, Vector{Float64}}
    middle2_secondary :: Union{Nothing, Vector{Float64}}
    verification_status
    distance_bound :: Float64
    work_share :: Float64
    task_bounds :: TaskBounds
    function VerificationTask(middle :: Vector{Float64},
                              distance :: Vector{Float64},
                              distance_indices :: Vector{Int},
                              distance1_secondary :: Union{Nothing, Vector{Float64}},
                              middle1_secondary :: Union{Nothing, Vector{Float64}},
                              distance2_secondary :: Union{Nothing, Vector{Float64}},
                              middle2_secondary :: Union{Nothing, Vector{Float64}},
                              verification_status,
                              distance_bound :: Float64,
                              work_share :: Float64)
        return new(middle,
                    distance,
                    distance_indices,
                    distance1_secondary,
                    middle1_secondary,
                    distance2_secondary,
                    middle2_secondary,
                    verification_status,
                    distance_bound,
                    work_share,
                    TaskBounds()
                )
    end
    function VerificationTask(middle :: Vector{Float64},
                              distance :: Vector{Float64},
                              distance_indices :: Vector{Int},
                              distance1_secondary :: Union{Nothing, Vector{Float64}},
                              middle1_secondary :: Union{Nothing, Vector{Float64}},
                              distance2_secondary :: Union{Nothing, Vector{Float64}},
                              middle2_secondary :: Union{Nothing, Vector{Float64}},
                              verification_status,
                              distance_bound :: Float64,
                              work_share :: Float64,
                              task_bounds :: TaskBounds)
        return new(middle,
                    distance,
                    distance_indices,
                    distance1_secondary,
                    middle1_secondary,
                    distance2_secondary,
                    middle2_secondary,
                    verification_status,
                    distance_bound,
                    work_share,
                    task_bounds
                )
    end
end
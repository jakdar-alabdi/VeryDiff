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
    bounds_cache :: BoundsCache
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
                   BoundsCache(Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
                )
    end
end
struct TaskBounds
    bounds_cache :: Dict{Int, BoundsCache}
    function TaskBounds()
        return new(Dict{Int, BoundsCache}())
    end
end

mutable struct SplitNode
    network :: Int
    layer :: Int
    neuron :: Int
    direction :: Int
    bounds :: Union{Matrix{Float64}, Nothing}
end

mutable struct Branch
    split_nodes :: Vector{SplitNode}
    undetermined :: BitMatrix
    function Branch(split_nodes=SplitNode[], undetermined=trues(1, 2))
        new(split_nodes, undetermined)
    end
end

struct InputBox
    lowers :: Vector{Vector{Float64}}
    uppers :: Vector{Vector{Float64}}
    generator_ids :: SortedVector{Int}
    function InputBox(generator_ids::SortedVector{Int}, generator_sizes::Vector{Int})
        @assert length(generator_ids) == length(generator_sizes)
        lowers = [fill(-1.0, d) for d in generator_sizes]
        uppers = [fill(1.0, d) for d in generator_sizes]
        new(lowers, uppers, generator_ids)
    end
    function InputBox(DZ::DiffZonotope)
        generator_ids = union(DZ.∂Z.generator_ids, union(DZ.Z₁.generator_ids, DZ.Z₂.generator_ids))
        generator_sizes = zeros(Int, length(generator_ids))
        for (i, id) in enumerate(generator_ids)
            idx = attempt_find_index_position(DZ.∂Z.generator_ids, id)
            if idx != -1
                generator_sizes[i] = size(DZ.∂Z.Gs[idx], 2)
                continue
            end
            idx = attempt_find_index_position(DZ.Z₁.generator_ids, id)
            if idx != -1
                generator_sizes[i] = size(DZ.Z₁.Gs[idx], 2)
                continue
            end
            idx = attempt_find_index_position(DZ.Z₂.generator_ids, id)
            generator_sizes[i] = size(DZ.Z₂.Gs[idx], 2)
        end
        InputBox(generator_ids, generator_sizes)
    end
    function InputBox(box :: InputBox)
        lowers = [zeros(length(l)) for l in box.lowers]
        uppers = [zeros(length(u)) for u in box.uppers]
        for i in 1:length(box.generator_ids)
            lowers[i] .= box.lowers[i]
            uppers[i] .= box.uppers[i]
        end
        new(lowers, uppers, box.generator_ids)
    end
end

mutable struct VerificationResult
    status :: VeryDiff.VerificationStatus
    initial_δ_bound :: Float64
    final_δ_bound :: Float64
    num_propagations :: Int
    num_input_splits :: Int
    num_neuron_splits :: Int
    verification_time :: UInt
    function VerificationResult()
        new(VeryDiff.UNKNOWN, Inf, Inf, 0, 0, 0, 0)
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
    branch :: Branch
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
                              branch :: Branch)
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
                    TaskBounds(),
                    branch
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
                              task_bounds :: TaskBounds,
                              branch :: Branch)
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
                    task_bounds,
                    branch
                )
    end
end
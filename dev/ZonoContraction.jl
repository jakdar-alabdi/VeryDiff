function contract_zono(bounds::Matrix{Float64}, g::Vector{Float64}, c::Float64, d::Int64)
    n = size(bounds, 1)
    # @assert d == 1 || d == -1 "Unspecified direction"
    
    l = @view bounds[:, 1]
    u = @view bounds[:, 2]
    
    # With (g, c, d) we impose a linear constraint on the input space
    # Depending on the given direction d (i.e., d = −1 or d = 1 for inactive or active ReLU-phase) 
    # we have one of the following constraints:
    # For d == −1 (incative phase): gᵀx + c <= 0.0 ⇔ gᵀx <= −c ⇔ -d * gᵀx <= d * c
    # For d == 1 (active phase):    gᵀx + c >= 0.0 ⇔ −gᵀx <= c ⇔ -d * gᵀx <= d * c
    g, c = -d * g, d * c

    # Compute a vector v ∈ [l₁, u₁] × ... × [lₙ, uₙ] that minimizes the dot prodoct gᵀv
    v = ifelse.(g .>= 0.0, l, u)

    # If gᵀv > c then we have an empty intersection
    s = g'v
    if s > c
        return nothing
    end

    # For each input dimension i we attempt to increase lᵢ and decrease uᵢ
    for i in 1:n
        if g[i] != 0.0
            # x = (1 / g[i]) * (c - g[1:i-1]'v[1:i-1] - g[i+1:end]'v[i+1:end])
            x = (1 / g[i]) * (c - (s - g[i] * v[i])) # ⇔ x = (1 / g[i]) (c - g[1:i-1]ᵀv[1:i-1] - g[i+1:]ᵀv[i+1:])
            if g[i] > 0
                u[i] = min(u[i], x)
            else
                l[i] = max(l[i], x)
            end
        end
    end

    return bounds
end

function transform_offset_zono!(bounds::Matrix{Float64}, Z::Zonotope; bounds_range=nothing)
    if isnothing(bounds_range)
        bounds_range = 1:size(Z.G, 2)
    end
    lower = @view bounds[bounds_range, 1]
    upper = @view bounds[bounds_range, 2]

    α = (upper - lower) ./ 2
    β = (upper + lower) ./ 2

    Z.c .+= Z.G * β
    Z.G .*= α'

    return Z
end

function transform_offset_zono(bounds::Matrix{Float64}, Z::Zonotope; bounds_range=nothing)
    return transform_offset_zono!(bounds, Zonotope(deepcopy(Z.G), deepcopy(Z.c), Z.influence); bounds_range=bounds_range)
end

function transform_offset_diff_zono!(bounds::Matrix{Float64}, Z::DiffZonotope)
    input_dim = size(Z.Z₁, 2) - Z.num_approx₁
    
    N̂ = size(Z.∂Z.G, 2)
    range₁, range₂ = falses(N̂), falses(N̂)
    range₁[1:input_dim] .= true
    range₂[1:input_dim] .= true
    range₁[input_dim + 1 : input_dim + Z.num_approx₁] .= true
    range₂[input_dim + Z.num_approx₁ + 1 : input_dim + Z.num_approx₁ + Z.num_approx₂] .= true

    Z.Z₁ = transform_offset_zono!(bounds, Z.Z₁; bounds_range=range₁)
    Z.Z₂ = transform_offset_zono!(bounds, Z.Z₂; bounds_range=range₂)
    Z.∂Z = transform_offset_zono!(bounds, Z.∂Z)
    return Z
end

function transform_verification_task!(task::VerificationTask, bounds::Matrix{Float64})
    input_dim = size(task.distance, 1)
    lower = @view bounds[1:input_dim, 1]
    upper = @view bounds[1:input_dim, 2]

    α = (upper - lower) ./ 2
    β = (upper + lower) ./ 2

    task.middle[task.distance_indices] .+= task.distance .* β
    task.distance .*= α

    return task
end

function contract_to_verification_task!(input_bounds::Matrix{Float64}, g::Vector{Float64}, c::Float64, direction::Int64, task::VerificationTask)
    input_bounds = contract_zono(input_bounds, g, c, direction)
    if !isnothing(input_bounds)
        if !all(isone.(abs.(input_bounds)))
            return transform_verification_task!(task, input_bounds)
        end
        return task
    end
    return nothing
end

function offset_zono_bounds(input_bounds::Matrix{Float64}, Z::Zonotope)
    lower = @view input_bounds[1:size(Z.G, 2), 1]
    upper = @view input_bounds[1:size(Z.G, 2), 2]
    bounds = mapreduce(g -> sum(ifelse.(g .>= 0, g .* [lower upper], g .* [upper lower]), dims=1), vcat, eachrow(Z.G))
    bounds .+= Z.c
    return bounds
end

function geometric_distance(x̂::Vector{Float64}, g::Vector{Float64}, c::Float64)
    return abs(g'x̂ + c) / sqrt(g'g)
end

function sort_constraints!(constraints::Vector{SplitConstraint}, x̂::Vector{Float64})
    sort!(constraints, by=constraint -> geometric_distance(x̂, constraint.g, constraint.c))
    return constraints
end

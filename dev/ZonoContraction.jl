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

function transform_offset_zono(bounds::Matrix{Float64}, Z::Zonotope)
    lower = @view bounds[1:size(Z.G, 2), 1]
    upper = @view bounds[1:size(Z.G, 2), 2]

    α = (upper - lower) ./ 2
    β = (upper + lower) ./ 2

    # Z.G .*= α' # Z.G * diag(α)
    # Z.c .+= Z.G * β

    # return Z

    G = Z.G .* α' # Z.G * diag(α)
    c = Z.c + Z.G * β

    return Zonotope(G, c, Z.influence)
end

function offset_zono_bounds(input_bounds::Matrix{Float64}, Z::Zonotope)
    lower = @view input_bounds[1:size(Z.G, 2), 1]
    upper = @view input_bounds[1:size(Z.G, 2), 2]
    ub_bounds = mapreduce(g -> sum(ifelse.(g .>= 0, g .* [lower upper], g .* [upper lower]), dims=1), vcat, eachrow(Z.G))
    return ub_bounds .+ Z.c
end

function contract_to_verification_task(input_bounds::Matrix{Float64}, g::Vector{Float64}, c::Float64, direction::Int64, Z::Zonotope, task::VerificationTask)
    
    input_bounds = contract_zono(input_bounds, g, c, direction)
    if !isnothing(input_bounds)

        # bounds = offset_zono_bounds(input_bounds, Z)
        Z = transform_offset_zono(input_bounds, Z)
        bounds = zono_bounds(Z)
        lower = @view bounds[task.distance_indices, 1]
        upper = @view bounds[task.distance_indices, 2]

        mid = (upper .+ lower) ./ 2
        task.distance .= mid .- lower
        task.middle[task.distance_indices] .= mid

        return task
    end

    return nothing
end

function geometric_distance(x̂::Vector{Float64}, g::Vector{Float64}, c::Float64)
    return abs(g'x̂ + c) / sqrt(g'g)
end

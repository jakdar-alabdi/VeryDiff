struct Zonotope
    # Vector of Generator Matrices
    Gs::Vector{Matrix{Float64}}
    # Center of Zonotope
    c::Vector{Float64}
    # Influence Matrix (necessary for input splitting heuristic)
    # TODO: Move this to PropState at some point?
    influence::Union{Vector{Matrix{Float64}},Nothing}
    # ID of generator matrices
    # must be same length as G
    generator_ids :: SortedVector{Int64}
    # Index of generator matrix owned by Zonotope
    # We may only append columns to the owned generator matrix
    # This Zonotope must be the only active Zonotope that modifies this generator matrix
    # If no owned generator matrix has been allocated, this field is nothing
    owned_generators :: Union{Int64, Nothing}
end

# Z₂ = Z₁ - ∂Z

struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
end

mutable struct BoundsCache
    lower :: Vector{Vector{Float64}}
    upper :: Vector{Vector{Float64}}
end

mutable struct CachedZonotope
    zonotope_proto :: DiffZonotope
    zonotope :: Union{Nothing, DiffZonotope}
    first_usage :: Union{Nothing,Int64} # The first usage of this Zonotope has permission to own generators
    function CachedZonotope(zono_proto :: DiffZonotope, first_usage :: Union{Nothing,Int64})
        return new(zono_proto, nothing, first_usage)
    end
end

struct ZonotopeStorage
    zonotopes :: Vector{CachedZonotope}
end

function get_zonotope!(zono :: CachedZonotope, needed_columns :: Vector{Int64})
    @assert all(needed_columns .<= size.(zono.zonotope.Z₁.Gs,2)) "Requested more generator columns than available in Zonotope!"
    # Create view based new Zonotope
    Z₁_gs = Matrix{Float64}[]
    influence = nothing
    for g in zono.zonotope.Z₁.Gs
        push!(Z₁_gs, @view g[:, needed_columns])
    end
    if !isnothing(zono.zonotope.Z₁.influence)
        influence = Matrix{Float64}[]
        for inf in zono.zonotope.Z₁.influence
            push!(influence, @view inf[:, needed_columns])
        end
    end
    Z₁ = Zonotope(Z₁_gs, zono.zonotope.Z₁.c, influence, zono.zonotope.Z₁.generator_ids, zono.zonotope.Z₁.owned_generators)
    Z₂_gs = Matrix{Float64}[]
    influence = nothing
    for g in zono.zonotope.Z₂.Gs
        push!(Z₂_gs, @view g[:, needed_columns])
    end
    if !isnothing(zono.zonotope.Z₂.influence)
        influence = Matrix{Float64}[]
        for inf in zono.zonotope.Z₂.influence
            push!(influence, @view inf[:, needed_columns])
        end
    end
    Z₂ = Zonotope(Z₂_gs, zono.zonotope.Z₂.c, influence, zono.zonotope.Z₂.generator_ids, zono.zonotope.Z₂.owned_generators)
    ∂Z_gs = Matrix{Float64}[]
    influence = nothing
    for g in zono.zonotope.∂Z.Gs
        push!(∂Z_gs, @view g[:, needed_columns])
    end
    if !isnothing(zono.zonotope.∂Z.influence)
        influence = Matrix{Float64}[]
        for inf in zono.zonotope.∂Z.influence
            push!(influence, @view inf[:, needed_columns])
        end
    end
    ∂Z = Zonotope(∂Z_gs, zono.zonotope.∂Z.c, influence, zono.zonotope.∂Z.generator_ids, zono.zonotope.∂Z.owned_generators)
    zono.zonotope = DiffZonotope(Z₁, Z₂, ∂Z)
    return zono.zonotope
end
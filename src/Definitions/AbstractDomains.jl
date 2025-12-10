struct Zonotope
    # Vector of Generator Matrices
    Gs::Vector{<:AbstractMatrix{Float64}}
    # Center of Zonotope
    c::Vector{Float64}
    # Influence Matrix (necessary for input splitting heuristic)
    # TODO: Move this to PropState at some point?
    influence::Union{Vector{<:AbstractMatrix{Float64}},Nothing}
    # ID of generator matrices
    # must be same length as G
    generator_ids :: SortedVector{Int64}
    # Index of generator matrix owned by Zonotope
    # We may only append columns to the owned generator matrix
    # This Zonotope must be the only active Zonotope that modifies this generator matrix
    # If no owned generator matrix has been allocated, this field is nothing
    owned_generators :: Union{Int64, Nothing}
    function Zonotope(
        Gs::Vector{<:AbstractMatrix{Float64}},
        c::Vector{Float64},
        influence::Union{Vector{<:AbstractMatrix{Float64}},Nothing},
        generator_ids :: SortedVector{Int64},
        owned_generators :: Union{Int64, Nothing}
    )
        @assert length(Gs) == length(generator_ids) "Number of generator matrices must match number of generator IDs"
        @assert isnothing(owned_generators) || owned_generators <= length(generator_ids) "Owned generator index out of bounds"
        return new(Gs, c, influence, generator_ids, owned_generators)
    end
end

# Z₂ = Z₁ - ∂Z

struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
end

mutable struct BoundsCache
    initialized :: Bool
    lower₁ :: Union{Vector{Float64}, Nothing}
    upper₁ :: Union{Vector{Float64}, Nothing}
    lower₂ :: Union{Vector{Float64}, Nothing}
    upper₂ :: Union{Vector{Float64}, Nothing}
    ∂lower :: Union{Vector{Float64}, Nothing}
    ∂upper :: Union{Vector{Float64}, Nothing}
    function BoundsCache()
        return new(false, nothing,nothing,nothing,nothing,nothing,nothing)
    end
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

function get_zonotope!(
    zono :: CachedZonotope,
    needed_columns₁ :: Vector{Int64},
    needed_columns₂ :: Vector{Int64},
    needed_columns_∂ :: Vector{Int64}) :: DiffZonotope
    # Check if we already have the needed columns
    #@assert all(needed_columns₁ .<= size.(zono.zonotope_proto.Z₁.Gs,2)) "Requested more generator columns than available in Zonotope 1!"
    #@assert all(needed_columns₂ .<= size.(zono.zonotope_proto.Z₂.Gs,2)) "Requested more generator columns than available in Zonotope 2!"
    #@assert all(needed_columns_∂ .<= size.(zono.zonotope_proto.∂Z.Gs,2)) "Requested more generator columns than available in Differential Zonotope!"
    # Create view based new Zonotope
    for (i, needed_columns) in enumerate(needed_columns₁)
        @assert needed_columns <= size(zono.zonotope.Z₁.Gs[i],2) "Requested $needed_columns columns, but only $(size(zono.zonotope.Z₁.Gs[i],2)) available in generator matrix $i of Z₁!"
        zono.zonotope.Z₁.Gs[i] = @view zono.zonotope_proto.Z₁.Gs[i][:, 1:needed_columns]
        if !isnothing(zono.zonotope.Z₁.influence)
            zono.zonotope.Z₁.influence[i] = @view zono.zonotope_proto.Z₁.influence[i][:, 1:needed_columns]
        end
    end
    for (i, needed_columns) in enumerate(needed_columns₂) "Requested $needed_columns columns, but only $(size(zono.zonotope.Z₂.Gs[i],2)) available in generator matrix $i of Z₂!"
        @assert needed_columns <= size(zono.zonotope.Z₂.Gs[i],2)
        zono.zonotope.Z₂.Gs[i] = @view zono.zonotope_proto.Z₂.Gs[i][:, 1:needed_columns]
        if !isnothing(zono.zonotope.Z₂.influence)
            zono.zonotope.Z₂.influence[i] = @view zono.zonotope_proto.Z₂.influence[i][:, 1:needed_columns]
        end
    end
    for (i, needed_columns) in enumerate(needed_columns_∂)
        @assert needed_columns <= size(zono.zonotope.∂Z.Gs[i],2) "Requested $needed_columns columns, but only $(size(zono.zonotope.∂Z.Gs[i],2)) available in generator matrix $i of ∂Z!"
        zono.zonotope.∂Z.Gs[i] = @view zono.zonotope_proto.∂Z.Gs[i][:, 1:needed_columns]
        if !isnothing(zono.zonotope.∂Z.influence)
            zono.zonotope.∂Z.influence[i] = @view zono.zonotope_proto.∂Z.influence[i][:, 1:needed_columns]
        end
    end
    return zono.zonotope
end
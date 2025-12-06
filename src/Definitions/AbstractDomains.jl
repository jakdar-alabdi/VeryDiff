struct Zonotope
    # Vector of Generator Matrices
    Gs::Vector{AbstractMatrix{Float64}}
    # Center of Zonotope
    c::Vector{Float64}
    # Influence Matrix (necessary for input splitting heuristic)
    # TODO: Move this to PropState at some point?
    influence::Union{Vector{AbstractMatrix{Float64}},Nothing}
    # ID of generator matrices
    # must be same length as G
    generator_ids :: SortedVector{Int64}
    # Index of generator matrix owned by Zonotope
    # We may only append columns to the owned generator matrix
    # This Zonotope must be the only active Zonotope that modifies this generator matrix
    # If no owned generator matrix has been allocated, this field is nothing
    owned_generators :: Union{Int64, Nothing}
    function Zonotope(
        Gs::Vector{AbstractMatrix{Float64}},
        c::Vector{Float64},
        influence::Union{Vector{AbstractMatrix{Float64}},Nothing},
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

function get_zonotope!(
    zono :: CachedZonotope,
    needed_columns₁ :: Vector{Int64},
    needed_columns₂ :: Vector{Int64},
    needed_columns_∂ :: Vector{Int64}) :: DiffZonotope
    # Check if we already have the needed columns
    @assert all(needed_columns₁ .<= size.(zono.zonotope_proto.Z₁.Gs,2)) "Requested more generator columns than available in Zonotope 1!"
    @assert all(needed_columns₂ .<= size.(zono.zonotope_proto.Z₂.Gs,2)) "Requested more generator columns than available in Zonotope 2!"
    @assert all(needed_columns_∂ .<= size.(zono.zonotope_proto.∂Z.Gs,2)) "Requested more generator columns than available in Differential Zonotope!"
    # Create view based new Zonotope
    Z₁_gs = AbstractMatrix{Float64}[]
    influence = nothing
    for (g, needed_columns) in zip(zono.zonotope_proto.Z₁.Gs, needed_columns₁)
        push!(Z₁_gs, @view g[:, 1:needed_columns])
    end
    if !isnothing(zono.zonotope_proto.Z₁.influence)
        influence = AbstractMatrix{Float64}[]
        for (inf, needed_columns) in zip(zono.zonotope_proto.Z₁.influence, needed_columns₁)
            push!(influence, @view inf[:, 1:needed_columns])
        end
    end
    Z₁ = Zonotope(Z₁_gs, zono.zonotope_proto.Z₁.c, influence, zono.zonotope_proto.Z₁.generator_ids, zono.zonotope_proto.Z₁.owned_generators)
    Z₂_gs = AbstractMatrix{Float64}[]
    influence = nothing
    for (g, needed_columns) in zip(zono.zonotope_proto.Z₂.Gs, needed_columns₂)
        push!(Z₂_gs, @view g[:, 1:needed_columns])
    end
    if !isnothing(zono.zonotope_proto.Z₂.influence)
        influence = AbstractMatrix{Float64}[]
        for (inf, needed_columns) in zip(zono.zonotope_proto.Z₂.influence, needed_columns₂)
            push!(influence, @view inf[:, 1:needed_columns])
        end
    end
    Z₂ = Zonotope(Z₂_gs, zono.zonotope_proto.Z₂.c, influence, zono.zonotope_proto.Z₂.generator_ids, zono.zonotope_proto.Z₂.owned_generators)
    ∂Z_gs = AbstractMatrix{Float64}[]
    influence = nothing
    for (g, needed_columns) in zip(zono.zonotope_proto.∂Z.Gs, needed_columns_∂)
        push!(∂Z_gs, @view g[:, 1:needed_columns])
    end
    if !isnothing(zono.zonotope_proto.∂Z.influence)
        influence = AbstractMatrix{Float64}[]
        for (inf, needed_columns) in zip(zono.zonotope_proto.∂Z.influence, needed_columns_∂)
            push!(influence, @view inf[:, 1:needed_columns])
        end
    end
    ∂Z = Zonotope(∂Z_gs, zono.zonotope_proto.∂Z.c, influence, zono.zonotope_proto.∂Z.generator_ids, zono.zonotope_proto.∂Z.owned_generators)
    zono.zonotope = DiffZonotope(Z₁, Z₂, ∂Z)
    return zono.zonotope
end
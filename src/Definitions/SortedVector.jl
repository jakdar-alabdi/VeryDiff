import Base: length, getindex, iterate, eltype, push!, union

struct SortedVector{T}
    data::Vector{T}
end

SortedVector{T}() where {T} = SortedVector{T}(Vector{T}())

@inline Base.length(v::SortedVector) = length(v.data)
@inline Base.getindex(v::SortedVector, i) = v.data[i]
@inline Base.iterate(v::SortedVector, s...) = iterate(v.data, s...)
@inline Base.eltype(::Type{SortedVector{T}}) where T = T
@inline Base.IndexStyle(::Type{<:SortedVector}) = IndexStyle(Vector)

@inline function Base.push!(v::SortedVector{T}, x::T) where T
    if !isempty(v.data) && x < v.data[end]
        throw(ArgumentError("cannot push $x into SortedVector: would break sortedness"))
    end
    push!(v.data, x)
end

function Base.union(v1::SortedVector{T}, v2::SortedVector{T}) where T
    result = SortedVector{T}()
    # Worst case:
    resize!(result.data, length(v1) + length(v2))
    i1 = 1
    i2 = 1
    ir = 1
    @inbounds while i1 <= length(v1) && i2 <= length(v2)
        if v1.data[i1] < v2.data[i2]
            result.data[ir] = v1.data[i1]
            i1 += 1
        elseif v2.data[i2] < v1.data[i1]
            result.data[ir] = v2.data[i2]
            i2 += 1
        else
            result.data[ir] = v1.data[i1]
            i1 += 1
            i2 += 1
        end
        ir += 1
    end
    while i1 <= length(v1)
        result.data[ir] = v1.data[i1]
        i1 += 1
        ir += 1
    end
    while i2 <= length(v2)
        result.data[ir] = v2.data[i2]
        i2 += 1
        ir += 1
    end
    resize!(result.data, ir - 1)
    return result
end

function Base.union(vs::Vararg{SortedVector{T},N}) where {T,N}
    result = SortedVector{T}()
    for v in vs
        result = union(result, v)
    end
    return result
end

function intersect_indices(v1::SortedVector{T}, v2::SortedVector{T}) where T
    @assert length(v1) >= length(v2)
    result = SortedVector{T}()
    resize!(result.data, length(v2))
    i1 = 1
    i2 = 1
    @inbounds while i1 <= length(v1) && i2 <= length(v2)
        if v1.data[i1] == v2.data[i2]
            result.data[i2] = i1
            i1 += 1
            i2 += 1
        elseif v1.data[i1] < v2.data[i2]
            i1 += 1
        else
            throw("v2 contains element not in v1: $(v2.data[i2])")
        end
    end
    if i2 < length(v2)
        throw("v2 contains element not in v1: $(v2.data[i2])")
    end
    return result
end

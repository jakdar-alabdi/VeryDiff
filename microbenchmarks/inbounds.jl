using Random

using BenchmarkTools

struct EachTrueIter{B<:AbstractVector{Bool}}
    b::B
end

Base.iterate(it::EachTrueIter) = _nexttrue(it.b, 1)
Base.iterate(it::EachTrueIter, state) = _nexttrue(it.b, state + 1)

@inline function _nexttrue(b::AbstractVector{Bool}, i::Int)
    j = findnext(b, i)
    j === nothing && return nothing
    return (j, j)
end

Base.length(it::EachTrueIter) = count(it.b)   # optional
Base.eltype(::Type{EachTrueIter}) = Int

function eachtrue(b::AbstractVector{Bool}) :: EachTrueIter
    return EachTrueIter(b)
end



struct ZContainer
    Gs::Vector{Matrix{Float64}}
    owned_generators::Int
end

function update_generators_original!(ZoutRef, crossing, γ, num_new_gens)
    for (i, row) in enumerate(findall(crossing))
        ZoutRef.Gs[ZoutRef.owned_generators][row, (end - num_new_gens + i)] = abs(γ[i])
    end

    return ZoutRef
end

function update_generators_moved!(ZoutRef, crossing, γ, num_new_gens)
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    M = size(A, 2)  # total number of generators
    # end - num_new_gens is (M - num_new_gens)
    basecol = M - num_new_gens

    for (i, row) in enumerate(findall(crossing))
        A[row, basecol + i] = abs(γ[i])
    end

    return ZoutRef
end

function update_generators_improved!(ZoutRef, crossing, γ, num_new_gens)
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    pos = size(A,2) - num_new_gens + 1
    @inbounds for i in eachindex(crossing)
        if crossing[i]
            A[i, pos] = abs(γ[i])
            pos += 1
        end
    end
    return ZoutRef
end

function update_generators_custom_iterator!(ZoutRef, crossing, γ, num_new_gens)
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    M = size(A, 2)  # total number of generators
    # end - num_new_gens is (M - num_new_gens)
    basecol = M - num_new_gens

    @inbounds for (i, row) in enumerate(eachtrue(crossing))
        A[row, basecol + i] = abs(γ[i])
    end

    return ZoutRef
end

function setup_data(N::Int, num_gens::Int, k::Int)
    # The matrix that will be written into
    A = zeros(N, num_gens)

    # Container structure
    ZoutRef = ZContainer([A], 1)

    # Boolean selector, with exactly k entries set to true
    crossing = falses(N)
    crossing[randperm(N)[1:k]] .= true

    # γ must have length >= k for the semantics to match your original code
    γ = randn(k)

    num_new_gens = k

    return ZoutRef, crossing, γ, num_new_gens
end

@info "k = 800"
ZoutRef, crossing, γ, num_new_gens = setup_data(1_000, 3_000, 800)

@info "Benchmarking original implementation"
@btime update_generators_original!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking mini loop implementation"
@btime update_generators_moved!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking improved implementation"
@btime update_generators_improved!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking custom iterator implementation"
@btime update_generators_custom_iterator!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "k = 50"
ZoutRef, crossing, γ, num_new_gens = setup_data(1_000, 3_000, 50)

@info "Benchmarking original implementation"
@btime update_generators_original!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking mini loop implementation"
@btime update_generators_moved!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking improved implementation"
@btime update_generators_improved!($ZoutRef, $crossing, $γ, $num_new_gens)

@info "Benchmarking custom iterator implementation"
@btime update_generators_custom_iterator!($ZoutRef, $crossing, $γ, $num_new_gens)

# [ Info: k = 800
# [ Info: Benchmarking original implementation
#   1.850 μs (1 allocation: 6.38 KiB)
# [ Info: Benchmarking mini loop implementation
#   1.285 μs (1 allocation: 6.38 KiB)
# [ Info: Benchmarking improved implementation
#   836.082 ns (0 allocations: 0 bytes)
# [ Info: Benchmarking custom iterator implementation
#   3.033 μs (0 allocations: 0 bytes)
# [ Info: k = 50
# [ Info: Benchmarking original implementation
#   132.438 ns (1 allocation: 496 bytes)
# [ Info: Benchmarking mini loop implementation
#   106.686 ns (1 allocation: 496 bytes)
# [ Info: Benchmarking improved implementation
#   317.021 ns (0 allocations: 0 bytes)
# [ Info: Benchmarking custom iterator implementation
#   196.631 ns (0 allocations: 0 bytes)
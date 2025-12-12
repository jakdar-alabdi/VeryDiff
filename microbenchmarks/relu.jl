# Compare matmul vs. row-wise propagation for ReLU layers

using BenchmarkTools
using Random
using VeryDiff

using VeryDiff.Definitions: updateGenerators!, updateGeneratorsMul!, updateGeneratorsAdd!, updateGeneratorsAddMul!, updateGeneratorsSub!, updateGeneratorsSubMul!

VeryDiff.NEW_HEURISTIC = false

import VeryDiff.Transformers: propagate_layer_matmul!, propagate_layer_legacy!, init_default_zono
const DiffLayer = VeryDiff.DiffLayer
const Zonotope = VeryDiff.Zonotope
const DiffZonotope = VeryDiff.DiffZonotope
const CachedZonotope = VeryDiff.CachedZonotope
const BoundsCache = VeryDiff.BoundsCache
const SortedVector = VeryDiff.SortedVector
const ReLU = VeryDiff.ReLU

# Configuration
const DIM = 784              # Number of rows in the zonotope matrices
const NUM_GENS = 1000          # Columns per generator matrix before ReLU
const EXTRA_CAPACITY = DIM   # Extra columns reserved for newly created generators

Random.seed!(42)

function sorted_ids(ids::AbstractVector{<:Integer})
	sv = SortedVector{Int64}()
	for id in sort(ids)
		push!(sv, Int64(id))
	end
	return sv
end

function make_input_zonotope(dim::Int, num_gens::Int)
	gid1 = sorted_ids([101])
	gid2 = sorted_ids([202])
	gid∂ = sorted_ids([101, 202, 303])

	Z₁ = Zonotope([randn(dim, num_gens)], randn(dim), nothing, gid1, 1)
	Z₂ = Zonotope([randn(dim, num_gens)], randn(dim), nothing, gid2, 1)
	∂Z = Zonotope([
		randn(dim, num_gens),
		randn(dim, num_gens),
		randn(dim, num_gens)
	], randn(dim), nothing, gid∂, 3)

	return DiffZonotope(Z₁, Z₂, ∂Z)
end


function make_cache(proto::DiffZonotope; extra_cols::Int=EXTRA_CAPACITY)
	dim = length(proto.Z₁.c)
	Z₁_proto = Zonotope([
		zeros(dim, size(proto.Z₁.Gs[1], 2) + extra_cols)
	], zeros(dim), nothing, proto.Z₁.generator_ids, proto.Z₁.owned_generators)
	Z₂_proto = Zonotope([
		zeros(dim, size(proto.Z₂.Gs[1], 2) + extra_cols)
	], zeros(dim), nothing, proto.Z₂.generator_ids, proto.Z₂.owned_generators)
	∂Z_proto = Zonotope([
		zeros(dim, size(g, 2) + extra_cols) for g in proto.∂Z.Gs
	], zeros(dim), nothing, proto.∂Z.generator_ids, proto.∂Z.owned_generators)

	cache = CachedZonotope(DiffZonotope(Z₁_proto, Z₂_proto, ∂Z_proto), nothing)
	init_default_zono(cache)
	return cache
end

const BASE_INPUT_PROTO = make_input_zonotope(DIM, NUM_GENS)
const BASE_CACHE_PROTO = make_cache(BASE_INPUT_PROTO)

function setup_case()
	# Fresh deep copies so both matmul and row-wise see identical generators
	zin = deepcopy(BASE_INPUT_PROTO)
	cache = deepcopy(BASE_CACHE_PROTO)
	init_default_zono(cache) # reset zonotope views after deepcopy
	bounds_cache = BoundsCache()
	return (cache=cache, zin=zin, bounds=bounds_cache)
end

function bench_relu()
	@info ("Benchmarking ReLU propagation (dim=$(DIM), gens=$(NUM_GENS))")
    relu_layer = DiffLayer(1, ReLU(), ReLU(), ReLU())

	@info ("\nmatmul implementation:")
	@btime propagate_layer_matmul!(case.cache, $relu_layer, [case.zin]; bounds_cache=case.bounds) setup=(case = setup_case()) evals=1

	@info ("\nlegacy implementation:")
	@btime propagate_layer_legacy!(case.cache, $relu_layer, [case.zin]; bounds_cache=case.bounds) setup=(case = setup_case()) evals=1

	nothing
end

function exec_relus()
    relu_layer = DiffLayer(1, ReLU(), ReLU(), ReLU())
    case = setup_case()
    propagate_layer_matmul!(case.cache, relu_layer, [case.zin]; bounds_cache=case.bounds)
    case = setup_case()
    propagate_layer_legacy!(case.cache, relu_layer, [case.zin]; bounds_cache=case.bounds)
end

function compareUpdateGenerators()
	@info "\nComparing updateGenerators! implementations:"
	Gout = AbstractMatrix{Float64}[randn(DIM, NUM_GENS) for _ in 1:4]
	Gin = AbstractMatrix{Float64}[randn(DIM, NUM_GENS) for _ in 1:3]
	indices = SortedVector{Int64}([1,2,4])
	selector = convert(BitVector,rand(Bool, DIM))
	gamma = randn(sum(selector))

	@info "Assign generators without multiplication:"
	@btime updateGenerators!($Gout, $indices, $Gin, $selector) evals=10

	@info "Assign generators with multiplication:"
	@btime updateGeneratorsMul!($Gout, $indices, $Gin, $gamma, $selector) evals=10

	@info "Add to generators without multiplication:"
	@btime updateGeneratorsAdd!($Gout, $indices, $Gin, $selector) evals=10

	@info "Add to generators with multiplication:"
	@btime updateGeneratorsAddMul!($Gout, $indices, $Gin, $gamma, $selector) evals=10

	@info "Subtract from generators without multiplication:"
	@btime updateGeneratorsSub!($Gout, $indices, $Gin, $selector) evals=10

	@info "Subtract from generators with multiplication:"
	@btime updateGeneratorsSubMul!($Gout, $indices, $Gin, $gamma, $selector) evals=10
end


if abspath(PROGRAM_FILE) == @__FILE__
    bench_relu()

	# Compare updateGenerators!, updateGeneratorsMul!, etc.
	compareUpdateGenerators()
end

# Old Transformers:
# [ Info: Benchmarking ReLU propagation (dim=784, gens=1000)
# ┌ Info: 
# └ matmul implementation:
#   11.738 ms (6451 allocations: 451.73 KiB)
# ┌ Info: 
# └ row-wise implementation:
#   10.260 ms (6365 allocations: 504.87 KiB)

# New Transformers:
# [ Info: Benchmarking ReLU propagation (dim=784, gens=1000)
# ┌ Info: 
# └ matmul implementation:
#   7.886 ms (6456 allocations: 437.84 KiB)
# ┌ Info: 
# └ legacy implementation:
#   11.223 ms (6451 allocations: 451.73 KiB)

# Generator Comparison:
# ┌ Info: 
# └ Comparing updateGenerators! implementations:
# [ Info: Assign generators without multiplication:
#   2.211 ms (27 allocations: 20.11 KiB)
# [ Info: Assign generators with multiplication:
#   2.283 ms (24 allocations: 19.92 KiB)
# [ Info: Add to generators without multiplication:
#   4.479 ms (30 allocations: 8.99 MiB)
# [ Info: Add to generators with multiplication:
#   4.609 ms (33 allocations: 8.99 MiB)
# [ Info: Subtract from generators without multiplication:
#   4.502 ms (30 allocations: 8.99 MiB)
# [ Info: Subtract from generators with multiplication:
#   4.575 ms (33 allocations: 8.99 MiB)
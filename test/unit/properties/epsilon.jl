using Test

# VeryDiff is already imported in test/runtests.jl
# We use exported types and functions from VeryDiff
using VeryDiff
using VeryDiff: Zonotope, DiffZonotope, zono_bounds
using VeryDiff.Definitions: SortedVector

# Helpers to build Zonotopes consistent with Definitions/AbstractDomains.jl
function make_zonotope(Gs::Vector{<:AbstractMatrix{Float64}}, c::Vector{Float64}; ids::Vector{Int}=collect(1:length(Gs)))
	@assert issorted(ids)
	gen_ids = SortedVector{Int64}()
	for id in ids
		push!(gen_ids, Int64(id))
	end
	return Zonotope(Gs, c, nothing, gen_ids, nothing)
end

make_empty_zonotope(dim::Int) = make_zonotope(Vector{Matrix{Float64}}(), zeros(dim))

function make_diff_zonotope(Z1::Zonotope, Z2::Zonotope, Zd::Zonotope)
	return DiffZonotope(Z1, Z2, Zd)
end

@testset "get_epsilon_property: semantics without focus_dim (provable)" begin
	# Two mock networks (identical) ⇒ no difference anywhere
	N1 = x -> x
	N2 = x -> x

	dim = 2
	# Zin: input diff zonotope with one common generator block (ID 1)
	G1_in = [1.0 0.0; 0.0 1.0]  # 2x2, arbitrary; only structure matters here
	Zin_Z1 = make_zonotope([G1_in], zeros(dim); ids=[1])
	Zin_Z2 = make_zonotope([G1_in], zeros(dim); ids=[1])
	Zin_d  = make_empty_zonotope(dim)
	Zin    = make_diff_zonotope(Zin_Z1, Zin_Z2, Zin_d)

	# Zout: differential zonotope has bound strictly below epsilon in all dims
	# Choose empty generators ⇒ bounds == [c c] == 0
	Zout_Z1 = make_zonotope([G1_in, zeros(dim, 1)], zeros(dim); ids=[1, 2]) # more blocks than Zin; block 1 has same column count
	Zout_Z2 = make_zonotope([G1_in, zeros(dim, 2)], zeros(dim); ids=[1, 3])
	Zout_d  = make_empty_zonotope(dim)
	Zout    = make_diff_zonotope(Zout_Z1, Zout_Z2, Zout_d)

	epsilon = 0.1
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)

	@test ok == true
	@test dist_bound <= epsilon
	@test all(zono_bounds(Zout.∂Z) .== 0.0)
end

@testset "get_epsilon_property: semantics with focus_dim (provable on one dim)" begin
	# Networks differ in dimension 2, but we only focus on dimension 1
	N1 = x -> [x[1], x[2]]
	N2 = x -> [x[1], x[2] + 10.0]

	dim = 2
	# Zin with one generator block ID 1 having same column count as Zout's block 1
	G1_in = reshape([1.0, 0.0], 2, 1)  # 2x1
	Zin_Z1 = make_zonotope([G1_in], zeros(dim); ids=[1])
	Zin_Z2 = make_zonotope([G1_in], zeros(dim); ids=[1])
	Zin_d  = make_empty_zonotope(dim)
	Zin    = make_diff_zonotope(Zin_Z1, Zin_Z2, Zin_d)

	# Zout with more generator blocks than Zin; ensure block ID 1 sizes match
	# Differential bounds: small on dim 1 (< epsilon), large on dim 2 (> epsilon)
	Gd = reshape([0.05, 10.0], 2, 1)  # sums per row ⇒ bounds [±0.05, ±10.0]
	Zout_d  = make_zonotope([Gd], zeros(dim); ids=[1])
	Zout_Z1 = make_zonotope([G1_in, zeros(dim, 2), zeros(dim, 1)], zeros(dim); ids=[1, 2, 3])
	Zout_Z2 = make_zonotope([G1_in], zeros(dim); ids=[1])
	Zout    = make_diff_zonotope(Zout_Z1, Zout_Z2, Zout_d)

	epsilon = 0.1
	focus_dim = 1
	check = VeryDiff.get_epsilon_property(epsilon; focus_dim)
	ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)

	# Even though another dimension can exceed epsilon, focus_dim bound is <= epsilon ⇒ provable
	@test ok == true
	@test dist_bound <= epsilon
end

@testset "get_epsilon_property_naive: semantics" begin
	# The naive method is marked as deprecated and may not match current data structures.

	# Provable case expectation
	begin
		N1 = x -> x
		N2 = x -> x
		dim = 2
		G1_in = [1.0 0.0; 0.0 1.0]
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		epsilon = 0.1
		check = VeryDiff.get_epsilon_property_naive(epsilon)
		ok, _cex, _bnd, _split, _dist = check(N1, N2, Zin, Zout, nothing)
		@test ok == true
	end

	begin
		# Create a center difference big enough to exceed epsilon at the sampled point
		N1 = x -> x
		N2 = x -> x .+ [0.2, 0.0]
		dim = 2
		G1_in = reshape([1.0, 0.0], 2, 1)
        G1_out = [
            G1_in,
            reshape([0.2, 0.0], 2, 1)
        ]
        G2_out = [
            G1_in,
            reshape([0.2, 0.0], 2, 1)
        ]
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout = make_diff_zonotope(
			make_zonotope(G1_out, zeros(dim); ids=[1,2]),
			make_zonotope(G2_out, zeros(dim); ids=[1,3]),
			# Differential zonotope with bound > epsilon in first dim
			make_empty_zonotope(dim)
		)
		epsilon = 0.1
		check = VeryDiff.get_epsilon_property_naive(epsilon)
		ok, _cex, _bnd, _split, _dist = check(N1, N2, Zin, Zout, nothing)
		@test ok == false
	end
end

@testset "get_epsilon_property: not provable" begin
	# Intention: distance_bound > epsilon should yield ok == false.
    N1 = x -> x
    N2 = x -> x .+ [1.0, 0.0]
    dim = 2
    G1_in = reshape([1.0, 0.0], 2, 1)
    Zin = make_diff_zonotope(
        make_zonotope([G1_in], zeros(dim); ids=[1]),
        make_zonotope([G1_in], zeros(dim); ids=[1]),
        make_empty_zonotope(dim)
    )
    # Differential has large bound in dim 1 ⇒ not provable
    Zout = make_diff_zonotope(
        make_zonotope([G1_in], zeros(dim); ids=[1]),
        make_zonotope([G1_in], zeros(dim); ids=[1]),
        make_zonotope([ reshape([1.0, 0.0], 2, 1) ], zeros(dim); ids=[1])
    )
    epsilon = 0.1
    check = VeryDiff.get_epsilon_property(epsilon)
    ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)
    @test dist_bound > epsilon
    @test ok == false
end

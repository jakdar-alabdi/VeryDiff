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

@testset "get_epsilon_property: edge cases and robustness" begin
	# Edge: epsilon == 0, identical networks, zero differential ⇒ provable
	begin
		N1 = x -> x
		N2 = x -> x
		dim = 3
		G1_in = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
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
		epsilon = 0.0
		check = VeryDiff.get_epsilon_property(epsilon)
		ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)
		@test ok == true
		@test dist_bound == 0.0
	end

	# Edge: focus_dim at last index; only that dim has small bound ⇒ provable
	begin
		N1 = x -> x
		N2 = x -> x .+ [100.0, 100.0, 0.0]
		dim = 3
		G1_in = reshape([1.0, 0.0, 0.0], 3, 1)
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		# Differential bounds tiny only on dim 3
		Gd = reshape([10.0, 10.0, 0.05], 3, 1)
		Zout = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([Gd], zeros(dim); ids=[1])
		)
		epsilon = 0.1
		focus_dim = 3
		check = VeryDiff.get_epsilon_property(epsilon; focus_dim)
		ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)
		@test ok == true
		@test dist_bound <= epsilon
	end

	# Robustness: Zout has multiple generator blocks; first block column counts match Zin
	begin
		N1 = x -> x
		N2 = x -> x
		dim = 2
		G1_in = reshape([1.0, 0.0], 2, 1)
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout = make_diff_zonotope(
			make_zonotope([G1_in, zeros(dim, 3), zeros(dim, 2)], zeros(dim); ids=[1, 2, 4]),
			make_zonotope([G1_in, zeros(dim, 2)], zeros(dim); ids=[1, 3]),
			make_empty_zonotope(dim)
		)
		epsilon = 1e-6
		check = VeryDiff.get_epsilon_property(epsilon)
		ok, _cex, _bnd, _split, dist_bound = check(N1, N2, Zin, Zout, nothing)
		@test ok == true
		@test dist_bound <= epsilon
	end
end

@testset "get_epsilon_property: positive case via zono_get_max_vector path" begin
	# Goal: Ensure the method explores multiple points and only reports a violation in a later call.
	# We don't assume a particular search strategy; instead we construct cases where
	# the first call is non-violating (provable), and the subsequent call (with a different Zout) violates.

	dim = 2
	G1_in = reshape([1.0, 0.0], 2, 1)

	# Networks with a small offset along the generator direction
	N1 = x -> x
	N2 = x -> x .+ [0.05, 0.0]
	epsilon = 0.1

	# Case A: Differential bound below epsilon ⇒ provable
	begin
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout_A = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([ reshape([0.05, 0.0], 2, 1) ], zeros(dim); ids=[1])
		)
		check = VeryDiff.get_epsilon_property(epsilon)
		okA, _cexA, _bndA, _splitA, distA = check(N1, N2, Zin, Zout_A, nothing)
		@test okA == true
		@test distA <= epsilon
	end

	# Case B: Increase differential so bound exceeds epsilon ⇒ violation only now
	begin
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout_B = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([ reshape([0.2, 0.0], 2, 1) ], zeros(dim); ids=[1])
		)
		check = VeryDiff.get_epsilon_property(epsilon)
		okB, cexB, bndB, _splitB, distB = check(N1, N2, Zin, Zout_B, nothing)
		@test distB > epsilon
		@test okB == false
		# If a counterexample is returned, verify that it lies within Zin bounds
		if cexB !== nothing
			boundsZ1 = zono_bounds(Zin.Z₁)
			boundsZ2 = zono_bounds(Zin.Z₂)
			@test all(boundsZ1[:,1] .<= cexB[1] .<= boundsZ2[:,2])
		end
		# Or if only bounds info is returned, ensure it reflects > epsilon
		if bndB !== nothing
			out_bounds, eps_ret, fd_ret = bndB
			@test maximum(abs.(out_bounds)) > eps_ret
		end
	end
end

@testset "get_epsilon_property: epsilon equality boundary" begin
	# Bound exactly equals epsilon should be provable.
	dim = 2
	G1_in = reshape([1.0, 0.0], 2, 1)
	N1 = x -> x
	N2 = x -> x .+ [0.1, 0.0]
	epsilon = 0.1
	Zin = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	Zout = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([ reshape([0.1, 0.0], 2, 1) ], zeros(dim); ids=[1])
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test dist == epsilon
	@test ok == true

	# With focus_dim equality
	focus_dim = 1
	checkf = VeryDiff.get_epsilon_property(epsilon; focus_dim)
	okf, _cexf, _bndf, _splitf, distf = checkf(N1, N2, Zin, Zout, nothing)
	@test distf == epsilon
	@test okf == true
end

@testset "get_epsilon_property: very large epsilon" begin
	dim = 2
	G1_in = reshape([1.0, 0.0], 2, 1)
	N1 = x -> x
	N2 = x -> x .+ [1000.0, -500.0]
	epsilon = 1e6
	Zin = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	Zout = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([ reshape([1e3, 5e2], 2, 1) ], zeros(dim); ids=[1])
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test ok == true
	@test dist <= epsilon
end

@testset "get_epsilon_property: empty vs non-empty generators in Zout" begin
	# Zout.Z₁/Z₂ empty, ∂Z non-empty: bound should be based solely on ∂Z generators
	dim = 3
	N1 = x -> x
	N2 = x -> x
	epsilon = 0.5
	Zin = make_diff_zonotope(
		make_empty_zonotope(dim),
		make_empty_zonotope(dim),
		make_empty_zonotope(dim)
	)
	Gd = reshape([0.2, 0.1, 0.3], 3, 1)
	Zout = make_diff_zonotope(
		make_empty_zonotope(dim),
		make_empty_zonotope(dim),
		make_zonotope([Gd], zeros(dim); ids=[1])
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test ok == true
	@test dist <= epsilon
end

@testset "get_epsilon_property: mismatched block 1 column counts" begin
	# If block 1 column counts differ between Zin and Zout, downstream logic may error.
	# We expect either a graceful false or a thrown error; assert thrown if applicable.
	dim = 2
	N1 = x -> x
	N2 = x -> x
	G1_zin = reshape([1.0, 0.0], 2, 1)
	Zin = make_diff_zonotope(
		make_zonotope([G1_zin], zeros(dim); ids=[1]),
		make_zonotope([G1_zin], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	# Zout block 1 has 2 columns vs Zin's 1 column
	G1_zout = [G1_zin  G1_zin]
    Gd_zout = reshape([0.0, 0.0, 0.0, 0.0], 2, 2)
	Zout = make_diff_zonotope(
		make_zonotope([G1_zout], zeros(dim); ids=[1]),
		make_zonotope([G1_zout], zeros(dim); ids=[1]),
		make_zonotope([Gd_zout], ones(dim).*0.2; ids=[1])
	)
	epsilon = 0.1
	check = VeryDiff.get_epsilon_property(epsilon)
	# Use try-catch to accept either exception or a boolean false outcome
	outcome = try
		ok, _, _, _, _ = check(N1, N2, Zin, Zout, nothing)
		ok
	catch
		:error
	end
	@test outcome in (false, :error)
end

@testset "get_epsilon_property: invalid focus_dim" begin
	# focus_dim outside range should throw or return false.
	dim = 2
	N1 = x -> x
	N2 = x -> x
	G1_in = reshape([1.0, 0.0], 2, 1)
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
	focus_dim = 3 # invalid for dim=2
	check = VeryDiff.get_epsilon_property(epsilon; focus_dim)
	outcome = try
		ok, _, _, _, _ = check(N1, N2, Zin, Zout, nothing)
		ok
	catch
		:error
	end
	@test outcome in (false, :error)
end

@testset "get_epsilon_property: non-linear networks" begin
	# Simple non-linearity: ReLU-like behavior
	dim = 2
	G1_in = reshape([1.0, 0.0], 2, 1)
	N1 = x -> [max(0.0, x[1]), x[2]]
	N2 = x -> [max(0.0, x[1] + 0.05), x[2]]
	epsilon = 0.1
	Zin = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	Zout = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([ reshape([0.05, 0.0], 2, 1) ], zeros(dim); ids=[1])
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test ok == true
	@test dist <= epsilon
end

@testset "get_epsilon_property: differential center offset" begin
	# ∂Z with non-zero center and no generators: bound equals |c|
	dim = 2
	N1 = x -> x
	N2 = x -> x
	epsilon = 0.1
	G1_in = reshape([1.0, 0.0], 2, 1)
	Zin = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	Zd = make_empty_zonotope(dim)
	Zd.c .= [0.1, 0.0]
	Zout = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		Zd
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test dist == epsilon
	@test ok == true
end

@testset "get_epsilon_property: numerical robustness with tiny values" begin
	dim = 2
	N1 = x -> x
	N2 = x -> x .+ [1e-14, 0.0]
	epsilon = 1e-12
	G1_in = reshape([1.0, 0.0], 2, 1)
	Zin = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_empty_zonotope(dim)
	)
	Zout = make_diff_zonotope(
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([G1_in], zeros(dim); ids=[1]),
		make_zonotope([ reshape([1e-14, 0.0], 2, 1) ], zeros(dim); ids=[1])
	)
	check = VeryDiff.get_epsilon_property(epsilon)
	ok, _cex, _bnd, _split, dist = check(N1, N2, Zin, Zout, nothing)
	@test ok == true
	@test dist <= epsilon
end

@testset "get_epsilon_property: violation only on third call" begin
	# Gradually increase ∂Z bounds; ensure only the third call violates
	dim = 2
	G1_in = reshape([1.0, 0.0], 2, 1)
	N1 = x -> x
	N2 = x -> x .+ [0.05, 0.0]
	epsilon = 0.1

	function run_once(scale)
		Zin = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_empty_zonotope(dim)
		)
		Zout = make_diff_zonotope(
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([G1_in], zeros(dim); ids=[1]),
			make_zonotope([ reshape([scale, 0.0], 2, 1) ], zeros(dim); ids=[1])
		)
		check = VeryDiff.get_epsilon_property(epsilon)
		return check(N1, N2, Zin, Zout, nothing)
	end

	ok1, _, _, _, dist1 = run_once(0.05)
	@test ok1 == true
	@test dist1 <= epsilon

	ok2, _, _, _, dist2 = run_once(0.1)
	@test ok2 == true
	@test dist2 == epsilon

	ok3, _, _, _, dist3 = run_once(0.2)
	@test ok3 == false
	@test dist3 > epsilon
end

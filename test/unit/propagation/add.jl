using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, Zonotope, zono_bounds, reset_ps!
using Test
using LinearAlgebra
using Random

@testset "ONNXAdd Layer Propagation Tests" begin
    @info "Starting ONNXAdd Layer Propagation Tests"
    
    if "VERYDIFF_TEST_SEED" in keys(ENV)
        test_seed = parse(Int, ENV["VERYDIFF_TEST_SEED"])
        @info "Using VERYDIFF_TEST_SEED: $(test_seed)"
    else
        test_seed = rand(1:999999)
    end
    Random.seed!(test_seed)
    @info "Test seed: $(test_seed)"

    # ─── Helper: run the soundness check for a given network pair ───
    function check_sampled_points_within_bounds(N1_onnx, N2_onnx, input_dim, output_dim, num_samples, tolerance)
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        input_samples = sample_points_in_hypercube(low, high, num_samples)

        N_gemini = GeminiNetwork(N1_onnx, N2_onnx)
        N1 = executable_network(N1_onnx)
        N2 = executable_network(N2_onnx)

        verification_task = create_verification_task(low, high)
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)
        @debug "Propagation through Gemini Network complete."
        @debug "$(length(prop_state.zono_storage.zonotopes)) zonotopes in storage."
        Zout = prop_state.zono_storage.zonotopes[end].zonotope

        bounds_z1 = zono_bounds(Zout.Z₁)
        bounds_z2 = zono_bounds(Zout.Z₂)
        bounds_z∂ = zono_bounds(Zout.∂Z)

        n1_violations = 0
        n2_violations = 0
        diff_violations = 0
        for i in 1:num_samples
            x = input_samples[:, i]

            Zin = prop_state.zono_storage.zonotopes[1].zonotope.Z₁
            @assert Zin.c .+ Zin.Gs[1]*x ≈ x atol=1e-8
            Zin2 = prop_state.zono_storage.zonotopes[1].zonotope.Z₂
            @assert Zin2.c .+ Zin2.Gs[1]*x ≈ x atol=1e-8

            y1 = N1(x)
            y2 = N2(x)

            # Check Zonotope containment with additional generators
            Z1_range = sum(g->sum(abs, g, dims=2), Zout.Z₁.Gs[2:end]; init=zeros(size(Zout.Z₁.c)))
            Z2_range = sum(g->sum(abs, g, dims=2), Zout.Z₂.Gs[2:end]; init=zeros(size(Zout.Z₂.c)))
            input_component = Zout.Z₁.c .+ Zout.Z₁.Gs[1]*x
            @test all(input_component .- Z1_range .<= y1 .+ 1e-8)
            @test all(y1 .<= input_component .+ Z1_range .+ 1e-8)
            if !all(input_component .- Z1_range .<= y1 .+ 1e-8) || !all(y1 .<= input_component .+ Z1_range .+ 1e-8)
                @info "N1 output: $y1"
                @info "Zonotope bounds (agnostic): $(bounds_z1)"
                @info "Zonotope bounds (generator sum): $((input_component .- Z1_range, input_component .+ Z1_range))"
                return
            end
            input_component2 = Zout.Z₂.c .+ Zout.Z₂.Gs[1]*x
            @test all(input_component2 .- Z2_range .<= y2 .+ 1e-8)
            @test all(y2 .<= input_component2 .+ Z2_range .+ 1e-8)
            if !all(input_component2 .- Z2_range .<= y2 .+ 1e-8) || !all(y2 .<= input_component2 .+ Z2_range .+ 1e-8)
                @info "N2 output: $y2"
                @info "Zonotope bounds (agnostic): $(bounds_z2)"
                @info "Zonotope bounds (generator sum): $((input_component2 .- Z2_range, input_component2 .+ Z2_range))"
                return
            end
            # Difference Zonotope
            diff_range = sum(g->sum(abs, g, dims=2), Zout.∂Z.Gs[2:end]; init=zeros(size(Zout.∂Z.c)))
            if length(Zout.∂Z.Gs) >= 1
                input_component_diff = Zout.∂Z.c .+ Zout.∂Z.Gs[1]*x
            else
                input_component_diff = Zout.∂Z.c
            end
            @test all((input_component_diff .- diff_range) .<= (y1 .- y2) .+ 1e-8)
            @test all((y1 .- y2) .<= (input_component_diff .+ diff_range) .+ 1e-8)

            # Check if outputs are within bounds
            for d in 1:output_dim
                if y1[d] < bounds_z1[d, 1] - tolerance || y1[d] > bounds_z1[d, 2] + tolerance
                    n1_violations += 1
                end
                if y2[d] < bounds_z2[d, 1] - tolerance || y2[d] > bounds_z2[d, 2] + tolerance
                    n2_violations += 1
                end
                if (y1[d] - y2[d]) < bounds_z∂[d, 1] - tolerance || (y1[d] - y2[d]) > bounds_z∂[d, 2] + tolerance
                    diff_violations += 1
                end
            end
        end
        violations = n1_violations + n2_violations + diff_violations
        @info "Found $violations bound violations out of $(num_samples * 3 * output_dim) total checks"
        if violations > 0
            @info "Of which N1: $n1_violations"
            @info "Of which N2: $n2_violations"
            @info "Of which Diff: $diff_violations"
        end
        @test violations == 0
    end

    # ═══════════════════════════════════════════════════════════
    # 1. Fork-Join Topology (Dense only)
    # ═══════════════════════════════════════════════════════════
    @testset "Fork-Join: Basic Gemini Network Propagation" begin
        input_dim = 5
        output_dim = 10
        # layer_dims: [stem, branch, out...]
        layer_dims = [rand(20:50), rand(20:50), output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)

        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (output_dim,)
        @test size(Zout.Z₂.c) == (output_dim,)
    end

    @testset "Fork-Join: Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        output_dim = 10
        layer_dims = [rand(200:400), rand(200:400), output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)

        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)

        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)

        @info "Fork-Join: First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "Fork-Join: First run time: $time_1 s, Second run time: $time_2 s"

        # TODO: Re-enable when we don't run two versions of RELU
        #@test alloc_2 < alloc_1 * 0.4
    end

    @testset "Fork-Join: Sampled Points Within Output Bounds" begin
        input_dim = 10
        output_dim = 10
        num_samples = 20_000
        tolerance = 1e-4

        for depth in 1:8
            # layer_dims: [stem, branch, out_1, ..., out_depth]
            layer_dims = [rand(20:60), rand(20:60)]
            for _ in 1:(depth-1)
                push!(layer_dims, rand(20:60))
            end
            push!(layer_dims, output_dim)

            @debug "Fork-Join depth=$depth, layer_dims=$layer_dims"
            N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join)
            check_sampled_points_within_bounds(N1, N2, input_dim, output_dim, num_samples, tolerance)
        end
    end

    # ═══════════════════════════════════════════════════════════
    # 2. Fork-Join Topology with ReLU and AddConst
    # ═══════════════════════════════════════════════════════════
    @testset "Fork-Join + ReLU + AddConst: Basic Gemini Network Propagation" begin
        input_dim = 5
        output_dim = 10
        layer_dims = [rand(20:50), rand(20:50), output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join, relu=true, add_const=true)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)

        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (output_dim,)
        @test size(Zout.Z₂.c) == (output_dim,)
    end

    @testset "Fork-Join + ReLU + AddConst: Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        output_dim = 10
        layer_dims = [rand(200:400), rand(200:400), output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join, relu=true, add_const=true)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)

        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)

        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)

        @info "Fork-Join+ReLU+AddConst: First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "Fork-Join+ReLU+AddConst: First run time: $time_1 s, Second run time: $time_2 s"

        # TODO: Re-enable when we don't run two versions of RELU
        #@test alloc_2 < alloc_1 * 0.4
    end

    @testset "Fork-Join + ReLU + AddConst: Sampled Points Within Output Bounds" begin
        input_dim = 10
        output_dim = 10
        num_samples = 20_000
        tolerance = 1e-4

        for depth in 1:8
            layer_dims = [rand(20:60), rand(20:60)]
            for _ in 1:(depth-1)
                push!(layer_dims, rand(20:60))
            end
            push!(layer_dims, output_dim)

            @debug "Fork-Join+ReLU+AddConst depth=$depth, layer_dims=$layer_dims"
            N1, N2 = make_add_pair(input_dim, layer_dims; topology=:fork_join, relu=true, add_const=true)
            check_sampled_points_within_bounds(N1, N2, input_dim, output_dim, num_samples, tolerance)
        end
    end

    # ═══════════════════════════════════════════════════════════
    # 3. ResNet-style Topology (multiple residual blocks with ONNXAdd)
    # ═══════════════════════════════════════════════════════════
    @testset "ResNet: Basic Gemini Network Propagation" begin
        input_dim = 5
        output_dim = 10
        hidden_dim = rand(20:50)
        # layer_dims: [hidden, hidden, ..., output]  — 3 residual blocks
        layer_dims = [hidden_dim, hidden_dim, hidden_dim, output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)

        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (output_dim,)
        @test size(Zout.Z₂.c) == (output_dim,)
    end

    @testset "ResNet: Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        output_dim = 10
        hidden_dim = rand(200:400)
        layer_dims = [hidden_dim, hidden_dim, hidden_dim, output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)

        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)

        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)

        @info "ResNet: First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "ResNet: First run time: $time_1 s, Second run time: $time_2 s"

        # TODO: Re-enable when we don't run two versions of RELU
        #@test alloc_2 < alloc_1 * 0.4
    end

    @testset "ResNet: Sampled Points Within Output Bounds" begin
        input_dim = 10
        output_dim = 10
        num_samples = 20_000
        tolerance = 1e-4

        for num_blocks in 1:8
            hidden_dim = rand(20:60)
            layer_dims = fill(hidden_dim, num_blocks)
            push!(layer_dims, output_dim)

            @debug "ResNet num_blocks=$num_blocks, hidden_dim=$hidden_dim"
            N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet)
            check_sampled_points_within_bounds(N1, N2, input_dim, output_dim, num_samples, tolerance)
        end
    end

    # ═══════════════════════════════════════════════════════════
    # 4. ResNet-style with ReLU and AddConst
    # ═══════════════════════════════════════════════════════════
    @testset "ResNet + ReLU + AddConst: Basic Gemini Network Propagation" begin
        input_dim = 5
        output_dim = 10
        hidden_dim = rand(20:50)
        layer_dims = [hidden_dim, hidden_dim, hidden_dim, output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet, relu=true, add_const=true)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        prop_state = propagate!(N_gemini, prop_state)

        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (output_dim,)
        @test size(Zout.Z₂.c) == (output_dim,)
    end

    @testset "ResNet + ReLU + AddConst: Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        output_dim = 10
        hidden_dim = rand(200:400)
        layer_dims = [hidden_dim, hidden_dim, hidden_dim, output_dim]

        N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet, relu=true, add_const=true)
        N_gemini = GeminiNetwork(N1, N2)

        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        verification_task = create_verification_task(low, high)

        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)

        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)

        @info "ResNet+ReLU+AddConst: First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "ResNet+ReLU+AddConst: First run time: $time_1 s, Second run time: $time_2 s"

        # TODO: Re-enable when we don't run two versions of RELU
        #@test alloc_2 < alloc_1 * 0.4
    end

    @testset "ResNet + ReLU + AddConst: Sampled Points Within Output Bounds" begin
        input_dim = 10
        output_dim = 10
        num_samples = 20_000
        tolerance = 1e-4

        for num_blocks in 1:8
            hidden_dim = rand(20:60)
            layer_dims = fill(hidden_dim, num_blocks)
            push!(layer_dims, output_dim)

            @debug "ResNet+ReLU+AddConst num_blocks=$num_blocks, hidden_dim=$hidden_dim"
            N1, N2 = make_add_pair(input_dim, layer_dims; topology=:resnet, relu=true, add_const=true)
            check_sampled_points_within_bounds(N1, N2, input_dim, output_dim, num_samples, tolerance)
        end
    end

end

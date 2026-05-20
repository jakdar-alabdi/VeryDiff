function deepsplit_verify_network(
    N₁::OnnxNet{LayerIdT,NShapeIn, NShapeOut}, 
    N₂::OnnxNet{LayerIdT,NShapeIn, NShapeOut}, 
    Zin::Zonotope, 
    property_check;
    timeout=Inf) where {LayerIdT,NShapeIn,NShapeOut}
    return deepsplit_verify_network(N₁, N₂, zono_bounds(Zin), property_check; timeout=Inf)
end

function deepsplit_verify_network(
    N₁::OnnxNet{LayerIdT,NShapeIn, NShapeOut}, 
    N₂::OnnxNet{LayerIdT,NShapeIn, NShapeOut}, 
    bounds, 
    property_check; 
    timeout=Inf, 
    fuzz_testing=nothing) where {LayerIdT,NShapeIn,NShapeOut}

    start_time = time_ns()
    try
        VeryDiff.NEW_HEURISTIC[] = false

        lower = @view bounds[:, 1]
        upper = @view bounds[:, 2]
        mid = (upper .+ lower) ./ 2
        distance = mid .- lower
        non_zero_indices = findall((!).(iszero.(distance)))
        distance = distance[non_zero_indices]
        
        initial_task = VerificationTask(mid, distance, non_zero_indices, nothing, nothing, nothing, nothing, nothing, Inf64, 1.0, Branch())

        N = GeminiNetwork(N₁, N₂)
        N₁ = executable_network(N₁)
        N₂ = executable_network(N₂)

        if N.diff_layers[end] isa VeryDiff.Definitions.DiffLayer{VNNLib.OnnxParser.ONNXSoftmax{S},VNNLib.OnnxParser.ONNXSoftmax{S},VNNLib.OnnxParser.ONNXSoftmax{S}} where S
            pop!(N.diff_layers)
            @warn "Removed final Softmax layer from differential network for verification."
            @warn "VeryDiff assumes this is handled by the choice of an appropriate property!"
        end

        veri_result, cex = deepsplit_verify_network(N, N₁, N₂, initial_task, property_check; timeout=timeout)

        if !isnothing(cex)
            println("Found counterexample: $cex")
        end
        println("Initial δ-bound: $(veri_result.initial_δ_bound), Final δ-bound: $(veri_result.final_δ_bound)")
        println("Verification Status: $(veri_result.status)")

        return veri_result
    catch e
        println("Caught an exception:")
        showerror(stderr, e, catch_backtrace())
        veri_result = VerificationResult()
        veri_result.verification_time = time_ns() - start_time
        return veri_result
    end
end

function deepsplit_verify_network(N::GeminiNetwork, N₁::Network, N₂::Network, initial_task::VerificationTask, property_check; timeout=Inf)
    relative_impact_func = get_relative_impact_func()

    prop_state = PropState(true)
    veri_result = VerificationResult()

    relu_layers = get_relu_layers(N)
    
    first_task = true
    global VeryDiff.FIRST_ROUND[] = true
        
    queue = Queue()
    push!(queue, initial_task)
    
    start_time = time_ns()
    while !isempty(queue)
        task = pop!(queue)
        veri_result.final_δ_bound = task.distance_bound
        @info "Distance Bound: $(task.distance_bound)"
        
        if !check_resources(start_time, timeout)
            empty!(queue)
            GC.gc()
            veri_result.verification_time = time_ns() - start_time
            return veri_result, nothing
        end

        prepare_prop_state!(prop_state, task)
        Zin = prop_state.zono_storage.zonotopes[1].zonotope
        prop_state = propagate!(N, prop_state)
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        veri_result.num_propagations += 1

        if prop_state.is_unsatisfiable
            continue
        end

        # @info "Z₁.Gs sizes: $(size.(Zout.Z₁.Gs, 2))"
        # @info "Z₂.Gs sizes: $(size.(Zout.Z₂.Gs, 2))"
        # @info "∂Z.Gs sizes: $(size.(Zout.∂Z.Gs, 2))"
        # @info "NumInstable: $(prop_state.num_instable)"
        # @assert (size(Zout.∂Z.Gs[2], 2) + size(Zout.∂Z.Gs[3], 2)) == prop_state.num_instable

        if first_task
            bounds = zono_bounds(Zout.∂Z)
            veri_result.initial_δ_bound = maximum(abs, bounds)
            veri_result.final_δ_bound = veri_result.initial_δ_bound
            first_task = false
            task.branch.undetermined = trues(size(bounds))
            println("Zono Bounds:")
            println(bounds[:, 1])
            println(bounds[:, 2])
        end
        
        prop_satisfied, cex, _, verification_status, distance_bound, box = property_check(N₁, N₂, prop_state)
        distance_bound = min(distance_bound, task.distance_bound)
        global VeryDiff.FIRST_ROUND[] = false
    
        if !prop_satisfied
            if !isnothing(cex)
                veri_result.status = UNSAFE
                veri_result.verification_time = time_ns() - start_time
                return veri_result, cex
            end

            if prop_state.num_instable == 0 && VeryDiff.EQUIVALENCE_PROPERTY[] == VeryDiff.DeltaTop1Equivalence
                veri_result.verification_time = time_ns() - start_time
                return veri_result, nothing
            end
            @assert prop_state.num_instable > 0

            split_nodes = prop_state.task.branch.split_nodes
            if VeryDiff.USE_ZONO_CONTRACT[] && !isempty(split_nodes)
                if isnothing(box)
                    box = InputBox(Zout)
                end
                box = contract_zono_all!(box, split_nodes, prop_state)
                if isnothing(box)
                    continue
                end
            end
            
            split_candidate = deepsplit_heuristic(prop_state, relu_layers, relative_impact_func)
            if split_candidate.layer == 0
                if VeryDiff.USE_ZONO_CONTRACT[] && !isempty(split_nodes)
                    task₁, task₂ = split_contract_zono(split_candidate.neuron, box, prop_state, verification_status, distance_bound)
                else
                    task₁, task₂ = split_zono(split_candidate.neuron, task, verification_status, distance_bound)
                end
                veri_result.num_input_splits += !isnothing(task₁) || !isnothing(task₂)
            else
                task₁, task₂ = split_neuron(split_candidate, box, prop_state, verification_status, distance_bound)
                veri_result.num_neuron_splits += !isnothing(task₁) || !isnothing(task₂)
            end

            if !isnothing(task₁)
                push!(queue, task₁)
            end
            if !isnothing(task₂)
                push!(queue, task₂)
            end
        end

        reset_ps!(prop_state)
    end
    veri_result.status = SAFE
    veri_result.verification_time = time_ns() - start_time
    return veri_result, nothing
end

function split_neuron(node::SplitNode, box::Union{Nothing,InputBox}, prop_state::PropState, verification_status, distance_bound::Float64)
    if !VeryDiff.PRE_CONTRACT[] && !isnothing(box) && !is_unit_hypercube(box)
        transform_verification_task!(box, prop_state.task)
    end

    direction₁, direction₂ = -1, 1 # inactive, active
    bounds₁, bounds₂ = nothing, nothing
    (;network, layer, neuron, diff_layer) = node
    
    old_node_idx = nothing
    if VeryDiff.USE_VERTICAL_SPLITTING[]
        split_nodes = prop_state.task.branch.split_nodes
        old_node_idx = findfirst(n -> (n.network, n.layer, n.neuron) == (network, layer, neuron), split_nodes)
        if !isnothing(old_node_idx)
            old_node = split_nodes[old_node_idx]
            if old_node.direction == 1
                l̲, u̲ = old_node.bounds[1], old_node.bounds[2]
                s₁, s₂ = l̲ / 2, u̲ / 2
                bounds₁ = [l̲ s₁; s₂ u̲]
                bounds₂ = [s₁ s₂]
            else
                direction₂ = -1
                l̅, s̅₁ = old_node.bounds[1, 1], old_node.bounds[1, 2]
                s̅₂, u̅ = old_node.bounds[2, 1], old_node.bounds[2, 2]
                s₁, s₂ = (l̅ + s̅₁) / 2, (s̅₂ + u̅) / 2
                bounds₁ = [l̅ s₁; s₂ u̅]
                bounds₂ = [s₁ s̅₁; s̅₂ s₂]
            end
        end
    end

    (;middle, distance, distance_indices, distance1_secondary, middle1_secondary, 
    distance2_secondary, middle2_secondary, work_share, task_bounds, branch) = prop_state.task
    
    branch₁, branch₂ = branch, deepcopy(branch)
    node₁ = SplitNode(network, layer, neuron, diff_layer, direction₁, bounds₁)
    node₂ = SplitNode(network, layer, neuron, diff_layer, direction₂, bounds₂)
    if isnothing(old_node_idx)     
        push!(branch₁.split_nodes, node₁)
        push!(branch₂.split_nodes, node₂)
    else
        branch₁.split_nodes[old_node_idx] = node₁
        branch₂.split_nodes[old_node_idx] = node₂
    end
    
    task₁ = VerificationTask(
        middle, distance, distance_indices, distance1_secondary, middle1_secondary, distance2_secondary, 
        middle2_secondary, verification_status, distance_bound, work_share, task_bounds, branch₁
    )
    task₂ = VerificationTask(
        deepcopy(middle), deepcopy(distance), deepcopy(distance_indices), deepcopy(distance1_secondary), 
        deepcopy(middle1_secondary), deepcopy(distance2_secondary), deepcopy(middle2_secondary), 
        deepcopy(verification_status), distance_bound, work_share, deepcopy(task_bounds), branch₂
    )

    if VeryDiff.PRE_CONTRACT[]
        Z = get_split_node_zono(node, prop_state)
        if isnothing(box)
            box₁, box₂ = InputBox(Z), InputBox(Z)
        else
            box₁, box₂ = box, InputBox(box)
        end
        task₁ = contract_to_verification_task!(box₁, node₁, Z, task₁)
        task₂ = contract_to_verification_task!(box₂, node₂, Z, task₂)
    end

    return task₁, task₂
end

function split_contract_zono(d::Int, box::InputBox, prop_state::PropState, verification_status, distance_bound::Float64)
    distance_d = findfirst(x -> x == d, prop_state.task.distance_indices)
    @assert !isnothing(distance_d)
    
    box₁, box₂ = box, InputBox(box)
    
    cutting_point = (box.lowers[1][distance_d] + box.uppers[1][distance_d]) / 2
    box₁.lowers[1][distance_d] = cutting_point
    box₂.uppers[1][distance_d] = cutting_point

    (;middle, distance, distance_indices, distance1_secondary, middle1_secondary, 
    distance2_secondary, middle2_secondary, work_share, task_bounds, branch) = prop_state.task

    box₁ = contract_zono_all!(box₁, branch.split_nodes, prop_state)
    task₁ = nothing
    if !isnothing(box₁)
        task₁ = VerificationTask(
            middle, distance, distance_indices, distance1_secondary, middle1_secondary, distance2_secondary, 
            middle2_secondary, verification_status, distance_bound, work_share / 2, task_bounds, branch
        )
        task₁ = transform_verification_task!(box₁, task₁)
    end

    box₂ = contract_zono_all!(box₂, branch.split_nodes, prop_state)
    task₂ = nothing
    if !isnothing(box₂)
        f = ifelse(isnothing(task₁), identity, deepcopy)
        task₂ = VerificationTask(
            f(middle), f(distance), f(distance_indices), f(distance1_secondary), 
            f(middle1_secondary), f(distance2_secondary), f(middle2_secondary), 
            f(verification_status), distance_bound, work_share / 2, f(task_bounds), f(branch)
        )
        task₂ = transform_verification_task!(box₂, task₂)
    end

    return task₁, task₂
end

function check_resources(start_time::UInt64, timeout=Inf)
    timeout_reached = (time_ns() - start_time) / 1.0e9 > timeout
    if timeout_reached
        println("\nTIMEOUT REACHED")
    end
    return !timeout_reached
end

function get_relu_layers(N::GeminiNetwork) :: Vector{DiffLayer{
        VeryDiff.VNNLib.OnnxParser.ONNXRelu{S1},
        VeryDiff.VNNLib.OnnxParser.ONNXRelu{S2},
        VeryDiff.VNNLib.OnnxParser.ONNXRelu{S3}} where {S1,S2,S3}}

    isdiffrelu = l -> l isa VeryDiff.Definitions.DiffLayer{VNNLib.OnnxParser.ONNXRelu{S1},VNNLib.OnnxParser.ONNXRelu{S2},VNNLib.OnnxParser.ONNXRelu{S3}} where {S1,S2,S3}
    relu_layers_pos = findall(isdiffrelu, get_layers(N))
    return @view get_layers(N)[relu_layers_pos]
end

function get_split_node_diffzono(node::SplitNode, prop_state::PropState) :: DiffZonotope
    inputs = get_zonos_at_pos(get_inputs(node.diff_layer), prop_state)
    return get_zonotope(inputs[1])
end

function get_split_node_zono(node::SplitNode, prop_state::PropState) :: Zonotope
    DZ = get_split_node_diffzono(node, prop_state)
    return ifelse(node.network == 1, DZ.Z₁, DZ.Z₂)
end

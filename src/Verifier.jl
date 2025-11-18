#using Random

@enum VerificationStatus UNKNOWN SAFE UNSAFE

function verify_network(
    N1 :: Network,
    N2 :: Network,
    bounds,
    property_check,
    split_heuristic;
    timeout=Inf,
    init_eps=0.0)
    global FIRST_ROUND = true
    verification_result = nothing
    # Timing
    reset_timer!(to)
    @timeit to "Initialize" begin
        # Prepare Zonotope Initialization
        #@timeit to "Prep_Zono_Init" begin
        input_dim = size(bounds,1)
        low = @view bounds[:,1]
        high = @view bounds[:,2]
        mid = (high.+low) ./ 2
        distance = mid .- low
        input_dim = length(low)
        #end

        # Initialize Zonotope
        #@timeit to "Zono_Init" begin
        non_zero_indices = findall((!).(iszero.(distance)))
        distance = distance[non_zero_indices]
        if init_eps > 0.0
            if !all(init_eps .<= (2.0 .* distance))
                println("ERROR: Initial epsilon too large for given bounds!")
                println("Inital Epsilon: $(init_eps); Max Epsilon: $(2.0*minimum(distance))")
                raise(ErrorException("Initial epsilon too large for given bounds!"))
            end
            distance .-= (init_eps/2)
            distance1_secondary = fill(init_eps/2, input_dim)
            distance2_secondary = fill(init_eps/2, input_dim)
            mid1_secondary = zeros(Float64, input_dim)
            mid2_secondary = zeros(Float64, input_dim)
        else
            println("Differential Zonotope initialized with zero perturbation.")
            distance1_secondary = nothing
            distance2_secondary = nothing
            mid1_secondary = nothing
            mid2_secondary = nothing
        end
    end

    #@timeit to "Network_Init" begin
    N = GeminiNetwork(N1,N2)
    println("Network initialized.")
    #end

    # Statistics
    #@timeit to "Statistics_Init" begin
    total_zonos = 1
    #end

    # Property
    # property_check = get_epsilon_property(epsilon;focus_dim=focus_dim)
    # split_heuristic = epsilon_split_heuristic
    # property_check = get_top1_property(1.0)
    # split_heuristic = top1_configure_split_heuristic(3) #epsilon_split_heuristic

    #Config
    prop_state = PropState(true)
    num_threads = Threads.nthreads()
    println("Running with $(num_threads) threads")
    #single_threaded = num_threads == 1
    #if single_threaded
    #    common_state = MultiThreaddedQueue(1)
    #else
    #    common_state = MultiThreaddedQueue(num_threads)
    #end
    work_queue = Queue()
    push!(work_queue,
        (1.0,VerificationTask(
            mid, distance, non_zero_indices,
            distance1_secondary, mid1_secondary,
            distance2_secondary, mid2_secondary,
            nothing, 1.0, [0] ) )
    )
    @timeit to "Verify" begin
        @Debugger.propagation_init_hook(N, prop_state)
        #if single_threaded
        verification_result = worker_function(work_queue, 1, prop_state,N,N1,N2, property_check, split_heuristic,num_threads;timeout=timeout)
        if verification_result == SAFE
            println("SAFE")
        elseif verification_result == UNSAFE
            println("UNSAFE")
        else
            println("UNKNOWN")
        end
    end
    show(to)
    #common_state=nothing
    work_queue=nothing
    return verification_result
end

function worker_function(work_queue, threadid, prop_state,N,N1,N2,property_check, split_heuristic, num_threads;timeout=Inf)
    try
        thread_result = worker_function_internal(work_queue, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic, timeout=timeout)
        return thread_result
    catch e
        println("[Thread $(threadid)] Caught exception: $(e)")
        showerror(stdout, e, catch_backtrace())
        thread_result = UNKNOWN
        return thread_result
    end
end
function worker_function_internal(work_queue, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic;timeout=Inf)
    # @debug "Worker initiated on thread $(threadid)"
    starttime = time_ns()
    prop_state = deepcopy(prop_state)
    k = 0
    total_zonos=0
    generated_zonos = 0
    splits = 0
    is_verified = SAFE
    # @debug "[Thread $(threadid)] Starting worker"
    #task_queue = Queue()
    # @debug "[Thread $(threadid)] Syncing queues"
    #should_terminate = sync_queues!(threadid, qork, task_queue)
    #sync_res = @timed sync_queues!(threadid, common_state, task_queue)
    should_terminate = length(work_queue) == 0
    do_not_split = false
    #wait_time = sync_res.time
    total_work = 0.0
    first=true
    # @debug "[Thread $(threadid)] Initiating loop"
    @timeit to "Zonotope Loop" begin
    loop_time = @elapsed begin
    while !should_terminate
        input_dim=1
        try
            work_share, verification_task = pop!(work_queue)
            Zin = to_diff_zono(verification_task)
            input_dim = size(Zin.Z₁.G,2)
            if k == 0
                println("[Thread $(threadid)] Time to first task: $(round((time_ns()-starttime)/1e9;digits=2))s")
            end
            # @debug "[Thread $(threadid)] got work share $(work_share) running on $(Threads.threadid())"
            #println("Processing task on thread $(threadid)")
            total_zonos+=1
            # Initial Pass
            #prop_state.i = 1
            @timeit to "Zonotope Propagate" begin
            Zout = N(Zin, prop_state)
            end
            if first #k%5 == 0
                #println("Zono Bounds:")
                if VeryDiff.USE_DIFFZONO
                    bounds = zono_bounds(Zout.∂Z)
                else
                    G = zeros(Float64, size(Zout.Z₁.G,1), size(Zout.Z₁.G,2) + size(Zout.Z₂.G,2) - size(Zout.∂Z.G,2))
                    G[:,1:size(Zout.Z₁.G,2)] .= Zout.Z₁.G
                    G[:,1:size(Zout.∂Z.G,2)] .-= Zout.Z₂.G[:,1:size(Zout.∂Z.G,2)]
                    G[:,(size(Zout.Z₁.G,2)+1):end] .= -Zout.Z₂.G[:, (size(Zout.∂Z.G,2)+1):end]
                    c = Zout.Z₁.c .- Zout.Z₂.c
                    bounds = zono_bounds(Zonotope(G,c,nothing))
                end
                println(bounds[:,1])
                println(bounds[:,2])
                first=false
            end

            @timeit to "Property Check" begin
            prop_satisfied, cex, heuristics_info, verification_status, distance_bound = property_check(N1, N2, Zin, Zout,
                verification_task.verification_status)
            end
            global FIRST_ROUND = false
            if !prop_satisfied
                if !isnothing(cex)
                    @assert all(zono_bounds(Zin.Z₁)[:,1] .<= cex[1] .&& cex[1] .<= zono_bounds(Zin.Z₂)[:,2])
                    println("\nFound counterexample: $(cex)")
                    should_terminate = true
                    is_verified = UNSAFE
                elseif !do_not_split
                    @timeit to "Compute Split" begin
                    splits += 1
                    split_d = split_heuristic(Zin,Zout,heuristics_info, verification_task)
                    #println("[Thread $(threadid)] Splitting on dimension $(split_d) ([$(verification_task.lower_bounds[split_d]), $(verification_task.upper_bounds[split_d])]) with work share $(work_share)")
                    Z1, Z2 = split_zono(split_d, verification_task,work_share,verification_status, distance_bound)
                    # Z1z = to_diff_zono(Z1[2])
                    # Z2z = to_diff_zono(Z2[2])
                    # zono1_bounds1_low = zono_optimize(-1.0,Z1z.Z₁,split_d)
                    # zono1_bounds1_high = zono_optimize(1.0,Z1z.Z₁,split_d)
                    # zono1_bound2_low = zono_optimize(-1.0,Z1z.Z₂,split_d)
                    # zono1_bound2_high = zono_optimize(1.0,Z1z.Z₂,split_d)
                    # zono1_boundsd_low = zono_optimize(-1.0,Z1z.∂Z,split_d)
                    # zono1_boundsd_high = zono_optimize(1.0,Z1z.∂Z,split_d)
                    # println(" Z1 Bounds on d=$(split_d): Z1: [$(zono1_bounds1_low), $(zono1_bounds1_high)], Z2: [$(zono1_bound2_low), $(zono1_bound2_high)], ∂Z: [$(zono1_boundsd_low), $(zono1_boundsd_high)]")
                    # zono2_bounds1_low = zono_optimize(-1.0,Z2z.Z₁,split_d)
                    # zono2_bounds1_high = zono_optimize(1.0,Z2z.Z₁,split_d)
                    # zono2_bound2_low = zono_optimize(-1.0,Z2z.Z₂,split_d)
                    # zono2_bound2_high = zono_optimize(1.0,Z2z.Z₂,split_d)
                    # zono2_boundsd_low = zono_optimize(-1.0,Z2z.∂Z,split_d)
                    # zono2_boundsd_high = zono_optimize(1.0,Z2z.∂Z,split_d)
                    # println(" Z2 Bounds on d=$(split_d): Z1: [$(zono2_bounds1_low), $(zono2_bounds1_high)], Z2: [$(zono2_bound2_low), $(zono2_bound2_high)], ∂Z: [$(zono2_boundsd_low), $(zono2_boundsd_high)]")
                    Zin=nothing
                    push!(work_queue, Z1)
                    push!(work_queue, Z2)
                    generated_zonos+=2
                    end
                end
            else
                total_work += work_share
            end
        finally
            # nothing right now
        end
        if (time_ns()-starttime)/1e9 > timeout
            println("\n\nTIMEOUT REACHED")
            println("UNKNOWN")
            should_terminate = true
            is_verified = UNKNOWN
        end
        # Even if we haven't done any work we may find a counterexample, but unfortunately
        # memory is bounded...
        if length(work_queue) > 2500 && total_work < 1e-3 && input_dim > 100
            println("2500 Zonotopes in task queue: NO MORE SPLITTING!")
            println("This is to avoid memory overflows.")
            println("WARNING: FROM THIS POINT ONWARDS, WE ARE ONLY SEARCHING FOR COUNTEREXAMPLES!")
            do_not_split = true
        end
        should_terminate |= length(work_queue) == 0
        k+=1
        if k%100 == 0
            println("[Thread $(threadid)] Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=5))%; Expected Zonos: $(total_zonos/total_work))")
        end
        #end
    end
    end
    end
    empty!(work_queue)
    if do_not_split
        println("\n\nTIMEOUT REACHED")
        println("UNKNOWN")
        is_verified = UNKNOWN
    end
    println("[Thread $(threadid)] Total splits: $(splits)")
    print("Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%); Generated $(generated_zonos) ($(loop_time/k)s/loop)\n")
    return is_verified
end

function split_zono(distance_d, verification_task :: VerificationTask, work_share, verification_status, distance_bound)
    # TODO(steuber): Make split prettier?
    (work_share,Z1), (work_share,Z2), input_pos = (() -> begin
    #split_stage = verification_task.split_stage[distance_d]
    #print("Work Share: $(work_share)")
    #print(verification_task.split_stage)
    #verification_task.split_stage[distance_d] = (split_stage + 1) % 3
    if distance_d <= size(verification_task.distance_indices,1)
        #split_stage == 2
        #println("Splitting on input dimension $(verification_task.distance_indices[distance_d])")
        input_pos = verification_task.distance_indices[distance_d]
        distance1 = verification_task.distance[distance_d]/2
        distance2 = verification_task.distance[distance_d]/2
        mid1 = verification_task.middle[input_pos] - distance1
        mid2 = verification_task.middle[input_pos] + distance2
        distance1_vec = deepcopy(verification_task.distance)
        distance1_vec[distance_d] = distance1
        middle1_vec = deepcopy(verification_task.middle)
        middle1_vec[input_pos] = mid1
        distance2_vec = verification_task.distance
        distance2_vec[distance_d] = distance2
        middle2_vec = verification_task.middle
        middle2_vec[input_pos] = mid2
        Z1 = VerificationTask(
            middle1_vec, distance1_vec,
            deepcopy(verification_task.distance_indices),
            deepcopy(verification_task.distance1_secondary),
            deepcopy(verification_task.middle1_secondary),
            deepcopy(verification_task.distance2_secondary),
            deepcopy(verification_task.middle2_secondary),
            deepcopy(verification_status),
            distance_bound,
            deepcopy(verification_task.split_stage))
        Z2 = VerificationTask(
            middle2_vec, distance2_vec,
            verification_task.distance_indices,
            verification_task.distance1_secondary,
            verification_task.middle1_secondary,
            verification_task.distance2_secondary,
            verification_task.middle2_secondary,
            verification_status,
            distance_bound,
            verification_task.split_stage)
        return (work_share/2.0,Z1), (work_share/2.0,Z2), input_pos
    elseif distance_d <= size(verification_task.distance_indices,1) + size(verification_task.distance1_secondary,1)
        #split_stage == 1
        input_pos = distance_d - size(verification_task.distance_indices,1)
        #println("Splitting on 1 differential dimension $(input_pos)")
        diff_distance1 = verification_task.distance1_secondary[input_pos]/2
        diff_distance2 = verification_task.distance1_secondary[input_pos]/2
        diff_mid1 = verification_task.middle1_secondary[input_pos] - diff_distance1
        diff_mid2 = verification_task.middle1_secondary[input_pos] + diff_distance2
        distance1_secondary = deepcopy(verification_task.distance1_secondary)
        distance1_secondary[input_pos] = diff_distance1
        middle1_secondary = deepcopy(verification_task.middle1_secondary)
        middle1_secondary[input_pos] = diff_mid1
        distance2_secondary = deepcopy(verification_task.distance1_secondary)
        distance2_secondary[input_pos] = diff_distance2
        middle2_secondary = deepcopy(verification_task.middle1_secondary)
        middle2_secondary[input_pos] = diff_mid2
        Z1 = VerificationTask(
            deepcopy(verification_task.middle), deepcopy(verification_task.distance),
            deepcopy(verification_task.distance_indices),
            distance1_secondary,
            middle1_secondary,
            deepcopy(verification_task.distance2_secondary),
            deepcopy(verification_task.middle2_secondary),
            deepcopy(verification_status),
            distance_bound,
            deepcopy(verification_task.split_stage))
        Z2 = VerificationTask(
            verification_task.middle, verification_task.distance,
            verification_task.distance_indices,
            distance2_secondary,
            middle2_secondary,
            verification_task.distance2_secondary,
            verification_task.middle2_secondary,
            verification_status,
            distance_bound,
            verification_task.split_stage)
        return (work_share/2.0,Z1), (work_share/2.0,Z2), input_pos
    else #split_stage == 0
        input_pos = distance_d - size(verification_task.distance_indices,1) - size(verification_task.distance1_secondary,1)
        #println("Splitting on 2 differential dimension $(input_pos)")
        diff_distance1 = verification_task.distance2_secondary[input_pos]/2
        diff_distance2 = verification_task.distance2_secondary[input_pos]/2
        diff_mid1 = verification_task.middle2_secondary[input_pos] - diff_distance1
        diff_mid2 = verification_task.middle2_secondary[input_pos] + diff_distance2
        distance1_secondary = deepcopy(verification_task.distance2_secondary)
        distance1_secondary[input_pos] = diff_distance1
        middle1_secondary = deepcopy(verification_task.middle2_secondary)
        middle1_secondary[input_pos] = diff_mid1
        distance2_secondary = deepcopy(verification_task.distance2_secondary)
        distance2_secondary[input_pos] = diff_distance2
        middle2_secondary = deepcopy(verification_task.middle2_secondary)
        middle2_secondary[input_pos] = diff_mid2
        Z1 = VerificationTask(
            deepcopy(verification_task.middle), deepcopy(verification_task.distance),
            deepcopy(verification_task.distance_indices),
            deepcopy(verification_task.distance1_secondary),
            deepcopy(verification_task.middle1_secondary),
            distance1_secondary,
            middle1_secondary,
            deepcopy(verification_status),
            distance_bound,
            deepcopy(verification_task.split_stage))
        Z2 = VerificationTask(
            verification_task.middle, verification_task.distance,
            verification_task.distance_indices,
            verification_task.distance1_secondary,
            verification_task.middle1_secondary,
            distance2_secondary,
            middle2_secondary,
            verification_status,
            distance_bound,
            verification_task.split_stage)
        return (work_share/2.0,Z1), (work_share/2.0,Z2), input_pos
    end
    end)()

    # # DEBUG
    # println("Original Bounds on d=$(input_pos)")
    # Z1z = to_diff_zono(Z1)
    # zono1_bounds1_low = zono_optimize(-1.0,Z1z.Z₁,input_pos)
    # zono1_bounds1_high = zono_optimize(1.0,Z1z.Z₁,input_pos)
    # zono1_bound2_low = zono_optimize(-1.0,Z1z.Z₂,input_pos)
    # zono1_bound2_high = zono_optimize(1.0,Z1z.Z₂,input_pos)
    # zono1_boundsd_low = zono_optimize(-1.0,Z1z.∂Z,input_pos)
    # zono1_boundsd_high = zono_optimize(1.0,Z1z.∂Z,input_pos)
    # println(" Z1 Bounds on d=$(input_pos): Z1: [$(zono1_bounds1_low), $(zono1_bounds1_high)], Z2: [$(zono1_bound2_low), $(zono1_bound2_high)], ∂Z: [$(zono1_boundsd_low), $(zono1_boundsd_high)]")
    # # END DEBUG

    # # DEBUG
    # Z2z = to_diff_zono(Z2)
    # zono2_bounds1_low = zono_optimize(-1.0,Z2z.Z₁,input_pos)
    # zono2_bounds1_high = zono_optimize(1.0,Z2z.Z₁,input_pos)
    # zono2_bound2_low = zono_optimize(-1.0,Z2z.Z₂,input_pos)
    # zono2_bound2_high = zono_optimize(1.0,Z2z.Z₂,input_pos)
    # zono2_boundsd_low = zono_optimize(-1.0,Z2z.∂Z,input_pos)
    # zono2_boundsd_high = zono_optimize(1.0,Z2z.∂Z,input_pos)
    # println(" Z2 Bounds on d=$(input_pos): Z1: [$(zono2_bounds1_low), $(zono2_bounds1_high)], Z2: [$(zono2_bound2_low), $(zono2_bound2_high)], ∂Z: [$(zono2_boundsd_low), $(zono2_boundsd_high)]")
    # # END DEBUG

    return (work_share,Z1), (work_share,Z2)
end
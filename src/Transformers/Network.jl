function propagate!(N :: GeminiNetwork, P :: PropState)
    for diff_layer in get_layers(N)
        # @debug "Propagating layer: $(typeof(diff_layer))"
        input_positions = get_inputs(diff_layer)
        # @debug "Input Positions: $input_positions"
        inputs = get_layer_inputs(input_positions, P)
        if first_pass(P)
            configure_first_usage!(diff_layer.layer_idx, inputs)
        end
        if !has_layer(P, diff_layer)
            @assert first_pass(P) "Layer $diff_layer not initialized in PropState! This should only happen during the first pass."
            init_layer!(P, diff_layer, inputs)
        end
        @assert has_layer(P, diff_layer)
        input_zonotopes = get_zonotope.(inputs)
        output_zonotope_ref = get_layer(P, diff_layer)
        if haskey(P.task_bounds.bounds_cache, diff_layer.layer_idx)
            # @debug "Bounds Cache: Found for layer index $(diff_layer.layer_idx)"
            bounds_cache = P.task_bounds.bounds_cache[diff_layer.layer_idx]
        else
            # @debug "Bounds Cache: Creating new for layer index $(diff_layer.layer_idx)"
            bounds_cache = BoundsCache()
            P.task_bounds.bounds_cache[diff_layer.layer_idx] = bounds_cache
        end
        # @debug "Processing DiffLayer at index $(diff_layer.layer_idx) with bounds cache initialized=$(bounds_cache.initialized)"
        propagate_layer!(output_zonotope_ref, diff_layer, input_zonotopes; bounds_cache=bounds_cache)
        #@debug "Layer $(diff_layer.layer_idx) output Zonotope ∂Z bounds: $(zono_bounds(output_zonotope_ref.zonotope.∂Z)[1:5,:])"
        #@debug "Layer $(diff_layer.layer_idx) output Zonotope Z₁ bounds: $(zono_bounds(output_zonotope_ref.zonotope.Z₁)[1:5,:])"
    end
    return P
end
function propagate!(N :: GeminiNetwork, P :: PropState)
    for diff_layer in get_layers(N)
        @debug "Propagating layer: $(typeof(diff_layer))"
        inputs = get_layer_inputs(get_inputs(diff_layer), P)
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
        propagate_layer!(output_zonotope_ref, diff_layer, input_zonotopes)
    end
    return P
end
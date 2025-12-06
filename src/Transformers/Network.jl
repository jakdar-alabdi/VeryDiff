function propagate(N :: GeminiNetwork, P :: PropState)
    for diff_layer in get_layers(N)
        inputs = get_layer_inputs(get_inputs(diff_layer), P)
        if first_pass(P)
            configure_first_usage!(diff_layer.layer_idx, inputs)
        end
        if !has_layer(P, diff_layer)
            @assert first_pass(P) "Layer $diff_layer not initialized in PropState! This should only happen during the first pass."
            init_layer!(P, diff_layer, inputs)
        end
        @assert has_layer(P, diff_layer)
        propagate_layer!(diff_layer, P)
    end
    return P
end
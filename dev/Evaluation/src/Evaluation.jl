module Evaluation
using StatsPlots
using DataFrames
using CSV
using LaTeXStrings

function δ_bound_improvement(initial_δ_bound::Float64, final_δ_bound::Float64, target::Float64, status::String)
    if status != "UNKNOWN"
        return 1.0
    end
    if isinf(initial_δ_bound) || isinf(final_δ_bound)
        return 0.0
    end
    return clamp((initial_δ_bound - final_δ_bound) / (initial_δ_bound - target), 0.0, 1.0)
end

function δ_bound_improvement(df::DataFrame, target::Float64)
    return δ_bound_improvement.(df.initial_δ_bound, df.final_δ_bound, target, String.(df.status))
end

function δ_bound_improvement(target::Float64)
    return (df::DataFrame) -> begin
        return δ_bound_improvement(df, target)
    end
end

export δ_bound_improvement

end # module Evaluation

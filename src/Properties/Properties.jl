module Properties

using LinearAlgebra

using JuMP
using GLPK
using Gurobi

const GRB_ENV = Ref{Any}(nothing)

using ..Debugger

using ..Definitions

import ..VeryDiff: USE_GUROBI

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
    if "VERYDIFF_NO_GUROBI" in keys(ENV)
        global USE_GUROBI = false
    end
    if USE_GUROBI
        GRB_ENV[] = Gurobi.Env()
        GRBsetintparam(GRB_ENV[], "OutputFlag", 0)
        GRBsetintparam(GRB_ENV[], "LogToConsole", 0)
        GRBsetintparam(GRB_ENV[], "Threads", 0)
    else
        @info "Gurobi not used, falling back to GLPK."
    end
    #GRBsetintparam(GRB_ENV[], "Method", 2)
    #       mnist_19_local_21.vnnlib        mnist_18_local_18
    #0 :    0.018826400587219343s/loop      0.03304489948205128s/loop
    #1 :    0.01705984154058722s/loop       0.03352098044717949s/loop
    #2 :    0.020955224224525042s/loop      0.038390683782564106s/loop
end

include("Epsilon.jl")

export epsilon_split_heuristic, get_epsilon_property_naive, get_epsilon_property

include("Top1.jl")

export get_top1_property, top1_configure_split_heuristic

end # module Properties
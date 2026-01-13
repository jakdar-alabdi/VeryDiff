module VeryDiff

#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning
using DataStructures

using GLPK

NEW_HEURISTIC = true
USE_GUROBI = true

USE_DIFFZONO = true

"""If true, then computation corresponding to DeepSplit neuron splitting are conducted during propagations"""
DEEPSPLIT_NEURON_SPLITTING = Ref{Bool}(false)

"""Use the alternative DeepSplit heuristic for neuron splitting"""
DEEPSPLIT_HUERISTIC_ALTERNATIVE = Ref{Bool}(false)

"""Use the generators of the difference zonotope for the heuristic instead of the corresponding network's zonotope"""
DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS = Ref{Bool}(false)

"""Incorporate DeepSplit input splitting into DeepSplit neuron splitting"""
DEEPSPLIT_INPUT_SPLITTING = Ref{Bool}(true)

"""Constant multiplier used to weight the effect of input nodes in the DeepSplit heuristic"""
INDIRECT_INPUT_MULTIPLIER = Ref{Float64}(2.0)

"""Different modes for the computation of the relative impactes in the DeepSplit heuristic"""
@enum DeepSplitHeuristicMode ZonoBiased ZonoUnbiased DeepSplitBiased DeepSplitUnbiased
DEEPSPLIT_HEURISTIC_MODE = Ref{DeepSplitHeuristicMode}(ZonoBiased)

function set_deepsplit_config(config::Tuple{Bool, Bool, Bool, Bool}; mode=ZonoBiased)
    global DEEPSPLIT_NEURON_SPLITTING = Ref{Bool}(config[1])
    global DEEPSPLIT_HUERISTIC_ALTERNATIVE = Ref{Bool}(config[2])
    global DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS = Ref{Bool}(config[3])
    global DEEPSPLIT_INPUT_SPLITTING = Ref{Bool}(config[4])
    global DEEPSPLIT_HEURISTIC_MODE = Ref{DeepSplitHeuristicMode}(mode)
end

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
    if USE_GUROBI
        GRB_ENV[] = Gurobi.Env()
        GRBsetintparam(GRB_ENV[], "OutputFlag", 0)
        GRBsetintparam(GRB_ENV[], "LogToConsole", 0)
        GRBsetintparam(GRB_ENV[], "Threads", 0)
    end
    #GRBsetintparam(GRB_ENV[], "Method", 2)
    #       mnist_19_local_21.vnnlib        mnist_18_local_18
    #0 :    0.018826400587219343s/loop      0.03304489948205128s/loop
    #1 :    0.01705984154058722s/loop       0.03352098044717949s/loop
    #2 :    0.020955224224525042s/loop      0.038390683782564106s/loop
end

#pinthreads(:cores)

FIRST_ROUND = true

using TimerOutputs
const to = TimerOutput()

using JuMP
#using GLPK
using Gurobi

const GRB_ENV = Ref{Any}(nothing)

include("Debugger.jl")
include("Definitions.jl")
include("Util.jl")
include("Network.jl")
include("Zonotope.jl")
include("Layers_Zonotope.jl")
include("Layers_DiffZonotope.jl")
include("MultiThreadding.jl")
include("Properties.jl")
include("Verifier.jl")
include("Cli.jl")
include("../dev/NeuronSplitting.jl")
include("../dev/DeepSplitHeuristic.jl")
# include("../util.jl")

include("../dev/experiments/acas.jl")
include("../dev/experiments/mnist.jl")
include("../dev/experiments/run.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState, SplitNode, Branch
export zono_optimize, zono_bounds
export verify_network
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic
export propagate_diff_layer
export run_cmd
export deepsplit_lp_search_epsilon, contract_zono
export deepsplit_heuristic
# export run_mnist_all
# export run_acas_all
export run_experiments

end # module AlphaZono

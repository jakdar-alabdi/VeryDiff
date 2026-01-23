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

"""If true, neuron splitting is utilized to refine the bounds of the output Zonotopes"""
global const USE_NEURON_SPLITTING = Ref{Bool}(false)

"""All the approaches used in VeryDiff to split a neuron"""
@enum NeuronSplittingApproach LP ZonoContraction VerticalSplitting
global const NEURON_SPLITTING_APPROACH = Ref{NeuronSplittingApproach}(LP)

"""Use the alternative implementation of the DeepSplit heuristic for neuron splitting"""
global const DEEPSPLIT_HUERISTIC_ALTERNATIVE = Ref{Bool}(false)

"""Use the generators of the Differential Zonotope for the heuristic instead of the corresponding NN's Zonotope"""
global const DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS = Ref{Bool}(false)

"""Incorporate DeepSplit input splitting into DeepSplit neuron splitting"""
global const DEEPSPLIT_INPUT_SPLITTING = Ref{Bool}(true)

"""Constant multiplier used to weight the effect of input nodes in the DeepSplit heuristic"""
global const INDIRECT_INPUT_MULTIPLIER = Ref{Float64}(2.0)

"""Different modes for the computation of the relative impactes in the DeepSplit heuristic"""
@enum DeepSplitHeuristicMode ZonoBiased ZonoUnbiased DeepSplitBiased DeepSplitUnbiased
global const DEEPSPLIT_HEURISTIC_MODE = Ref{DeepSplitHeuristicMode}(ZonoBiased)

function set_neuron_splitting_config(config::Tuple{Bool, Bool, Bool, Bool}; mode=ZonoBiased, approach=LP)
    global USE_NEURON_SPLITTING[] = config[1]
    global DEEPSPLIT_HUERISTIC_ALTERNATIVE[] = config[2]
    global DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS[] = config[3]
    global DEEPSPLIT_INPUT_SPLITTING[] = config[4]
    global DEEPSPLIT_HEURISTIC_MODE[] = mode
    global NEURON_SPLITTING_APPROACH[] = approach
end

function get_config()
    if !USE_NEURON_SPLITTING[]
        return "VeryDiff"
    end
    config = "$(NEURON_SPLITTING_APPROACH[])-$(DEEPSPLIT_HEURISTIC_MODE[])"
    config *= ifelse(DEEPSPLIT_HUERISTIC_ALTERNATIVE[], "-Alt", "-Base")
    config *= ifelse(DEEPSPLIT_INPUT_SPLITTING[], "-Input", "")
    config *= ifelse(DEEPSPLIT_HUERISTIC_USE_DIFF_GENERATORS[], "-DiffZono", "")
    return config
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
include("../dev/ZonoContraction.jl")
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
export deepsplit_lp_search_epsilon
export contract_zono, geometric_distance, transform_offset_zono
export deepsplit_heuristic
# export run_mnist_all
# export run_acas_all
export run_experiments

end # module AlphaZono

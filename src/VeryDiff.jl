module VeryDiff

#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning

@enum VerificationStatus UNKNOWN SAFE UNSAFE

const NEW_HEURISTIC = Ref{Bool}(true)

const USE_DIFFZONO = Ref{Bool}(true)

function __init__()
    BLAS.set_num_threads(1)
end

#pinthreads(:cores)

const FIRST_ROUND = Ref{Bool}(true)

"""If true, neuron splitting is utilized to refine the bounds of the output Zonotopes"""
global const USE_NEURON_SPLITTING = Ref{Bool}(false)

"""All the approaches used in VeryDiff to split a neuron"""
@enum NeuronSplittingApproach LP ZonoContraction VerticalSplitting

"""Use the generators of the Differential Zonotope for the heuristic instead of the corresponding NN's Zonotope"""
global const USE_DIFF_GENERATORS_DEEPSPLIT = Ref{Bool}(false)

"""Incorporate DeepSplit input splitting into DeepSplit neuron splitting"""
global const INCORPORATE_INPUT_SPLITTING = Ref{Bool}(true)

"""Constant multiplier used to weight the effect of input nodes in the DeepSplit heuristic"""
global const INDIRECT_INPUT_MULTIPLIER = Ref{Float64}(2.0)

"""Different modes for the computation of the relative impactes in the DeepSplit heuristic"""
@enum DeepSplitHeuristicApproach ZonoBiased ZonoUnbiased DeepSplitBiased DeepSplitUnbiased
global const DEEPSPLIT_HEURISTIC_APPROACH = Ref{DeepSplitHeuristicApproach}(ZonoBiased)

"""Different modes for the contraction of Zonotopes"""
@enum ZonoContractApproach ZonoContract ZonoContractPre ZonoContractPost ZonoContractInter LPZonoContract

global const USE_LP = Ref{Bool}(true)
global const USE_ZONO_CONTRACT = Ref{Bool}(false)
global const USE_LP_ZONO_CONTRACT = Ref{Bool}(false)
global const USE_VERTICAL_SPLITTING = Ref{Bool}(false)
global const POST_CONTRACT = Ref{Bool}(true)
global const INTER_CONTRACT = Ref{Bool}(false)
global const PRE_CONTRACT = Ref{Bool}(false)

global const USE_CACHED_BOUNDS_IN_LP = Ref{Bool}(true)

global const USE_ZONO_ROW_SUBST = Ref{Bool}(false)
global const INCORPORATE_CONCRETE_SPLIT_BOUNDS = Ref{Bool}(true)
global const INCORPORATE_RELATIONAL_SPLIT_BOUNDS = Ref{Bool}(false)
global const INCORPORATE_RELATIONAL_DIFF_BOUNDS = Ref{Bool}(false)

function set_config(
    heuristic_config :: Tuple{Bool, Bool, Bool} = (true, false, false),
    split_bounds_config :: Tuple{Bool, Bool, Bool, Bool} = (false, true, false, false),
    use_cached_bounds_in_lp :: Bool = true,
    heuristic_approach :: DeepSplitHeuristicApproach = DeepSplitUnbiased,
    neuron_splitting_approach :: NeuronSplittingApproach = LP,
    zono_contract_approach :: ZonoContractApproach = ZonoContract
)
    global USE_NEURON_SPLITTING[] = heuristic_config[1]
    global USE_DIFF_GENERATORS_DEEPSPLIT[] = heuristic_config[2]
    global INCORPORATE_INPUT_SPLITTING[] = heuristic_config[3]
    global DEEPSPLIT_HEURISTIC_APPROACH[] = heuristic_approach
    global USE_CACHED_BOUNDS_IN_LP[] = use_cached_bounds_in_lp
    global USE_ZONO_ROW_SUBST[] = split_bounds_config[1]
    global INCORPORATE_CONCRETE_SPLIT_BOUNDS[] = split_bounds_config[2]
    global INCORPORATE_RELATIONAL_SPLIT_BOUNDS[] = split_bounds_config[3]
    global INCORPORATE_RELATIONAL_DIFF_BOUNDS[] = split_bounds_config[4]
    global USE_ZONO_CONTRACT[] = USE_NEURON_SPLITTING[] && neuron_splitting_approach == ZonoContraction
    global USE_LP_ZONO_CONTRACT[] = USE_ZONO_CONTRACT[] && zono_contract_approach == LPZonoContract
    global USE_LP[] = USE_LP_ZONO_CONTRACT[] || USE_NEURON_SPLITTING[] && neuron_splitting_approach == LP
    global USE_VERTICAL_SPLITTING[] = USE_NEURON_SPLITTING[] && neuron_splitting_approach == VerticalSplitting
    global INTER_CONTRACT[] = USE_ZONO_CONTRACT[] && (zono_contract_approach in [ZonoContractInter, LPZonoContract])
    global POST_CONTRACT[] = USE_ZONO_CONTRACT[] && (zono_contract_approach in [ZonoContract, ZonoContractPost])
    global PRE_CONTRACT[] = USE_ZONO_CONTRACT[] && (zono_contract_approach in [ZonoContract, ZonoContractPre])
    global NEW_HEURISTIC[] = !USE_NEURON_SPLITTING[]
end

function get_config()
    config = "APT-" # "APT" = Any-Positive-Transformer
    if !USE_NEURON_SPLITTING[]
        config *= "VeryDiff"
    else
        if USE_ZONO_CONTRACT[]
            config *= "ZC"
            if USE_LP_ZONO_CONTRACT[]
                config = "LP-" * config
            elseif INTER_CONTRACT[]
                config *= "-Inter"
            elseif POST_CONTRACT[] && !PRE_CONTRACT[]
                config *= "-Post"
            elseif PRE_CONTRACT[] && !POST_CONTRACT[]
                config *= "-Pre"
            end
        elseif USE_LP[]
            config *= "LP"
        elseif USE_VERTICAL_SPLITTING[]
            config *= "VS"
        end
        if USE_LP[] && USE_CACHED_BOUNDS_IN_LP[]
            config *= "-CB"
        end
        if DEEPSPLIT_HEURISTIC_APPROACH[] == ZonoBiased
            config *= "-ZB"
        elseif DEEPSPLIT_HEURISTIC_APPROACH[] == ZonoUnbiased
            config *= "-ZU"
        elseif DEEPSPLIT_HEURISTIC_APPROACH[] == DeepSplitBiased
            config *= "-DB"
        else
            config *= "-DU"
        end
        if INCORPORATE_INPUT_SPLITTING[]
            config *= "-Input"
        end
        if USE_DIFF_GENERATORS_DEEPSPLIT[]
            config *= "-DiffZono"
        end
    end
    if USE_ZONO_ROW_SUBST[]
       config *= "-RS"
    end
    if INCORPORATE_CONCRETE_SPLIT_BOUNDS[]
        config *= "-CSB"
    end
    if INCORPORATE_RELATIONAL_SPLIT_BOUNDS[]
        config *= "-RSB"
    end
    if INCORPORATE_RELATIONAL_DIFF_BOUNDS[]
        config *= "-RDB"
    end
    return config
end

include("Util/simd_bool.jl")
include("Debugger/Debugger.jl")
include("Definitions/Definitions.jl")
using .Definitions

include("Transformers/Transformers.jl")
using .Transformers

include("MultiThreadding.jl")

include("Properties/Properties.jl")
using .Properties

include("Verifier.jl")
include("Cli.jl")

include("../dev/NeuronSplitting.jl")
include("../dev/ZonoContraction.jl")
include("../dev/DeepSplitHeuristic.jl")

include("../dev/experiments/acas.jl")
include("../dev/experiments/mnist.jl")
include("../dev/experiments/lhc.jl")
include("../dev/experiments/run.jl")
# include("../dev/testing/fuzzing.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState
export zono_optimize, zono_bounds
export verify_network
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic

export deepsplit_verify_network
export contract_zono!, contract_zono_all!, transform_offset_diff_zono!, 
contract_to_verification_task!, transform_verification_task!, sort_split_nodes!, 
is_unit_hypercube, transform_constraints!
export deepsplit_heuristic

end # module AlphaZono

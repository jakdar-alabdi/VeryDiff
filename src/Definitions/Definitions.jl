module Definitions

using LinearAlgebra
using Statistics

using VNNLib
using VNNLib.OnnxParser: Node, ONNXLinear, ONNXRelu

using VeryDiff

include("Network.jl")
include("SortedVector.jl")
include("AbstractDomains.jl")
include("DataStructures.jl")
include("PropState.jl")
include("Zonotope.jl")

Dense = ONNXLinear
ReLU = ONNXRelu

export Network,GeminiNetwork,Layer,ZeroDense,Dense,ReLU,DiffLayer, get_input_indices, get_layer_inputs
export Zonotope,DiffZonotope,BoundsCache,CachedZonotope,ZonotopeStorage
export VerificationTask, PropState, reset_ps!, first_pass, get_zonotope, get_layer, get_zonotope!, get_free_generator_id!
export SortedVector, union, intersect_indices, find_index_position, attempt_find_index_position
export parse_network, get_layers, get_inputs, get_layer1, get_diff_layer, get_layer2
export configure_first_usage!, prepare_prop_state!, has_layer
export updateGenerators!, updateGeneratorsMul!, updateGeneratorsAdd!, updateGeneratorsAddMul!, updateGeneratorsSub!, updateGeneratorsSubMul!
export zono_optimize, zono_bounds, zono_get_max_vector

end
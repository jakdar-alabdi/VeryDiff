module Definitions

using LinearAlgebra

using VNNLib.NNLoader: Network,Dense,ReLU,Layer

import ..VeryDiff: NEW_HEURISTIC, to

include("Network.jl")
include("SortedVector.jl")
include("AbstractDomains.jl")
include("DataStructures.jl")
include("PropState.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,ZeroDense, DiffLayer, get_input_indices, get_layer_inputs
export Zonotope,DiffZonotope,BoundsCache,CachedZonotope,ZonotopeStorage
export VerificationTask, PropState, reset_ps!, first_pass, get_zonotope, get_layer, get_zonotope!
export SortedVector, union, intersect_indices
export parse_network, get_layers, get_inputs, get_layer1, get_diff_layer, get_layer2
export configure_first_usage!, prepare_prop_state!, has_layer

end
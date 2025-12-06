module Definitions

using VNNLib.NNLoader: Network,Dense,ReLU

include("Network.jl")
include("SortedVector.jl")
include("AbstractDomains.jl")
include("DataStructures.jl")
include("PropState.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,ZeroDense, DiffLayer, get_input_indices
export Zonotope,DiffZonotope,BoundsCache,CachedZonotope,ZonotopeStorage
export VerificationTask, PropState, reset!
export SortedVector
export parse_network, get_layers, get_inputs, get_layer1, get_diff_layer, get_layer2
export configure_first_usage!, prepare_prop_state!, has_layer

end
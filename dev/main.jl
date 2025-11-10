using VeryDiff
using LinearAlgebra
using JuMP, Gurobi

VeryDiff.NEW_HEURISTIC = false

eps = 0.1
num_layer = rand(10:42)
input_dim = rand(10:42)
output_dim = 10
layers1 = Layer[]
layers2 = Layer[]

next_dim = input_dim
for l in 1:num_layer
    if l == num_layer
        new_out_dim = output_dim
    else
        new_out_dim = rand(10:42)
    end

    W1 = randn(Float64, (new_out_dim, next_dim))
    c1 = randn(Float64, new_out_dim)
    push!(layers1, Dense(W1, c1))
    push!(layers1, ReLU())

    W2 = W1 + eps * randn(Float64, (new_out_dim, next_dim))
    c2 = c1 + eps * randn(Float64, new_out_dim)
    push!(layers2, Dense(W2, c2))
    push!(layers2, ReLU())

    next_dim = new_out_dim
end

N1 = Network(layers1)
N2 = Network(layers2)
N = GeminiNetwork(N1, N2)

input_range = max(eps, 3.0 * rand())

Zin = Zonotope(Matrix(input_range * I, input_dim, input_dim), rand(input_dim), nothing)
prop_state = PropState(true)

L1 = N.network1.layers
L2 = N.network2.layers
∂L = N.diff_network.layers

∂G = Matrix(1.0I, 2, 3)
mul!((@view ∂G[:, 1:2]), [2.0 0; 0.0 2], [2.0 0; 0.0 2])

Z = Zonotope(Matrix(input_range * I, input_dim, input_dim), rand(input_dim), nothing)

bounds = zono_bounds(Z)
lower = bounds[:, 1]
upper = bounds[:, 2]

split_index = argmax((i -> ifelse(lower[i] < 0 < upper[i], upper[i] - lower[i], 0)), 1:size(bounds, 1))
model = JuMP.Model(() -> Gurobi.Optimizer(VeryDiff.GRB_ENV[]))
set_time_limit_sec(model, 10)

g = copy(Z.G[split_index, :])
c = copy(Z.c[split_index])
var_num = size(Z.G, 2)

@variable(model, -1.0 <= x[1:var_num] <= 1.0)
@constraint(model, g ⋅ x + c <= 0.0)
@objective(model, Max, x)


optimize!(model)

objective_value(model)

# println("Verify epsilon property:")
# property = get_epsilon_property(eps)
# input_split_heuristic = epsilon_split_heuristic


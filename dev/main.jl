using VeryDiff

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

println("Verify epsilon property:")
property = get_epsilon_property(eps)
input_split_heuristic = epsilon_split_heuristic


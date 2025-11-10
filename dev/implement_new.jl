using VeryDiff
using LinearAlgebra

Z = Zonotope([-1.0 0.0; 0.5 1.0], zeros(2), nothing)

row_count = size(Z.G, 1)
bounds = zono_bounds(Z)
lower = bounds[:, 1]
upper = bounds[:, 2]

α = clamp.(upper./(upper.-lower), 0.0, 1.0)
λ = ifelse.(upper.<=0.0, 0.0, ifelse.(lower.>=0.0, 1.0, α))
crossing = lower.<0.0 .&& upper.>0.0
γ = 0.5 .* max.(-λ .* lower, 0.0, (1.0 .- λ) .* upper)

ĉ = λ .* Z.c + crossing .* γ

Ĝ = zeros(Float64, row_count, size(Z.G, 2) + count(crossing))
Ĝ[:, 1:size(Z.G, 2)] .= Z.G
Ĝ[crossing, size(Z.G, 2)+1:end] .= (@view I(row_count)[crossing, crossing])
Ĝ[:, 1:size(Z.G, 2)] .*= λ
Ĝ[:, size(Z.G, 2)+1:end] .*= abs.(γ)

Ẑ = Zonotope(Ĝ, ĉ, nothing)


# println(count(crossing))
# println(size(Ĝ, 2))
# println(Ẑ)
# @view I(row_count)[crossing, crossing]
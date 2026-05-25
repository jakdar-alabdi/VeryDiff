using Random
using BenchmarkTools

unit_box = [[-ones(n) ones(n)] for n in [rand(50:100) for _ in 1:4]]
unit_lowers = [bounds[:, 1] for bounds in unit_box]
unit_uppers = [bounds[:, 2] for bounds in unit_box]

perturbated_box = [randn()]

println("---------------")
@btime begin
    @assert all(l -> all(x -> isone(-x), l), unit_lowers) && all(u -> all(x -> isone(x), u), unit_uppers)
end
@btime begin
    @assert mapreduce((l, u) -> all(x -> isone(-x), l) && all(x -> isone(x), u), &, unit_lowers, unit_uppers)
end
# @btime begin
#     d_lower, d_upper = copy(c), copy(c)
#     for G in Gs
#         d_lower .+= sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2)
#         d_upper .+= sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2)
#     end
# end
println("---------------")
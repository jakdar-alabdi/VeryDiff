using Random
using BenchmarkTools

n, m = 784, 10
Gs = [rand(m, n) for _ in 1:4]
c = rand(m)

println("---------------")
@btime begin 
    d_lower = sum(sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2) for G in Gs) + c
    d_upper = sum(sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2) for G in Gs) + c
end
@btime begin
    d_lower = mapreduce(G -> sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2), +, Gs) + c
    d_upper = mapreduce(G -> sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2), +, Gs) + c
end
@btime begin
    d_lower, d_upper = copy(c), copy(c)
    for G in Gs
        d_lower .+= sum(x -> ifelse(x < 0.0, x, 0.0), G, dims=2)
        d_upper .+= sum(x -> ifelse(x > 0.0, x, 0.0), G, dims=2)
    end
end
println("---------------")
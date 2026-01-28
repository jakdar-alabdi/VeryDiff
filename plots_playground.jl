using Pkg
Pkg.add("Plots", io=devnull)

using Plots

mem_consums = Float64[]
open("runlim.log", "r") do f
    while !eof(f)
        line = readline(f)
        if contains(line, "[runlim] sample:")
            attrs = split(line, ",")
            push!(mem_consums, parse(Int64, replace(attrs[end - 1], "MB" => "", " " => "")) / (1 << 10))
        end
    end    
end

plot(dpi=500)
xlabel!("Runlim Sample")
ylabel!("Memory (GB)")

plot!(1:size(mem_consums, 1), mem_consums, label=nothing)

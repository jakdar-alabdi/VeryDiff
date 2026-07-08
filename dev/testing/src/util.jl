using VeryDiff, Plots

# function VeryDiff.Zonotope(low::AbstractVector{N}, high::AbstractVector{N}) where N<:Number
#     @assert length(low) == length(high)
#     mid = (low .+ high) ./ 2
#     distance = (high .- low) ./ 2
#     input_dim = length(low)
#     G = distance .* Matrix(I, input_dim, input_dim)[:, distance .> 0]
#     influence = Matrix(1.0I, sum(distance .> 0), sum(distance .> 0))
#     generators_id = SortedVector{Int}()
#     push!(generators_id, 1)
#     return Zonotope([G], mid, influence, generators_id, 1)
# end

# function VeryDiff.Zonotope(G::AbstractMatrix{N}, c::AbstractVector{N}) where N<:Number
#     m, n = size(G)
#     @assert length(c) == m "Generator matrix should have as many rows as entries in center vector!"
#     influence = Matrix(1.0I, m, m)
#     generators_id = SortedVector{Int}()
#     push!(generators_id, 1)
#     return Zonotope([G], c, influence, generators_id, 1)
# end

# function sample_zono(z::Zonotope; n_samples=100)
#     return sum(G * 2 .* rand(size(G, 2), n_samples) for G in z.Gs) .+ z.c
# end

# function sample_nn(net, z; max_layer=typemax(Int), n_samples=100)
#     max_layer = min(length(net.layers), max_layer)
#     nn = Network(net.layers[1:max_layer])

#     x = sample_zono(z, n_samples=n_samples)
   
#     y = hcat([nn(Vector(xi)) for xi in eachcol(x)]...)
# end

function box_to_zono(box::Matrix{Float64})
    α = (box[:, 2] - box[:, 1]) ./ 2
    β = (box[:, 2] + box[:, 1]) ./ 2
    ids = VeryDiff.SortedVector{Int}()
    push!(ids, 1)
    return Zonotope([Matrix(Diagonal(α))], β, nothing, ids, 1)
end

function box_to_inputbox(box::Matrix{Float64})
    Z = box_to_zono(box)
    VeryDiff.InputBox(Z)
end

function inputbox_to_box(box::VeryDiff.InputBox)
    return [reduce(vcat, box.lowers) reduce(vcat, box.uppers)]
end

function inputbox_to_zono(box::VeryDiff.InputBox)
    return box_to_zono(inputbox_to_box(box))
end

function to_new_zono(G::Matrix{Float64}, c::Vector{Float64})
    ids = VeryDiff.SortedVector{Int}()
    push!(ids, 1)
    return Zonotope([G], c, nothing, ids, 1)
end

"""
Returns a closed list of the boundary vertices of a 2D zonotope.

Instead of trying all 2ⁿ combinations of error-terms, we stack together the
Generators sorted by their angle with the positive x-axis to trace the boundary
of the zonotope.
(Can't plot zonotopes with large number of generators via LazySets otherwise)

Algorithm taken from
-  https://github.com/JuliaReach/LazySets.jl/pull/2288 (LazySets issue about vertices list of 2D zonotopes) and
- https://github.com/TUMcps/CORA/blob/master/contSet/%40zonotope/polygon.m (CORA implementation for zonotope to polygon in MATLAB)
"""
function vertices_list_2d_zonotope(z::Zonotope)
    c = z.c
    G = reduce(hcat, z.Gs)
    d, n = size(G)
    @assert d == 2 string("Only plot 2-D zonotopes!")

    # maximum in x and y direction (assuming 0-center)
    x_max = sum(abs.(G[1,:]))
    y_max = sum(abs.(G[2,:]))

    # make all generators pointing up
    Gnorm = copy(G)
    Gnorm[:, G[2,:] .< 0] .= -1 .* G[:, G[2,:] .< 0]

    # sort generators according to angle to the positive x-axis
    θ = atan.(Gnorm[2,:], Gnorm[1,:])
    θ[θ .< 0] .+= 2*π
    Gsort = Gnorm[:, sortperm(θ)]

    # get boundary of zonotope by stacking the generators together
    # first the generators pointing the most right, then up then left.
    ps = zeros(2, n+1)
    for i in 1:n
        ps[:, i+1] = ps[:, i] + 2*Gsort[:, i]
    end

    ps[1,:] .= ps[1,:] .+ x_max .- maximum(ps[1,:])
    ps[2,:] .= ps[2,:] .- y_max

    # since zonotope is centrally symmetric, we can get the left half of the
    # zonotope by mirroring the right half
    ps = [ps ps[:,end] .+ ps[:,1] .- ps[:,2:end]]

    # translate by the center of the zonotope
    ps .+= c
    return ps
end

"""
Plots a sparse polynomial by plotting zonotope-overapproximations refined by
iteratively splitting the largest generators up to a certain splitting depth.

Overrides the usual plot() function by redirecting it to plot_zono
-> to use it just call plot(z)
"""
@recipe function plot_zono(z::Zonotope)
    label --> get(plotattributes, :label, nothing)
    seriesalpha --> get(plotattributes, :seriesalpha, nothing)
    c_series = get(plotattributes, :seriescolor, nothing)
    c_line = get(plotattributes, :linecolor, nothing)
    if !isnothing(c_line)
        linecolor --> c_line
    elseif !isnothing(c_series)
        linecolor --> c_series
    end

    ps = vertices_list_2d_zonotope(z)
    x = ps[1,:]
    y = ps[2,:]

    seriestype --> :shape
    @series x, y
end

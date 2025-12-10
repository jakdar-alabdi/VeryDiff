# direction: 1 (maximize) or -1 (minimize)
function zono_optimize(direction::Float64, Z::Zonotope, d :: Int) :: Float64
    @assert isone(direction) || isone(-direction)
    row = view(Z.G,d,:)
    result = direction*sum(abs,row) + Z.c[d]
    return result
end

function zono_bounds(Z::Zonotope)
    #return @timeit to "Zonotope_Bounds" begin
    if length(Z.Gs) > 0
        b = sum(abs,Z.Gs[1];dims=2)
        for i in 2:length(Z.Gs)
            b .+= sum(abs,Z.Gs[i];dims=2)
        end
        return [Z.c.-b b.+Z.c]
    else
        return [Z.c Z.c]
    end
    #end
end

function zono_get_max_vector(Z::Zonotope, direction::Vector{Float64})
    weights = direction' * Z.G
    return -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
end

function zono_get_max_vector(Z::Zonotope, d)
    weights = Z.G[d,:]
    return -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
end

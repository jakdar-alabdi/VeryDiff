macro generatorUpdateLoop(op, outGs, indices, inGs, rows, factor=:(nothing))
    colVar = gensym(:cols)
    assigned_expr = :(@view $(esc(inGs))[i][$(esc(rows)), 1:$colVar])
    if factor !== :(nothing)
        assigned_expr = :($(esc(factor)) .* $assigned_expr)
    end
    # Extract the symbol from op if it's a QuoteNode
    op_symbol = op isa QuoteNode ? op.value : op
    return quote
        @inbounds for (i, idx) in enumerate($(esc(indices)))
            $colVar = size($(esc(inGs))[i], 2)
            $(Expr(
                op_symbol, 
                :($(esc(outGs))[idx][$(esc(rows)), 1:$colVar]), 
                assigned_expr
            ))
        end
    end
end

@inline function updateGenerators!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, rows :: BitVector)
    @generatorUpdateLoop(:(.=), outGs, indices, inGs, rows)
end
@inline function updateGeneratorsMul!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, muls :: Union{Float64,Vector{Float64}}, rows :: Union{BitVector,Colon})
    @generatorUpdateLoop(:(.=), outGs, indices, inGs, rows, muls)
end
@inline function updateGeneratorsAdd!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, rows :: BitVector)
    @generatorUpdateLoop(:(.+=), outGs, indices, inGs, rows)
end
@inline function updateGeneratorsAddMul!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, muls :: Union{Float64,Vector{Float64}}, rows :: BitVector)
    @generatorUpdateLoop(:(.+=), outGs, indices, inGs, rows, muls)
end
@inline function updateGeneratorsSub!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, rows :: BitVector)
    @generatorUpdateLoop(:(.-=), outGs, indices, inGs, rows)
end
@inline function updateGeneratorsSubMul!(outGs :: Vector{<:AbstractMatrix{Float64}}, indices :: SortedVector{Int}, inGs::Vector{<:AbstractMatrix{Float64}}, muls :: Union{Float64,Vector{Float64}}, rows :: BitVector)
    @generatorUpdateLoop(:(.-=), outGs, indices, inGs, rows, muls)
end

# direction: 1 (maximize) or -1 (minimize)
function zono_optimize(direction::Float64, Z::Zonotope, d :: Int) :: Float64
    @assert isone(direction) || isone(-direction)
    result = Z.c[d]
    for gidx in eachindex(Z.Gs)
        result += direction * sum(abs,@view Z.Gs[gidx][d, :])
    end
    return result
end

function zono_bounds(Z::Zonotope)
    if isempty(Z.Gs)
        return [Z.c Z.c]
    end
    
    n = size(Z.Gs[1], 1)
    b = zeros(n)
    
    @inbounds for G in Z.Gs
        m = size(G, 2)
        for i in 1:n
            b[i] += BLAS.asum(m, pointer(G,i), n)
        end
    end
    
    return [Z.c .- b Z.c .+ b]
end

function zono_get_max_vector(Z::Zonotope, direction::Vector{Float64})
    output_vector = Vector{Vector{Float64}}(undef, length(Z.Gs))
    for i in eachindex(Z.Gs)
        weights = direction' * Z.Gs[i]
        output_vector[i] = -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
    end
    return output_vector
end

function zono_get_max_vector(Z::Zonotope, d)
    output_vector = Vector{Vector{Float64}}(undef, length(Z.Gs))
    for i in eachindex(Z.Gs)
        weights = Z.Gs[i][d,:]
        output_vector[i] = -1.0*(weights .< 0.0) + 1.0*(weights .>= 0.0)
    end
    return output_vector
end
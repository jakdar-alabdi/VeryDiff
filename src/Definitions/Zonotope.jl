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
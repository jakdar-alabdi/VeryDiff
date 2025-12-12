"""
    @simd_bool_expr N expr

Create a fresh BitVector of length `N` and compute `expr` elementwise
using an allocation-free SIMD loop.

All array references in `expr` are automatically replaced with
`A[i]`, and Boolean operators are used as-is (`&`, `|`, `!`, <, >, etc).
"""
macro simd_bool_expr(N, expr)
    idx = gensym(:i)
    # recursively rewrite any variable `x` into `x[idx]`
    # except literal constants
    function rewrite(e)
        if e isa Symbol
            return :( $e[$idx] )
        elseif e isa Number || e isa String || e === nothing
            return e
        elseif e isa Expr
            # recursively rewrite all arguments
            if e.head == :call
                return Expr(e.head, [e.args[1]; map(rewrite, e.args[2:end])...]...)
            else
                throw(ArgumentError("Unsupported expression head: $(e.head) in expression $(e)"))
            end
        else
            return e
        end
    end

    loop_expr = rewrite(expr)

    quote
        local out = BitVector(undef, $N)
        @inbounds @simd for $idx = 1:$N
            out[$idx] = $loop_expr
        end
        out
    end |> esc
end

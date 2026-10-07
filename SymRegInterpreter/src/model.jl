
# The abstract model type. Concrete subtypes:
#   `ExprModel{T}` — the model is stored as an expression `(x,p) -> body`; the
#                    matching linear code is generated lazily on the first
#                    `code(m)` call (and cached).
#   `CodeModel{T}` — the model is stored directly as `Vector{Instruction{T}}`;
#                    no expression is ever constructed.
# Both expose the same interface: `code(m)`, `n_param(m)`;
# `ExprModel` additionally exposes `expr(m)` / `core_expr(m)`.
"""
    Model{T}

A symbolic regression model `f(x, p)` with element type `T`, with the concrete
subtypes [`ExprModel`](@ref) and [`CodeModel`](@ref).  The constructors choose
the subtype from their argument:

    Model(T, expr::Expr, n_param)             -> ExprModel{T}
    Model(T, expr::Expr, core_expr::Expr, n_param) -> ExprModel{T}
    Model(code::Vector{Instruction{T}}, n_param)   -> CodeModel{T}

`expr` is a function expression `(x, p) -> body` in which `x[i]` is the `i`-th
input variable and `p[j]` the `j`-th of the `n_param` parameters, e.g.
`Model(Float64, :((x, p) -> p[1] * exp(p[2] * x[1])), 2)`.
"""
abstract type Model{T} end

"""
    ExprModel{T}

Model backed by an expression `(x, p) -> body`, plus its `core_expr`
(the original expression used for function complexity and display; `expr` may
be the log-space transformed variant when fitting in log-space, e.g. for RAR).
The code matching `expr` is *lazily* generated on the first `code(m)` call and
cached; the cached code vector must be treated as immutable (structural edits
must build a fresh model, e.g. via `CodeModel`).
"""
mutable struct ExprModel{T} <: Model{T}
    code_cache::Union{Nothing,Vector{Instruction{T}}} # lazily generated code for expr
    expr::Expr # (transformed) expr matching the code, transformation is necessary when we fit in log-space (e.g. for RAR)
    core_expr::Expr  # core expression used for function complexity and display
    n_param::Int

    function ExprModel(::Type{T}, expr::Expr, n_param::Integer) where {T}
        @assert expr.head == :-> "Model expression must be a function of the form (x,p) -> body"
        clean = deepcopy(expr); Base.remove_linenums!(clean)
        new{T}(nothing, clean, clean, Int(n_param))
    end

    function ExprModel(::Type{T}, expr::Expr, core_expr::Expr, n_param::Integer) where {T}
        @assert expr.head == :-> "Model expression must be a function of the form (x,p) -> body"
        clean = deepcopy(expr); Base.remove_linenums!(clean)
        clean_core = deepcopy(core_expr); Base.remove_linenums!(clean_core)
        new{T}(nothing, clean, clean_core, Int(n_param))
    end
end

"""
    CodeModel{T}

Model backed directly by a linear program `Vector{Instruction{T}}`; no
expression is stored or constructed.
"""
struct CodeModel{T} <: Model{T}
    code::Vector{Instruction{T}} # code
    n_param::Int

    function CodeModel(code::Vector{Instruction{T}}, n_param::Integer) where {T}
        m = new{T}(code, Int(n_param))
        _validate_code(code, Int(n_param))
        m
    end
end

# Convenience constructors: an expression yields an `ExprModel`, a code vector
# a `CodeModel`.
Model(code::Vector{Instruction{T}}, n_param::Integer) where {T} = CodeModel(code, n_param)
Model(::Type{T}, code::Vector{Instruction{T}}, n_param::Integer) where {T} = CodeModel(code, n_param)
Model{T}(code::Vector{Instruction{T}}, n_param::Integer) where {T} = CodeModel(code, n_param)
Model(::Type{T}, expr::Expr, n_param::Integer) where {T} = ExprModel(T, expr, n_param)
Model(::Type{T}, expr::Expr, core_expr::Expr, n_param::Integer) where {T} = ExprModel(T, expr, core_expr, n_param)

# Shared code invariants (indices, parameter range).
function _validate_code(code::Vector{Instruction{T}}, n_param::Int) where {T}
    length(code) == 0 && throw(ArgumentError("Model code must not be empty"))
    # every function child index must be lower than its parent and in range
    for i in eachindex(code)
        instr = code[i]
        d = degree(instr.opcode)
        if d >= 1 && !(1 <= instr.arg1idx < i)
            throw(ArgumentError("invalid arg1idx $(instr.arg1idx) at instruction $i"))
        end
        if d == 2 && !(1 <= instr.arg2idx < i)
            throw(ArgumentError("invalid arg2idx $(instr.arg2idx) at instruction $i"))
        end
    end
    # parameters must be indexed 1..n_param and all present
    maxp = 0
    for instr in code
        if instr.opcode == PARAM
            instr.arg1idx == 0 && throw(ArgumentError("PARAM instruction with zero index"))
            maxp = max(maxp, Int(instr.arg1idx))
        elseif instr.opcode == SCALEDVAR
            instr.arg2idx == 0 && throw(ArgumentError("SCALEDVAR instruction with zero parameter index"))
            maxp = max(maxp, Int(instr.arg2idx))
        end
    end
    if maxp != n_param
        throw(ArgumentError("model declares n_param=$n_param but parameter indices imply $maxp"))
    end
    nothing
end

"""
    code(m::Model) -> Vector{Instruction{T}}

The linear program of the model.  For `ExprModel` the code is generated from
the stored expression on the first call and cached afterwards (the cached
vector must be treated as immutable); for `CodeModel` the stored code is
returned directly and no expression is ever constructed.
"""
code(m::CodeModel) = m.code
function code(m::ExprModel{T}) where {T}
    cached = m.code_cache
    cached === nothing || return cached
    newcode = _convert_expr_to_code(T, m.expr)
    _validate_code(newcode, n_param(m))
    m.code_cache = newcode
    newcode
end

"`(x,p) -> body` expression backing an `ExprModel` (possibly log-space transformed)."
expr(m::ExprModel) = m.expr

"Original expression of an `ExprModel`, used for function complexity and display."
core_expr(m::ExprModel) = m.core_expr

"""
    n_param(m::Model) -> Int

The number of model parameters of `m`, the length of the parameter vector `p`.
"""
n_param(m::ExprModel) = m.n_param
n_param(m::CodeModel) = m.n_param

# function _numvar(code::Vector{Instruction{T}}) where {T}
#     m = 0
#     for instr in code
#         if instr.opcode == VAR || instr.opcode == SCALEDVAR
#             m = max(m, Int(instr.arg1idx))
#         end
#     end
#     m
# end

# number of input variables used by the model
# numvar(m::Model) = _numvar(code(m))


"""
    to_logspace_y_model(m::Model) -> Model

Output of transformed model is in logspace for y, but input is in original space for x.
f' = log10(abs(f(x,p)))
This is useful for fitting models where the output is expected to be positive.
"""
to_logspace_y_model(m::ExprModel{T}) where {T} = begin @assert m.expr == m.core_expr; ExprModel(T, to_logspace_y_expr(m.expr), core_expr(m), n_param(m)) end
to_logspace_y_model(m::CodeModel{T}) where {T} = CodeModel(_to_logspace_y_code(m.code, T), m.n_param)
function to_logspace_y_expr(expr::Expr)
    @assert expr.head == :-> "Model expression must be a function of the form (x,p) -> body"
    Expr(:->, expr.args[1], :(log10(abs($(expr.args[2])))))
end

"""
    to_logspace_x_model(m::Model) -> Model

Input of transformed model is in logspace for x, but output is in original space for y.
f' = f(10^x,p)
"""
to_logspace_x_model(m::ExprModel{T}) where {T} = begin @assert m.expr == m.core_expr; ExprModel(T, to_logspace_x_expr(m.expr), core_expr(m), n_param(m)) end
to_logspace_x_model(m::CodeModel{T}) where {T} = CodeModel(_to_logspace_x_code(m.code, T), m.n_param)
function to_logspace_x_expr(expr::Expr)
    @assert expr.head == :-> "Model expression must be a function of the form (x,p) -> body"
    paramTup = expr.args[1]
    xSy = paramTup.args[1]
    Expr(:->, paramTup, _replace_x_with_pow10x(expr.args[2], xSy))
end

"""
    to_logspace_xy_model(m::Model) -> Model

Input and output of transformed model is in logspace for x and y.
f' = log10(abs(f(10^x,p)))
"""
to_logspace_xy_model(m::ExprModel{T}) where {T} = begin @assert m.expr == m.core_expr; ExprModel(T, to_logspace_xy_expr(m.expr), core_expr(m), n_param(m)) end
to_logspace_xy_model(m::CodeModel{T}) where {T} = to_logspace_y_model(to_logspace_x_model(m))
to_logspace_xy_expr(expr::Expr) = to_logspace_y_expr(to_logspace_x_expr(expr))

# ---------------------------------------------------------------------------
# Code-level log-space transformations for `CodeModel` (no expression is ever
# constructed).  The emitted instructions mirror exactly what the Expr path
# produces: `log10(abs(f))` appends ABS then LOG10 above the root, and a
# variable read `x[i]` becomes `POW(CONSTANT(10), x[i])`; `SCALEDVAR p[k]*x[i]` becomes
# `p[k] * POW(10, x[i])`.
# ---------------------------------------------------------------------------

# f' = log10(abs(f)): wrap the root in ABS then LOG10.
function _to_logspace_y_code(code::Vector{Instruction{T}}, ::Type{T}) where {T}
    n = length(code)
    newcode = Vector{Instruction{T}}(undef, n + 2)
    copyto!(newcode, code)
    newcode[n + 1] = Instruction{T}(ABS, UInt32(n), UInt32(0), zero(T))
    newcode[n + 2] = Instruction{T}(LOG10, UInt32(n + 1), UInt32(0), zero(T))
    newcode
end

# f' = f(10^x): replace every VAR/SCALEDVAR terminal by its 10^x variant and
# remap the child indices of all other instructions accordingly.
function _to_logspace_x_code(code::Vector{Instruction{T}}, ::Type{T}) where {T}
    newcode = Vector{Instruction{T}}()
    sizehint!(newcode, 3 * length(code))
    newpos = Vector{UInt32}(undef, length(code))
    for i in eachindex(code)
        instr = code[i]
        opc = instr.opcode
        if opc == VAR
            # x[i] -> POW(10, x[i])
            push!(newcode, create_const_instruction(T(10)))
            cpos = UInt32(length(newcode))
            push!(newcode, create_var_instruction(T, Int(instr.arg1idx)))
            vpos = UInt32(length(newcode))
            push!(newcode, Instruction{T}(POW, cpos, vpos, zero(T)))
        elseif opc == SCALEDVAR
            # p[k]*x[i] -> p[k] * POW(10, x[i]); keeps the parameter value
            push!(newcode, create_param_instruction(T, Int(instr.arg2idx); val=instr.val))
            ppos = UInt32(length(newcode))
            push!(newcode, create_const_instruction(T(10)))
            cpos = UInt32(length(newcode))
            push!(newcode, create_var_instruction(T, Int(instr.arg1idx)))
            vpos = UInt32(length(newcode))
            push!(newcode, Instruction{T}(POW, cpos, vpos, zero(T)))
            powpos = UInt32(length(newcode))
            push!(newcode, Instruction{T}(MUL, ppos, powpos, zero(T)))
        else
            d = degree(opc)
            a1 = d >= 1 ? newpos[Int(instr.arg1idx)] : instr.arg1idx
            a2 = d == 2 ? newpos[Int(instr.arg2idx)] : instr.arg2idx
            push!(newcode, Instruction{T}(opc, a1, a2, instr.val))
        end
        newpos[i] = UInt32(length(newcode))
    end
    newcode
end

_replace_x_with_pow10x(val::Number, _) = val
_replace_x_with_pow10x(sy::Symbol, _) = sy
function _replace_x_with_pow10x(expr::Expr, xSy::Symbol)
    op = expr.head
    if op == :ref
        arrSy = expr.args[1]
        if arrSy == xSy
            return Expr(:call, :^, 10, expr)
        else
            return expr
        end
    else
        return Expr(op, map(a -> _replace_x_with_pow10x(a, xSy), expr.args)...)
    end
end


# ---------------------------------------------------------------------------
# Model editing (constant snapping)
# ---------------------------------------------------------------------------

"""
    replace_param_with_const(m::Model, pix::Int, val) -> CodeModel

A new model in which the parameter `p[pix]` is replaced by the constant `val`
and the parameters after it are renumbered, so the result has one parameter
less than `m`.
"""
function replace_param_with_const(m::Model{T}, pix::Int, val) where {T}
    pix < 1 && throw(ArgumentError("parameter index to replace must be >= 1"))
    pix <= n_param(m) || throw(ArgumentError("parameter index $pix out of range (model has $(n_param(m)) parameters)"))

    # Rebuild the code, dropping PARAM/SCALEDVAR references to `pix` and
    # replacing their occurrences with a constant node, reindexing parameters
    # > pix down by one.  A shared intermediate node keeps one position (the
    # DAG is preserved) by emitting on first visit and reusing that position.
    prog = code(m)
    n = length(prog)

    newcode = Instruction{T}[]
    emitted = falses(n)
    emit_pos = Vector{Int}(undef, n)

    # Recursive emit with a worklist implemented as a simple closure.
    function emit!(i::Int)::Int
        if emitted[i]
            return emit_pos[i]
        end
        instr = prog[i]
        opc = instr.opcode
        if opc == PARAM && instr.arg1idx == pix
            # replace with constant; create a fresh node each time (valid, no
            # child arguments).  Mark this specific instruction emitted once.
            emitted[i] = true
            emit_pos[i] = _emit_constant!(newcode, val)
            return emit_pos[i]
        elseif opc == SCALEDVAR && instr.arg2idx == pix
            # replace the parameter with the constant, keeping the * x[idx]
            # factor (a SCALEDVAR computes p[arg2idx] * x[arg1idx]).
            emitted[i] = true
            cpos = _emit_constant!(newcode, val)
            push!(newcode, create_var_instruction(T, Int(instr.arg1idx)))
            vpos = length(newcode)
            push!(newcode, Instruction{T}(MUL, UInt32(cpos), UInt32(vpos), zero(T)))
            emit_pos[i] = length(newcode)
            return emit_pos[i]
        end
        # child args (function, SCALEDVAR handled separately)
        a1 = (degree(opc) >= 1) ? emit!(Int(instr.arg1idx)) : 0
        a2 = (degree(opc) == 2) ? emit!(Int(instr.arg2idx)) : 0
        if opc == PARAM
            # reindex params > pix (params < pix are kept unchanged)
            newp = (instr.arg1idx > pix) ? Int(instr.arg1idx) - 1 : Int(instr.arg1idx)
            push!(newcode, create_param_instruction(T, newp; val=instr.val))
        elseif opc == SCALEDVAR
            newp = (instr.arg2idx > pix) ? Int(instr.arg2idx) - 1 : Int(instr.arg2idx)
            push!(newcode, create_scaledvar_instruction(T, newp, Int(instr.arg1idx); val=instr.val))
        else
            # For functions arg1/arg2 are the recomputed child positions; for
            # terminals (VAR/CONSTANT) they hold indices and are copied verbatim.
            newa1 = (degree(opc) >= 1) ? UInt32(a1) : instr.arg1idx
            newa2 = (degree(opc) == 2) ? UInt32(a2) : instr.arg2idx
            push!(newcode, Instruction{T}(opc, newa1, newa2, instr.val))
        end
        emitted[i] = true
        emit_pos[i] = length(newcode)
        return emit_pos[i]
    end

    emit!(n)
    _nparam = n_param(m) - 1
    # a code-built model never needs an expression
    return CodeModel(newcode, _nparam)
end

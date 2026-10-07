# `code` is a Vector{Instruction} which is a linear representation of a
# directed acyclic graph of expressions.  The code can be evaluated from left
# to right.  Every function child index (`arg1idx`/`arg2idx`) is lower than its
# parent index, and the final instruction is the root of the program.

"""
    Opcode

The operation of an [`Instruction`](@ref), an `@enum` backed by `UInt8`.
[`terminal_opcodes`](@ref), [`unary_opcodes`](@ref) and [`binary_opcodes`](@ref)
group the opcodes by arity; the docstring of each opcode describes its operation.
"""
@enum Opcode::UInt8 begin
    STOP = 0
    ADD
    SUB
    MUL
    DIV
    INV
    LOG
    LOG10
    LOGABS
    EXP
    POW
    POWCONST
    POWABS
    NEG
    ABS
    SIGN
    SQR
    SQRT
    SQRTABS
    SIN
    SINH
    ASIN
    TAN
    TANH
    COS
    COSH
    CONSTANT
    PARAM
    VAR
    SCALEDVAR
end

# `a` and `b` are the results of the instructions at `arg1idx` and `arg2idx`.
for (opc, doc) in (
        STOP => "Placeholder terminal (code 0) that computes no value.",
        ADD => "Binary `a + b`.",
        SUB => "Binary `a - b`.",
        MUL => "Binary `a * b`.",
        DIV => "Binary `a / b`.",
        INV => "Unary `1 / a`.",
        LOG => "Unary `log(a)`, `NaN` for `a < 0`.",
        LOG10 => "Unary `log10(a)`, `NaN` for `a < 0`.",
        LOGABS => "Unary `log(abs(a))`.",
        EXP => "Unary `exp(a)`.",
        POW => "Binary `a ^ b`, `NaN` for `a < 0` and a non-integer `b`.",
        POWCONST => "Unary `a ^ val` with the constant exponent `val` of the instruction, `NaN` for `a < 0` and a non-integer `val`.",
        POWABS => "Binary `abs(a) ^ b`.",
        NEG => "Unary `-a`.",
        ABS => "Unary `abs(a)`.",
        SIGN => "Unary `sign(a)`.",
        SQR => "Unary `a * a`.",
        SQRT => "Unary `sqrt(a)`, `NaN` for `a < 0`.",
        SQRTABS => "Unary `sqrt(abs(a))`.",
        SIN => "Unary `sin(a)`.",
        SINH => "Unary `sinh(a)`.",
        ASIN => "Unary `asin(a)`.",
        TAN => "Unary `tan(a)`.",
        TANH => "Unary `tanh(a)`.",
        COS => "Unary `cos(a)`.",
        COSH => "Unary `cosh(a)`.",
        CONSTANT => "Terminal holding the constant `val` of the instruction.",
        PARAM => "Terminal reading the model parameter `p[arg1idx]`.",
        VAR => "Terminal reading the input variable `x[:, arg1idx]`.",
        SCALEDVAR => "Terminal `p[arg2idx] * x[:, arg1idx]`, an input variable scaled by a model parameter.")
    @eval @doc $("    $opc\n\n[`Opcode`](@ref): " * doc) $(Symbol(opc))
end

"Terminal opcodes (degree 0): `STOP`, `CONSTANT`, `PARAM`, `VAR` and `SCALEDVAR`."
const terminal_opcodes = [STOP, CONSTANT, PARAM, VAR, SCALEDVAR]
"Unary opcodes (degree 1); their argument is the instruction at `arg1idx`."
const unary_opcodes = [LOG, LOG10, LOGABS, EXP, ABS, SIGN, SIN, SINH, COS, COSH,
                       ASIN, TAN, TANH, POWCONST, NEG, INV, SQR, SQRT, SQRTABS]
"Binary opcodes (degree 2); their arguments are the instructions at `arg1idx` and `arg2idx`."
const binary_opcodes = [ADD, SUB, MUL, DIV, POW, POWABS]

"""
    opcode(sy::Symbol) -> Opcode

The function opcode for the function symbol `sy` of a model expression, e.g.
`opcode(:+) == ADD` or `opcode(:sqrtabs) == SQRTABS`.  Throws an `ArgumentError`
for a symbol without an opcode.
"""
function opcode(sy::Symbol)::Opcode
    if sy == :+       return ADD
    elseif sy == :-   return SUB
    elseif sy == :*   return MUL
    elseif sy == :/   return DIV
    elseif sy == :inv return INV
    elseif sy == :log return LOG
    elseif sy == :log10 return LOG10
    elseif sy == :logabs return LOGABS
    elseif sy == :exp return EXP
    elseif sy == :^   return POW
    elseif sy == :powabs return POWABS
    elseif sy == :abs return ABS
    elseif sy == :sign return SIGN
    elseif sy == :sqr return SQR
    elseif sy == :sqrt return SQRT
    elseif sy == :sqrtabs return SQRTABS
    elseif sy == :sin return SIN
    elseif sy == :sinh return SINH
    elseif sy == :asin return ASIN
    elseif sy == :cos return COS
    elseif sy == :cosh return COSH
    elseif sy == :tan return TAN
    elseif sy == :tanh return TANH
    else throw(ArgumentError("no opcode for symbol $sy"))
    end
end

# Arity of every opcode, indexed by `UInt8(opc) + 1`, with `-1` marking a code
# that is in none of the three lists above.
const opcode_degrees = let
    tbl = fill(Int8(-1), Int(maximum(UInt8, instances(Opcode))) + 1)
    for (d, opcs) in ((0, terminal_opcodes), (1, unary_opcodes), (2, binary_opcodes))
        for opc in opcs
            tbl[Int(UInt8(opc)) + 1] = Int8(d)
        end
    end
    tbl
end

"""
    degree(opc::Opcode) -> Int

The number of arguments of `opc`: 0 for terminals, 1 for unary and 2 for binary
opcodes.
"""
function degree(opc::Opcode)::Int
    d = opcode_degrees[Int(UInt8(opc)) + 1]
    d < 0 && throw(ArgumentError("unknown degree of opcode $opc"))
    Int(d)
end

# The shared instruction.  For functions `arg1idx`/`arg2idx` are child
# instruction indices.  For terminals the dedicated index arguments are reused:
# `VAR` stores the variable index in `arg1idx`; `PARAM` stores the parameter
# index in `arg1idx`; `SCALEDVAR` stores the variable index in `arg1idx` and the
# model-parameter index in `arg2idx`.  For constants `val` holds the value.
# `val` also holds the current optimized value of parameters so
# it becomes heritable genome state.
# Immutable (isbits) value type: structural edits build new Instructions rather
# than mutating fields, so instructions can be freely shared/aliased across
# programs without cross-corruption.
"""
    Instruction{T}(opcode::Opcode, arg1idx, arg2idx, val::T)
    Instruction(opcode::Opcode, arg1idx, arg2idx, val)

One instruction of the linear program of a [`CodeModel`](@ref).  For unary and
binary opcodes `arg1idx` and `arg2idx` are the indices of the argument
instructions, which must precede the instruction (`arg2idx` is 0 for unary
opcodes).  Terminals reuse the index fields: `VAR` reads `x[:, arg1idx]`, `PARAM`
reads `p[arg1idx]` and `SCALEDVAR` computes `p[arg2idx] * x[:, arg1idx]`.  `val`
holds the value of a `CONSTANT`, the exponent of `POWCONST`, and the current
value of a parameter.

Instructions are immutable; see also [`create_const_instruction`](@ref),
[`create_var_instruction`](@ref), [`create_param_instruction`](@ref) and
[`create_scaledvar_instruction`](@ref).
"""
struct Instruction{T}
    opcode::Opcode
    arg1idx::UInt32 # function: first child index; VAR: variable index; PARAM: parameter index; SCALEDVAR: variable index
    arg2idx::UInt32 # function: second child index (0 for unary); SCALEDVAR: model-parameter index; otherwise 0
    val::T          # for constants and current parameter/scale values

    function Instruction{T}(opcode::Opcode, arg1idx::UInt32, arg2idx::UInt32, val::T) where {T}
        new{T}(opcode, arg1idx, arg2idx, val)
    end
end

Instruction(opcode::Opcode, arg1idx::UInt32, arg2idx::UInt32, val::T) where {T} =
    Instruction{T}(opcode, arg1idx, arg2idx, val)

# Convenience for user/test code: infer the element type from `val` and accept
# plain integer indices.
Instruction(opcode::Opcode, arg1idx::Integer, arg2idx::Integer, val) =
    Instruction{typeof(val)}(opcode, UInt32(arg1idx), UInt32(arg2idx), val)

# Convenience constructor accepting plain integer index literals (tests, user
# code) while the parametrized constructor stays on UInt32 for the hot path.
Instruction{T}(opcode::Opcode, arg1idx::Integer, arg2idx::Integer, val::T) where {T} =
    Instruction{T}(opcode, UInt32(arg1idx), UInt32(arg2idx), val)

"""
    create_const_instruction(val::T) -> Instruction{T}

A `CONSTANT` instruction with value `val`.
"""
create_const_instruction(val::T) where {T} = Instruction{T}(CONSTANT, UInt32(0), UInt32(0), val)

"""
    create_var_instruction(T, varidx) -> Instruction{T}

A `VAR` instruction reading the input variable `x[:, varidx]`.
"""
create_var_instruction(::Type{T}, varidx) where {T} = Instruction{T}(VAR, UInt32(varidx), UInt32(0), zero(T))

"""
    create_param_instruction(T, paramidx; val = zero(T)) -> Instruction{T}

A `PARAM` instruction reading the model parameter `p[paramidx]`, whose current
value is `val`.
"""
create_param_instruction(::Type{T}, paramidx; val::T = zero(T)) where {T} = Instruction{T}(PARAM, UInt32(paramidx), UInt32(0), val)

"""
    create_scaledvar_instruction(T, paramidx, varidx; val = zero(T)) -> Instruction{T}

A `SCALEDVAR` instruction computing `p[paramidx] * x[:, varidx]`, whose
parameter has the current value `val`.
"""
create_scaledvar_instruction(::Type{T}, paramidx, varidx; val::T = zero(T)) where {T} =
    Instruction{T}(SCALEDVAR, UInt32(varidx), UInt32(paramidx), val)

function Base.show(io::IO, instr::Instruction)
    Printf.format(io, Printf.format"%15s %3d %3d %f", instr.opcode, instr.arg1idx,
                  instr.arg2idx, instr.val)
end

function Base.show(io::IO, code::AbstractArray{Instruction{T}}) where {T}
    sym = Dict(
        STOP => ".",
        ADD => "+", SUB => "-", NEG => "neg", MUL => "*", DIV => "/",
        INV => "inv", POW => "^", POWABS => "abs^", POWCONST => "^c",
        LOG => "log", LOG10 => "l10", LOGABS => "labs", EXP => "exp",
        ABS => "abs", SIGN => "sgn", SQR => "sqr", SQRT => "sqrt", SQRTABS => "sqrtabs",
        SIN => "sin", COS => "cos", TAN => "tan", TANH => "tanh",
        SINH => "sinh", COSH => "cosh", ASIN => "asin",
        VAR => "var", CONSTANT => "con", PARAM => "par", SCALEDVAR => "svar",
    )
    for i in eachindex(code)
        instr = code[i]
        Printf.format(io, Printf.format"%4d %4s %3d %3d %f", i, sym[instr.opcode],
                      instr.arg1idx, instr.arg2idx, instr.val)
        println(io)
    end
end


"""
    print_program(io::IO, code, varnames = nothing, pos = length(code))

Print the expression computed by instruction `pos` of `code` (by default the
root) in infix notation.  Parameters are printed with their current values and
variables as `X1, X2, …` or by their names in `varnames`.
"""
function print_program(io::IO, code::AbstractVector{Instruction{T}}, varnames=nothing, pos::Int=length(code)) where {T}
    @assert pos > 0 && pos <= length(code) "invalid instruction index $pos"
    instr = code[pos]
    opc = instr.opcode
    if opc == CONSTANT
        print(io, string(instr.val))
    elseif opc == PARAM
        # print the fitted parameter value (not the index), matching NeoGP main
        print(io, string(instr.val))
    elseif opc == VAR
        print(io, isnothing(varnames) ? "X$(instr.arg1idx)" : varnames[instr.arg1idx])
    elseif opc == SCALEDVAR
        print(io, "(", instr.val, " * ", isnothing(varnames) ? "X$(instr.arg1idx)" : varnames[instr.arg1idx], ")")
    elseif degree(opc) == 1
        local a = Int(instr.arg1idx)
        local nm = lowercase(string(opc))
        if nm == "logabs"
            print(io, "log(abs(")
            print_program(io, code, varnames, a)
            print(io, "))")
        elseif nm == "sqrtabs"
            print(io, "sqrt(abs(")
            print_program(io, code, varnames, a)
            print(io, "))")
        elseif nm == "powconst"
            if instr.val == 1/2
                print(io, "sqrt(")
                print_program(io, code, varnames, a)
                print(io, ")")
            else
                print_program(io, code, varnames, a)
                print(io, " ^ $(instr.val)")
            end
        else
            # wrappers
            print(io, nm == "inv" ? "(1.0/(" : "$nm(")
            print_program(io, code, varnames, a)
            print(io, nm == "inv" ? "))" : ")")
        end
    else
        local l = Int(instr.arg1idx); local r = Int(instr.arg2idx)
        if opc == SymRegInterpreter.ADD || opc == SymRegInterpreter.SUB || opc == SymRegInterpreter.MUL ||
           opc == SymRegInterpreter.DIV || opc == SymRegInterpreter.POW
            print(io, "(")
            print_program(io, code, varnames, l)
            print(io, " ", _binary_name(opc), " ")
            print_program(io, code, varnames, r)
            print(io, ")")
        elseif opc == SymRegInterpreter.POWABS
            print(io, "abs(")
            print_program(io, code, varnames, l)
            print(io, ") ^ ")
            print_program(io, code, varnames, r)
        else
            # other functions: f(a,b)
            print(io, string(opc), "(")
            print_program(io, code, varnames, l)
            print(io, ", ")
            print_program(io, code, varnames, r)
            print(io, ")")
        end
    end
    nothing
end

function _binary_name(opc)
    opc == SymRegInterpreter.ADD && return "+"
    opc == SymRegInterpreter.SUB && return "-"
    opc == SymRegInterpreter.MUL && return "*"
    opc == SymRegInterpreter.DIV && return "/"
    opc == SymRegInterpreter.POW && return "^"
    string(opc)
end


# ---------------------------------------------------------------------------
# Expression <-> code conversion
# ---------------------------------------------------------------------------
# These functions are not expected to be called in tight loops. 
function _convert_expr_to_code(::Type{T}, expr::Expr)::Vector{Instruction{T}} where {T}
    # Convert on a private copy so the caller's expression is not mutated (the
    # shared `Model` stores the same expression for model-selection complexity).
    expr = deepcopy(expr)
    code = Vector{Instruction{T}}()

    Base.remove_linenums!(expr)
    paramTup = expr.args[1]
    xSy = paramTup.args[1]
    pSy = paramTup.args[2]
    body = expr.args[2]

    cache = Dict{Any,Int32}() # hashcons cache to de-duplicate subexpressions
    _convert_expr_to_code!(code, cache, body, xSy, pSy)
    code
end

# uses cache (hashcons) to de-duplicate subexpressions in the tree.
function _convert_expr_to_code!(code::Vector{Instruction{T}}, cache, val::TV, xSy, pSy)::UInt32 where {T,TV}
    # constants are also hashconsed to shorten the code if the same constant appears multiple times
    get!(() -> _emit_constant!(code, val), cache, val)
end

function _convert_expr_to_code!(code::Vector{Instruction{T}}, cache, expr::Expr, xSy, pSy)::UInt32 where {T}
    haskey(cache, expr) && return cache[expr]

    is_abs(a) = a isa Expr && a.head == :call && a.args[1] == :abs

    sy = expr.head
    if sy == :call
        func = expr.args[1]
        arg1idx::UInt32 = 0
        arg2idx::UInt32 = 0
        if length(expr.args) == 2
            a1 = expr.args[2]
            # Decide the `sqrt(abs(.))` / `log(abs(.))` fusion *before* converting
            # the argument.  
            # A genuine `abs(.)` elsewhere in the expression is unaffected --
            # hashconsing emits it for that use and the inner conversion here is
            # then a cache hit.
            if (func == :sqrt || func == :log) && is_abs(a1)
                arg1idx = _convert_expr_to_code!(code, cache, a1.args[2], xSy, pSy)
                push!(code, Instruction{T}(func == :sqrt ? SQRTABS : LOGABS, arg1idx, UInt32(0), zero(T)))
            else
                arg1idx = _convert_expr_to_code!(code, cache, a1, xSy, pSy)
                if func == :- push!(code, Instruction{T}(NEG, arg1idx, UInt32(0), zero(T)))
                else          push!(code, Instruction{T}(opcode(func), arg1idx, UInt32(0), zero(T)))
                end
            end
        elseif length(expr.args) == 3
            a1 = expr.args[2]
            # `POWCONST` keeps precedence over the `abs` fusion, and it does read
            # the `ABS`, so it is decided first.  The `POWABS` fusion is decided
            # before the base is converted, for the reason given above.
            powconst = func == :^ && expr.args[3] isa Number && isinteger(expr.args[3])
            if func == :^ && !powconst && is_abs(a1)
                # fuse abs(x)^y --> POWABS(x, y)
                arg1idx = _convert_expr_to_code!(code, cache, a1.args[2], xSy, pSy)
                arg2idx = _convert_expr_to_code!(code, cache, expr.args[3], xSy, pSy)
                push!(code, Instruction{T}(POWABS, arg1idx, arg2idx, zero(T)))
            elseif powconst
                # integer constant power: keep POWCONST with integer exponent
                arg1idx = _convert_expr_to_code!(code, cache, a1, xSy, pSy)
                push!(code, Instruction{T}(POWCONST, arg1idx, UInt32(0), T(expr.args[3])))
            else
                arg1idx = _convert_expr_to_code!(code, cache, a1, xSy, pSy)
                arg2idx = _convert_expr_to_code!(code, cache, expr.args[3], xSy, pSy)
                push!(code, Instruction{T}(opcode(func), arg1idx, arg2idx, zero(T)))
            end
        elseif length(expr.args) > 3 && func in (:+, :-, :*, :/)
            # Julia flattens a+b+c into a single n-ary call. Fold left-to-right
            # (matches Julia's own left-associative evaluation for + - * /).
            opc = opcode(func)
            k = length(expr.args)
            acc = _convert_expr_to_code!(code, cache, expr.args[2], xSy, pSy)
            for j in 3:k
                next = _convert_expr_to_code!(code, cache, expr.args[j], xSy, pSy)
                push!(code, Instruction{T}(opc, acc, next, zero(T)))
                acc = UInt32(length(code))
            end
        else
            throw(ArgumentError("only unary and binary functions are supported ($func with $(length(expr.args) - 1) arguments is not supported)"))
        end
    elseif sy == :ref
        arrSy = expr.args[1]
        idx = expr.args[2]
        if arrSy == xSy     push!(code, create_var_instruction(T, idx))
        elseif arrSy == pSy push!(code, create_param_instruction(T, idx))
        else
            dump(expr)
            throw(UndefVarError("unknown symbol"))
        end
    elseif sy == :block
        # This is a convenience because some expressions taken from the REPL contain blocks
        @assert length(expr.args) == 1
        # A `:block` emits no instruction of its own, so its result is the
        # child's index, not `length(code)`. 
        idx = _convert_expr_to_code!(code, cache, expr.args[1], xSy, pSy)
        cache[expr] = idx
        return idx
    else
        error("Unsupported symbol $sy")
    end

    cache[expr] = length(code)
    length(code)
end

# Appends the constant `val` to `code` and returns its position.  A
# non-integer rational `n//d` is emitted as `DIV(CONSTANT n, CONSTANT d)`, so
# that the code keeps the exact rational (needed by the function complexity)
# instead of its rounded value.
function _emit_constant!(code::Vector{Instruction{T}}, val) where {T}
    if val isa Rational && !isone(denominator(val))
        push!(code, create_const_instruction(T(numerator(val))))
        push!(code, create_const_instruction(T(denominator(val))))
        n = length(code)
        push!(code, Instruction{T}(DIV, UInt32(n - 1), UInt32(n), zero(T)))
    else
        push!(code, create_const_instruction(T(val)))
    end
    length(code)
end


# ---------------------------------------------------------------------------
# Function complexity
# ---------------------------------------------------------------------------
#
# Constants are part of the structural complexity.  Only integer and rational
# constants are allowed; a float constant throws an `ArgumentError`.
#   * an integer `c` costs `log(|c| + constant_offset)`,
#   * a rational `n/d` with `d != 1` costs its numerator like an integer,
#     `log(|n| + constant_offset)`, its denominator `log(d)` (never zero), and
#     one division symbol,
#   * a negative constant (integer or rational) counts one `-` symbol.
# A rational is a `Rational` literal or a division of two integer literals
# with a positive denominator in the `Expr` path, and
# `DIV(CONSTANT n, CONSTANT d)` with `d > 0` in the bytecode path.

# Symbol-frequency counting on the Expr AST.

"""
    _symbol_frequency(expr; count_params = false) -> (Dict{Symbol,Int}, Vector{Float64}, Vector{Float64})

Symbol frequencies of an expression AST. Parameters (`p…` symbols) are not
counted unless `count_params = true`; `x[i]` references are normalized to
`xi` symbols. Integer constants and the numerators of rationals are collected
(as absolute values) in `constants`, the denominators of rationals in
`denominators`.
"""
function _symbol_frequency(expr; count_params = false)
    function remove_refs(expr)
        if expr isa Symbol || expr isa Number
            expr
        elseif expr.head == :ref
            @assert expr.args[1] isa Symbol && expr.args[2] isa Integer
            Symbol(string(expr.args[1], expr.args[2])) # x[1] -> x1, p[1] -> p1
        else
            Expr(expr.head, map(remove_refs, expr.args)...)
        end
    end

    dict = Dict{Symbol,Int}()
    constants = Vector{Float64}()
    denominators = Vector{Float64}()

    _collect_symbols_and_constants!(dict, constants, denominators, remove_refs(expr); count_params = count_params)

    dict, constants, denominators
end

function _collect_symbols_and_constants!(symbolDict, constants, denominators, num::Integer; count_params = false)
    num < 0 && (symbolDict[:-] = 1 + get(symbolDict, :-, 0))
    push!(constants, abs(num))
    nothing
end

function _collect_symbols_and_constants!(symbolDict, constants, denominators, num::Rational; count_params = false)
    # the sign is carried by the numerator
    _collect_symbols_and_constants!(symbolDict, constants, denominators, numerator(num); count_params = count_params)
    isone(denominator(num)) && return nothing
    push!(denominators, denominator(num))
    symbolDict[:/] = 1 + get(symbolDict, :/, 0)
    nothing
end

_collect_symbols_and_constants!(symbolDict, constants, denominators, num::Number; count_params = false) =
    throw(ArgumentError("constant $num is not supported; only integer and rational constants are allowed"))

function _collect_symbols_and_constants!(symbolDict, constants, denominators, sy::Symbol; count_params = false)
    # GPSR-MDL Paper (https://arxiv.org/abs/2605.22374) says we should not count parameters
    is_param(a::Symbol) = string(a)[1] == 'p'
    if !is_param(sy)    symbolDict[sy] = 1 + get(symbolDict, sy, 0)
    elseif count_params symbolDict[:param] = 1 + get(symbolDict, :param, 0)
    end
end

function _collect_symbols_and_constants!(symbolDict, constants, denominators, expr::Expr; count_params = false)
    countSy! = (sy) -> begin
        symbolDict[sy] = 1 + get(symbolDict, sy, 0)
    end

    sy = expr.head
    if sy == :call
        func = expr.args[1]
        # unary functions
        if length(expr.args) == 2
            _collect_symbols_and_constants!(symbolDict, constants, denominators, expr.args[2]; count_params = count_params)
            # From the old code, but using x^(1/2) seems longer than sqrt
            # if func == :sqrt
            #     # x^(1/2)
            #     countSy!(:^)
            #     _collect_symbols_and_constants!(symbolDict, constants, denominators, 1//2; count_params = count_params)
            # else
                countSy!(func)
            # end
        elseif length(expr.args) == 3
            a, b = expr.args[2], expr.args[3]
            _collect_symbols_and_constants!(symbolDict, constants, denominators, a; count_params = count_params)
            if func == :/ && a isa Integer && b isa Integer && b > 0
                # a division of two integer literals is a rational constant
                # (the form of a rational in the bytecode path)
                push!(denominators, b)
            else
                _collect_symbols_and_constants!(symbolDict, constants, denominators, b; count_params = count_params)
            end
            countSy!(func)
        else
            throw(ArgumentError("only unary and binary functions are supported ($func is not supported)"))
        end
    elseif sy == :ref
        @error "unexpected :ref in expression, this should have been removed by symbol_frequency" expr
    elseif sy == :->
        # only count the body
        _collect_symbols_and_constants!(symbolDict, constants, denominators, expr.args[2]; count_params = count_params)
    elseif sy == :block
        # only count the first expression
        @assert length(expr.args) == 1
        _collect_symbols_and_constants!(symbolDict, constants, denominators, expr.args[1]; count_params = count_params)
    end

    return nothing
end

# `#nodes * log(#distinct symbols)` plus the code lengths of the constants.
function _code_length(symfreq::Dict, constants, denominators, constant_offset)
    numnodes = sum(values(symfreq); init = 0)
    numsymbols = length(symfreq)

    (numnodes > 0 ? numnodes * log(numsymbols) : 0.0) +
        sum(c -> log(c + constant_offset), constants; init = 0.0) + # a value of zero has no complexity
        sum(log, denominators; init = 0.0)                          # denominators are never zero
end

"""
    func_complexity(expr::Expr; count_params = false, constant_offset = 1) -> Float64

Code length of an expression AST: `#nodes * log(#distinct symbols)` plus the
code lengths of the constants: `log(|c| + constant_offset)` per integer
constant and rational numerator, `log(d)` per rational denominator (each
occurrence counts individually). A rational `n/d` with `d != 1` adds one `/`
symbol, a negative constant one `-` symbol. Float constants throw an
`ArgumentError`.
"""
function func_complexity(expr::Expr; count_params = false, constant_offset = 1)
    dict, constants, denominators = _symbol_frequency(expr; count_params = count_params)
    _code_length(dict, constants, denominators, constant_offset)
end

"""
    func_complexity(model::ExprModel; kwargs...) -> Float64

Function complexity of an expression-backed model, computed from its
`core_expr` (the Expr-AST counting path).
"""
func_complexity(model::ExprModel; kwargs...) = func_complexity(core_expr(model); kwargs...)

"""
    func_complexity(model::CodeModel; count_params = false, constant_offset = 1) -> Float64

Function complexity of a code-backed model, computed directly from its code
(the DAG bytecode counting path; no expression is constructed).
"""
func_complexity(model::CodeModel; count_params = false, constant_offset = 1) =
    func_complexity(code(model); count_params = count_params, constant_offset = constant_offset)

# Collects the constant `c` of instruction `i` into `target`; a negative one
# counts one `:-` symbol.
function _collect_code_constant!(symfreq, target, c, i)
    isinteger(c) || throw(ArgumentError("constant $c at instruction $i is not supported; " *
        "only integer constants and rationals DIV(CONSTANT n, CONSTANT d) are allowed"))
    c < 0 && (symfreq[:-] = 1 + get(symfreq, :-, 0))
    push!(target, abs(c))
    nothing
end

# `DIV(CONSTANT n, CONSTANT d)` with `d > 0` encodes the rational `n/d`.
_is_rational_div(code, instr) =
    instr.opcode == DIV &&
    code[instr.arg1idx].opcode == CONSTANT &&
    code[instr.arg2idx].opcode == CONSTANT && code[instr.arg2idx].val > 0

"""
    _symbol_frequencies!(symfreq::Dict, constants::AbstractVector, denominators::AbstractVector,
                         code::Vector{<:Instruction}; count_params = false) -> (Dict, Vector, Vector)

Symbol frequencies of a linear program (`Model` code vector): every distinct
variable, scaled variable, and opcode is a distinct symbol; model parameters
are *not* counted unless `count_params = true` (they are accounted for by the
parameter complexity term). Constants are collected into `constants` (each
occurrence contributes its absolute value; negative constants additionally
count one `:-` symbol), except the denominators of rationals
`DIV(CONSTANT n, CONSTANT d)`, which are collected into `denominators`, and
the exponent of every `POWCONST`, mirroring the Expr-path handling. A
non-integer constant or exponent throws an `ArgumentError`.
"""
function _symbol_frequencies!(symfreq::Dict, constants::AbstractVector, denominators::AbstractVector,
        code::Vector{<:Instruction}; count_params = false)
    empty!(symfreq)
    empty!(constants)
    empty!(denominators)
    isdenominator = falses(length(code))
    for instr in code
        _is_rational_div(code, instr) && (isdenominator[instr.arg2idx] = true)
    end
    for (i, instr) in pairs(code)
        if instr.opcode == VAR
            # different variables are different symbols
            key = 1_000_000_000 + Int(instr.arg1idx)
        elseif instr.opcode == SCALEDVAR
            key = 2_000_000_000 + Int(instr.arg1idx)
        elseif instr.opcode == PARAM
            if count_params
                key = PARAM
            else
                continue
            end
        elseif instr.opcode == CONSTANT
            _collect_code_constant!(symfreq, isdenominator[i] ? denominators : constants, instr.val, i)
            continue
        elseif instr.opcode == POWCONST
            # the integer exponent is a constant, as `x^c` in the Expr path
            _collect_code_constant!(symfreq, constants, instr.val, i)
            key = Int(instr.opcode)
        else
            key = Int(instr.opcode)
        end
        symfreq[key] = 1 + get(symfreq, key, 0)
    end
    symfreq, constants, denominators
end

"""
    func_complexity(code::Vector{<:Instruction}; count_params = false, constant_offset = 1) -> Float64

Code length of a linear program, with the same formula as the `Expr` path:
`#nodes * log(#distinct symbols)` plus `log(|c| + constant_offset)` per
integer constant, `POWCONST` exponent and rational numerator and `log(d)` per
rational denominator, where a rational `n/d` is encoded as
`DIV(CONSTANT n, CONSTANT d)` with `d > 0`. A non-integer constant or exponent
throws an `ArgumentError`.

Note: the code is a DAG, so shared subexpressions (including hashconsed
constants) are counted once, while the `Expr` path counts every occurrence;
the two representations agree for programs without shared subexpressions.
"""
function func_complexity(code::Vector{<:Instruction}; count_params = false, constant_offset = 1)
    symfreq = Dict{Any,Int}()
    constants = Float64[]
    denominators = Float64[]
    _symbol_frequencies!(symfreq, constants, denominators, code; count_params = count_params)
    _code_length(symfreq, constants, denominators, constant_offset)
end

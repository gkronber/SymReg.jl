# Symbolic differentiation of linear DAG code.
#
# `differentiate` transforms a `Vector{Instruction{T}}` (a topologically
# sorted DAG whose last instruction is the root) into a new code vector that
# evaluates `f(x)` and `df/dx[varidx]` in a single left-to-right evaluation
# pass.  The two results are read out at two positions: `f` at the returned
# `fidx` and the derivative at `dfidx`.
#
# Callers must read the derivative at `dfidx` and must not assume it is the
# last instruction of `dcode`.  Emission is hashconsed, so whenever the root's
# derivative is structurally equal to a node that was already emitted -- most
# easily the shared zero constant, when the root does not depend on
# `x[varidx]` and the primal already contains a literal zero -- `emit!` returns
# that earlier index and pushes nothing.  `dfidx` then points into the middle
# of the program.  `dcode` is topologically sorted, so `dcode[1:dfidx]` is a
# valid program whose root is the derivative if a single-output evaluation is
# wanted.
#
# Both the remapped primal nodes and the derivative nodes are emitted through
# a single hashcons cache keyed on `(opcode, arg1idx, arg2idx, val)`, so any
# subexpression needed by both `f` and `df/dx` (e.g. the operands of a
# product rule term) exists exactly once in the output code and is evaluated
# once per input row.  Sub-DAGs that neither are reachable from the root nor
# depend on `x[varidx]` are pruned.

"""
    differentiate(code, varidx::Integer)
        -> (dcode::Vector{Instruction{T}}, fidx::Int, dfidx::Int)
    differentiate(code, varidxs::AbstractVector{<:Integer})
        -> (dcode, fidx, dfidxs::Vector{Int})

Symbolically differentiate linear DAG `code` with respect to input variable
`x[varidx]` (or several `x[varidxs]` at once).  The returned `dcode` evaluates
`f(x)` and all `df/dx[varidxs[k]]` in one pass: `f` is the result of
instruction `fidx`, and the derivative with respect to `varidxs[k]` is the
result of instruction `dfidxs[k]`.  Because emission is hashconsed, a returned
index is *not* in general the last instruction of its chain: read every result
at the index that was returned for it.  The primal section is shared by all chains — derivative
nodes of different chains hashcons against the same primal nodes, so shared
subexpressions between value and derivatives are de-duplicated and evaluated
once.  Derivative rules mirror the reverse-mode kernels in the interpreter;
away from the interpreter's guarded degenerate points (zero denominators,
negative `POW` bases) both agree.
"""
function differentiate(code::Vector{Instruction{T}}, varidx::Integer) where {T}
    dcode, fidx, dfidxs = _differentiate_multi(code, Int[varidx])
    dcode, fidx, dfidxs[1]
end

function differentiate(code::Vector{Instruction{T}}, varidxs::AbstractVector{<:Integer}) where {T}
    _differentiate_multi(code, collect(Int, varidxs))
end

function _differentiate_multi(code::Vector{Instruction{T}}, varidxs::Vector{Int}) where {T}
    n = length(code)
    n == 0 && throw(ArgumentError("cannot differentiate empty code"))
    for v in varidxs
        v >= 1 || throw(ArgumentError("varidx must be >= 1, got $v"))
    end

    # is each node reachable from the root (last instruction)?
    needed = falses(n)
    needed[n] = true
    @inbounds for i in n:-1:1
        if needed[i]
            opc = code[i].opcode
            if opc != STOP && degree(opc) >= 1
                needed[code[i].arg1idx] = true
                degree(opc) == 2 && (needed[code[i].arg2idx] = true)
            end
        end
    end

    # hashconsed emission: returns the index of the (unique) instruction with
    # the given (opcode, arg1idx, arg2idx, val).  For function nodes `val` is
    # always zero; for CONSTANT it is the value, so equal constants dedupe.
    dcode = Vector{Instruction{T}}()
    sizehint!(dcode, (1 + length(varidxs)) * n)
    cache = Dict{Tuple{Opcode,UInt32,UInt32,T},UInt32}()
    emit! = function (opc::Opcode, a1, a2, val::T)
        key = (opc, UInt32(a1), UInt32(a2), val)
        get!(cache, key) do
            push!(dcode, Instruction{T}(opc, UInt32(a1), UInt32(a2), val))
            UInt32(length(dcode))
        end
    end
    # PARAM nodes are keyed on the parameter index only (not on `val`): the
    # interpreter reads parameter values from `p`, never from `instr.val`
    parammap = Dict{UInt32,UInt32}()
    emit_param! = function (pidx)
        get!(parammap, UInt32(pidx)) do
            push!(dcode, create_param_instruction(T, pidx))
            UInt32(length(dcode))
        end
    end
    emit_zero! = () -> emit!(CONSTANT, UInt32(0), UInt32(0), zero(T))
    emit_one! = () -> emit!(CONSTANT, UInt32(0), UInt32(0), one(T))

    idxmap = zeros(UInt32, n) # primal remapping: old node -> new node

    # Phase 1: primal remapped copies, hashconsed (shared nodes collapse here)
    @inbounds for i in 1:n
        needed[i] || continue
        instr = code[i]
        opc = instr.opcode
        deg = degree(opc)
        if deg == 0
            if opc == PARAM
                idxmap[i] = emit_param!(instr.arg1idx)
            elseif opc == CONSTANT
                # only for constants we want to dispatch on val
                idxmap[i] = emit!(opc, UInt32(0), UInt32(0), instr.val)
            else # VAR, SCALEDVAR, STOP
                idxmap[i] = emit!(opc, instr.arg1idx, instr.arg2idx, zero(T))
            end
        elseif deg == 1
            idxmap[i] = emit!(opc, idxmap[instr.arg1idx], UInt32(0), instr.val)
        else
            idxmap[i] = emit!(opc, idxmap[instr.arg1idx], idxmap[instr.arg2idx], instr.val)
        end
    end
    fidx = Int(idxmap[n])

    # Phase 2: one derivative chain per input variable, appended in order.
    # All chains share the primal section and the hashcons cache; the chain
    # for var k ends with the root's derivative `dmap_k[n]` (or the shared
    # zero constant if the root does not depend on `x[varidxs[k]]`).
    dfidxs = Vector{Int}(undef, length(varidxs))
    dmap = zeros(UInt32, n)
    for (chain, varidx) in enumerate(varidxs)
        # does each node (transitively) depend on x[varidx]?
        depends = falses(n)
        @inbounds for i in 1:n
            opc = code[i].opcode
            if opc == VAR || opc == SCALEDVAR
                depends[i] = code[i].arg1idx == varidx
            elseif opc == CONSTANT || opc == PARAM || opc == STOP
                depends[i] = false
            else
                depends[i] = depends[code[i].arg1idx]
                degree(opc) == 2 && (depends[i] |= depends[code[i].arg2idx])
            end
        end

        fill!(dmap, UInt32(0))
        @inbounds for i in 1:n
            needed[i] || continue
            instr = code[i]
            opc = instr.opcode
            deg = degree(opc)
            if !depends[i]
                dmap[i] = emit_zero!()
            elseif deg == 0
                if opc == VAR
                    dmap[i] = instr.arg1idx == varidx ? emit_one!() : emit_zero!()
                elseif opc == SCALEDVAR
                    # d(p_k * x_j) / dx_j = p_k, read dynamically as a PARAM node
                    dmap[i] = instr.arg1idx == varidx ? emit_param!(instr.arg2idx) : emit_zero!()
                else # CONSTANT, PARAM, STOP
                    dmap[i] = emit_zero!()
                end
            elseif deg == 1
                dmap[i] = _diff_unary!(emit!, emit_zero!, emit_one!, T, opc,
                    idxmap[instr.arg1idx], idxmap[i], dmap[instr.arg1idx], instr.val)
            else
                dmap[i] = _diff_binary!(emit!, emit_zero!, emit_one!, T, opc,
                    idxmap[instr.arg1idx], idxmap[instr.arg2idx], idxmap[i],
                    dmap[instr.arg1idx], dmap[instr.arg2idx])
            end
        end
        dfidxs[chain] = Int(dmap[n])
    end

    dcode, Int(fidx), dfidxs
end

# Derivative of a unary operation.  `a` is the primal index of the argument,
# `cur` the primal index of the operation node itself, `da` the index of the
# argument's derivative; `exponent` is the stored exponent for POWCONST.
# Every emitted node goes through `emit!` (hashconsed).
function _diff_unary!(emit!, zero!, one!, ::Type{T}, opc::Opcode,
                      a::UInt32, cur::UInt32, da::UInt32, exponent::T) where {T}
    if opc == NEG
        emit!(NEG, da, UInt32(0), zero(T))
    elseif opc == INV
        # d(1/u) = -u' / u^2
        emit!(DIV, emit!(NEG, da, UInt32(0), zero(T)),
              emit!(MUL, a, a, zero(T)), zero(T))
    elseif opc == LOG || opc == LOGABS
        # d(log(u)) = d(log(|u|)) = u' / u
        emit!(DIV, da, a, zero(T))
    elseif opc == LOG10
        # d(log10(u)) = u' / (u * log(10))
        emit!(DIV, da,
              emit!(MUL, a, emit!(CONSTANT, UInt32(0), UInt32(0), T(log(10))), zero(T)),
              zero(T))
    elseif opc == EXP
        # d(exp(u)) = exp(u) * u'  (cur is exp(u))
        emit!(MUL, da, cur, zero(T))
    elseif opc == ABS
        # d(|u|) = sign(u) * u'
        emit!(MUL, da, emit!(SIGN, a, UInt32(0), zero(T)), zero(T))
    elseif opc == SIGN
        # sign'(u) = 0 almost everywhere
        zero!()
    elseif opc == SQR
        # d(u^2) = 2 * u * u'
        emit!(MUL, emit!(CONSTANT, UInt32(0), UInt32(0), T(2)),
              emit!(MUL, a, da, zero(T)), zero(T))
    elseif opc == SQRT
        # d(sqrt(u)) = u' / (2 * sqrt(u))  (cur is sqrt(u))
        emit!(DIV, da, emit!(MUL, emit!(CONSTANT, UInt32(0), UInt32(0), T(2)), cur, zero(T)),
              zero(T))
    elseif opc == SQRTABS
        # d(sqrt(|u|)) = sign(u) * u' / (2 * sqrt(|u|))  (cur is sqrt(|u|))
        emit!(DIV, emit!(MUL, da, emit!(SIGN, a, UInt32(0), zero(T)), zero(T)),
              emit!(MUL, emit!(CONSTANT, UInt32(0), UInt32(0), T(2)), cur, zero(T)),
              zero(T))
    elseif opc == POWCONST
        # d(u^c) = c * u^(c-1) * u'
        emit!(MUL, emit!(MUL, da,
                         emit!(POWCONST, a, UInt32(0), exponent - one(T)), zero(T)),
              emit!(CONSTANT, UInt32(0), UInt32(0), exponent), zero(T))
    elseif opc == SIN
        emit!(MUL, da, emit!(COS, a, UInt32(0), zero(T)), zero(T))
    elseif opc == COS
        emit!(NEG, emit!(MUL, da, emit!(SIN, a, UInt32(0), zero(T)), zero(T)),
              UInt32(0), zero(T))
    elseif opc == SINH
        emit!(MUL, da, emit!(COSH, a, UInt32(0), zero(T)), zero(T))
    elseif opc == COSH
        emit!(MUL, da, emit!(SINH, a, UInt32(0), zero(T)), zero(T))
    elseif opc == TAN
        # d(tan(u)) = (1 + tan(u)^2) * u'  (cur is tan(u))
        emit!(MUL, da,
              emit!(ADD, one!(), emit!(MUL, cur, cur, zero(T)), zero(T)), zero(T))
    elseif opc == TANH
        # d(tanh(u)) = (1 - tanh(u)^2) * u'  (cur is tanh(u))
        emit!(MUL, da,
              emit!(SUB, one!(), emit!(MUL, cur, cur, zero(T)), zero(T)), zero(T))
    elseif opc == ASIN
        # d(asin(u)) = u' / sqrt(1 - u^2)
        emit!(DIV, da,
              emit!(SQRT, emit!(SUB, one!(), emit!(SQR, a, UInt32(0), zero(T)), zero(T)),
                    UInt32(0), zero(T)),
              zero(T))
    else
        throw(ArgumentError("no symbolic derivative for opcode $opc"))
    end
end

# Derivative of a binary operation.  `a1`/`a2` are the primal indices of the
# operands, `cur` the primal index of the operation node, `da1`/`da2` the
# indices of the operands' derivatives.
function _diff_binary!(emit!, zero!, one!, ::Type{T}, opc::Opcode,
                       a1::UInt32, a2::UInt32, cur::UInt32,
                       da1::UInt32, da2::UInt32) where {T}
    if opc == ADD
        emit!(ADD, da1, da2, zero(T))
    elseif opc == SUB
        emit!(SUB, da1, da2, zero(T))
    elseif opc == MUL
        # d(u*v) = u'*v + u*v'
        emit!(ADD, emit!(MUL, da1, a2, zero(T)), emit!(MUL, a1, da2, zero(T)), zero(T))
    elseif opc == DIV
        # d(u/v) = (u'*v - u*v') / v^2
        emit!(DIV, emit!(SUB, emit!(MUL, da1, a2, zero(T)),
                         emit!(MUL, a1, da2, zero(T)), zero(T)),
              emit!(MUL, a2, a2, zero(T)), zero(T))
    elseif opc == POW
        # d(u^v) = u^v * (v'*log(u) + v*u'/u)  (cur is u^v)
        emit!(MUL, cur,
              emit!(ADD, emit!(MUL, da2, emit!(LOG, a1, UInt32(0), zero(T)), zero(T)),
                    emit!(DIV, emit!(MUL, a2, da1, zero(T)), a1, zero(T)), zero(T)),
              zero(T))
    elseif opc == POWABS
        # d(|u|^v) = |u|^v * (v'*log(|u|) + v*sign(u)*u'/|u|)  (cur is |u|^v);
        # the inner derivative is v*d(log|u|) = v*sign(u)*u'/|u| — dividing by
        # the signed `u` would cancel `sign(u)` and be wrong for u < 0.
        emit!(MUL, cur,
              emit!(ADD, emit!(MUL, da2, emit!(LOGABS, a1, UInt32(0), zero(T)), zero(T)),
                    emit!(DIV, emit!(MUL, emit!(MUL, a2, da1, zero(T)),
                                     emit!(SIGN, a1, UInt32(0), zero(T)), zero(T)),
                          emit!(ABS, a1, UInt32(0), zero(T)), zero(T)),
                    zero(T)),
              zero(T))
    else
        throw(ArgumentError("no symbolic derivative for opcode $opc"))
    end
end

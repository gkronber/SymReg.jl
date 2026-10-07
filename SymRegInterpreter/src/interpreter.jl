# Evaluator: forward/reverse value, parameter-Jacobian, and input
# Jacobian interpretation of a linear DAG code.  Buffers are drawn from the
# per-task fit cache so repeated evaluations in one task do not allocate.
#
# The primary, user-facing calls are the ones that mutate caller-owned storage:
#   interpret_vec!(vec, code, x, p; batchsize)
#   interpret_grad!(grad, code, x, p; batchsize)
#   interpret_jac!(vec, jacp, jacx, code, x, p; batchsize)
# Convenience wrappers that return a freshly drawn cached buffer:
#   interpret(code, x, p; batchsize)       -> scalar sum of the model output
#   interpret_vec(code, x, p; batchsize)   -> vector output
#   interpret_jac(code, x, p; batchsize)   -> (vector output, parameter Jacobian)
#


# Rows of the tape workspaces: one batch, but no more than the data has.
_tape_rows(x::AbstractMatrix, batchsize::Integer) = max(1, min(Int(batchsize), size(x, 1)))

############################################
# Sum output
############################################

"""
    interpret(m::Model, x::AbstractMatrix, p::AbstractVector; batchsize = 1024)
    interpret(code::Vector{Instruction}, x, p; batchsize = 1024)

The sum of the model output `f(x[i, :], p)` over all rows `i` of `x`.  The rows of
`x` are the observations and its columns the input variables; the rows are
evaluated in batches of `batchsize`.
"""
function interpret(code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024)::TD where {T,TD}
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_sum!(taperesults, tmp, code, x, p, batchsize)
end

function interpret(m::Model{T}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024)::TD where {T,TD}
    interpret(code(m), x, p; batchsize)
end

function _interpret_sum!(taperesults, tmp, code, x, p::AbstractVector{TD}, batchsize)::TD where {TD}
    fval = zero(TD)
    allrows = axes(x, 1)

    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        nactive = length(batchrows)
        _x = @view x[batchrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, batchrows)
        fval += sum(@view taperesults[1:nactive, length(code)])
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        nactive = length(remrows)
        _x = @view x[remrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, remrows)
        fval += sum(@view taperesults[1:nactive, length(code)])
    end
    fval
end


############################################
# Vector output and gradient interpretation
############################################

"""
    interpret_vec(m::Model, x::AbstractMatrix, p::AbstractVector; batchsize = 1024) -> Vector
    interpret_vec(code::Vector{Instruction}, x, p; batchsize = 1024) -> Vector

The model output `f(x[i, :], p)` for every row `i` of `x`.  The returned vector
is a buffer of the fit buffer cache (see [`with_fit_buffer_cache`](@ref)) that
the next evaluation overwrites; copy it to keep it, or use
[`interpret_vec!`](@ref) with your own vector.
"""
function interpret_vec(code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    vec = get_buffer(get_vector_diffcache(T, :vec, size(x, 1)), TD)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_vec!(vec, taperesults, tmp, code, x, p, batchsize)
    vec
end

function interpret_vec(m::Model{T}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    interpret_vec(code(m), x, p; batchsize)
end

"""
    interpret_vec!(vec, code::Vector{Instruction}, x::AbstractMatrix, p::AbstractVector; batchsize = 1024)

Write the model output `f(x[i, :], p)` for every row `i` of `x` into `vec` and
return `nothing`.
"""
function interpret_vec!(vec::AbstractVector{TD}, code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    @assert length(vec) == size(x, 1)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_vec!(vec, taperesults, tmp, code, x, p, batchsize)
    nothing
end

function _interpret_vec!(vec, taperesults, tmp, code, x, p, batchsize)
    allrows = axes(x, 1)
    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        _x = @view x[batchrows, :]
        _interpret_batch!(code, nothing, nothing, taperesults, nothing, tmp, _x, p, batchrows)
        vec[batchrows] .= @view taperesults[1:length(batchrows), length(code)]
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        _x = @view x[remrows, :]
        _interpret_batch!(code, nothing, nothing, taperesults, nothing, tmp, _x, p, remrows)
        vec[remrows] .= @view taperesults[1:length(remrows), length(code)]
    end
    nothing
end

# Two-output variant for `differentiate`d code: evaluates `code` in one
# left-to-right pass and copies the primal value at column `fidx` into `f` and
# the derivative at column `dfidx` into `df`.  Both columns are the indices
# `differentiate` returned; neither is in general the last instruction, because
# hashconsing can collapse the root's derivative onto a node that was already
# emitted.  Buffers are drawn from the per-task fit cache; works for plain and
# ForwardDiff-dual parameters alike.
function interpret_vec2!(f::AbstractVector{TD}, df::AbstractVector{TD},
        code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD},
        fidx::Integer, dfidx::Integer; batchsize=1024) where {T,TD}
    @assert length(f) == length(df) == size(x, 1)
    @assert 1 <= fidx <= length(code)
    @assert 1 <= dfidx <= length(code)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_vec2!(f, df, taperesults, tmp, code, x, p, Int(fidx), Int(dfidx), batchsize)
    nothing
end

function _interpret_vec2!(f, df, taperesults, tmp, code, x, p, fidx::Int, dfidx::Int, batchsize)
    allrows = axes(x, 1)
    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        _x = @view x[batchrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, batchrows)
        f[batchrows] .= @view taperesults[1:length(batchrows), fidx]
        df[batchrows] .= @view taperesults[1:length(batchrows), dfidx]
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        _x = @view x[remrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, remrows)
        f[remrows] .= @view taperesults[1:length(remrows), fidx]
        df[remrows] .= @view taperesults[1:length(remrows), dfidx]
    end
    nothing
end

# Multi-output variant of `interpret_vec2!`: evaluates `code` in one
# left-to-right pass and copies the columns `cols[j]` into `outs[:, j]`.
# Used with `differentiate`d code whose `d` derivative chains are rooted at
# `cols[2:end]` (the primal at `cols[1]`), so `f` and all input partials are
# produced in a single forward pass.
function interpret_vecmat!(outs::AbstractMatrix{TD},
        code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD},
        cols::AbstractVector{<:Integer}; batchsize=1024) where {T,TD}
    @assert size(outs) == (size(x, 1), length(cols))
    cs = Int.(cols)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_vecmat!(outs, taperesults, tmp, code, x, p, cs, batchsize)
    nothing
end

function _interpret_vecmat!(outs, taperesults, tmp, code, x, p, cols, batchsize)
    allrows = axes(x, 1)
    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        _x = @view x[batchrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, batchrows)
        @inbounds for j in eachindex(cols)
            outs[batchrows, j] .= @view taperesults[1:length(batchrows), cols[j]]
        end
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        _x = @view x[remrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, remrows)
        @inbounds for j in eachindex(cols)
            outs[remrows, j] .= @view taperesults[1:length(remrows), cols[j]]
        end
    end
    nothing
end

# Reverse-mode parameter gradient of the scalar objective
# `Σ_i Σ_{(c, λ)} λ_i * code_i[c]`: one forward pass over `code` plus one
# adjoint sweep seeded with per-row adjoint weights `λ` at arbitrary output
# columns `c`.  The gradient is *accumulated* into `grad` (`fill!(grad,
# zero)` first for a fresh gradient), so contributions computed outside the
# interpreter — e.g. likelihood-parameter gradients — can be added before the
# call.  Designed for objectives that combine interpreted model values (and
# their `differentiate`d derivatives) with row-wise likelihood terms evaluated
# outside the interpreter; the cost is independent of the number of parameters.
function interpret_grad_seeded!(grad::AbstractVector{TD},
        code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD},
        seeds::Tuple{Vararg{Pair{Int,<:AbstractVector{TD}}}}; batchsize=1024) where {T,TD}
    @assert length(grad) == length(p)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tapediffs = get_buffer(get_matrix_scratch_diffcache(T, :tape_diffs, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_grad_seeded!(grad, code, taperesults, tapediffs, tmp, x, p, seeds, batchsize)
    nothing
end

function _interpret_grad_seeded!(grad, code, taperesults, tapediffs, tmp, x, p, seeds, batchsize)
    allrows = axes(x, 1)
    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        _x = @view x[batchrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, batchrows)
        _seeded_rev!(grad, code, taperesults, tapediffs, tmp, _x, p, batchrows, seeds)
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        _x = @view x[remrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, remrows)
        _seeded_rev!(grad, code, taperesults, tapediffs, tmp, _x, p, remrows, seeds)
    end
    nothing
end

# adjoint sweep over pre-seeded tapediffs: seeds accumulate into the zeroed
# buffer (`+=` so coinciding seed columns add up), then the standard reverse
# sweep propagates them to the PARAM / SCALEDVAR nodes.
function _seeded_rev!(grad, code, taperesults, tapediffs, tmp, x, p, rows, seeds)
    nactive = length(rows)
    TD = eltype(tapediffs)
    # Only the `nactive x length(code)` corner is read by the sweep below, and
    # `tapediffs` comes from a grow-to-fit workspace that keeps the largest
    # shape any model in this task has ever needed.  Zeroing the whole buffer
    # is not necessary.
    fill!((@view tapediffs[1:nactive, 1:length(code)]), zero(TD))
    @inbounds for (col, adj) in seeds
        for (i, row) in enumerate(rows)
            tapediffs[i, col] += adj[row]
        end
    end
    dep = _reverse_mask(code, false)
    pc = length(code)
    while pc >= 1
        @inbounds dep[pc] && _step_backwards!(code[pc], nothing, grad, nothing, taperesults, tapediffs, tmp, x, p, pc)
        pc -= 1
    end
    nothing
end

"""
    interpret_grad!(grad, m::Model, x::AbstractMatrix, p::AbstractVector; batchsize = 1024)
    interpret_grad!(grad, code::Vector{Instruction}, x, p; batchsize = 1024)

Write the gradient with respect to `p` of the summed model output
`Σᵢ f(x[i, :], p)` into `grad` (reverse mode) and return the sum.
"""
function interpret_grad!(grad::AbstractVector{TD}, code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024)::TD where {T,TD}
    @assert length(grad) == length(p)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tapediffs = get_buffer(get_matrix_scratch_diffcache(T, :tape_diffs, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_grad!(grad, code, taperesults, tapediffs, tmp, x, p, batchsize)
end

function interpret_grad!(grad::AbstractVector{TD}, m::Model{T}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024)::TD where {T,TD}
    interpret_grad!(grad, code(m), x, p; batchsize)
end

function _interpret_grad!(grad::AbstractVector{TD}, code, taperesults, tapediffs, tmp, x, p::AbstractVector{TD}, batchsize)::TD where {TD}
    fill!(grad, zero(TD))
    fval = zero(TD)
    allrows = axes(x, 1)

    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        nactive = length(batchrows)
        _x = @view x[batchrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, batchrows)
        fval += sum(@view taperesults[1:nactive, length(code)])
        _interpret_rev_grad!(code, grad, taperesults, tapediffs, tmp, nactive, _x, p)
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        nactive = length(remrows)
        _x = @view x[remrows, :]
        _interpret_fwd!(code, taperesults, tmp, _x, p, remrows)
        fval += sum(@view taperesults[1:nactive, length(code)])
        _interpret_rev_grad!(code, grad, taperesults, tapediffs, tmp, nactive, _x, p)
    end
    fval
end


############################################
# Vector output and Jacobian interpretation
############################################

"""
    interpret_jac(m::Model, x::AbstractMatrix, p::AbstractVector; batchsize = 1024) -> (vec, jacp)
    interpret_jac(code::Vector{Instruction}, x, p; batchsize = 1024) -> (vec, jacp)

The model output for every row of `x` and the Jacobian with respect to the
parameters, a matrix with one row per row of `x` and one column per parameter.
Both are buffers of the fit buffer cache (see [`with_fit_buffer_cache`](@ref))
that the next evaluation overwrites; use [`interpret_jac!`](@ref) to write into
your own arrays.
"""
function interpret_jac(code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    vec = get_buffer(get_vector_diffcache(T, :vec, size(x, 1)), TD)
    jp = get_matrix_scratch_view(T, :jacp, size(x, 1), length(p), TD) # Grow-to-fit + exact-size view
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tapediffs = get_buffer(get_matrix_scratch_diffcache(T, :tape_diffs, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_jac!(vec, jp, nothing, taperesults, tapediffs, tmp, code, x, p, batchsize)
    vec, jp
end

function interpret_jac(m::Model{T}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    interpret_jac(code(m), x, p; batchsize)
end

"""
    interpret_jac!(vec, jacp, jacx, m::Model, x::AbstractMatrix, p::AbstractVector; batchsize = 1024)
    interpret_jac!(vec, jacp, jacx, code::Vector{Instruction}, x, p; batchsize = 1024)

Write the model output for every row of `x` into `vec`, its Jacobian with respect
to the parameters into `jacp` (size `size(x, 1) × length(p)`) and its Jacobian
with respect to the inputs into `jacx` (size `size(x)`, one column per column of
`x` whether or not the model uses it), and return `nothing`.  Pass `nothing` for
`jacp` or `jacx` to skip that Jacobian.
"""
function interpret_jac!(vec::AbstractVector{TD}, jacp::Union{Nothing,AbstractMatrix{TD}}, jacx::Union{Nothing,AbstractMatrix{TD}},
        code::Vector{Instruction{T}}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    @assert length(vec) == size(x, 1)
    @assert isnothing(jacp) || size(jacp) == (size(x, 1), length(p))
    @assert isnothing(jacx) || size(x) == size(jacx)
    taperesults = get_buffer(get_matrix_scratch_diffcache(T, :tape_results, _tape_rows(x, batchsize), length(code)), TD)
    tapediffs = get_buffer(get_matrix_scratch_diffcache(T, :tape_diffs, _tape_rows(x, batchsize), length(code)), TD)
    tmp = get_buffer(get_vector_scratch_diffcache(T, :tmp, _tape_rows(x, batchsize)), TD)
    _interpret_jac!(vec, jacp, jacx, taperesults, tapediffs, tmp, code, x, p, batchsize)
    nothing
end

function interpret_jac!(vec::AbstractVector{TD}, jacp::Union{Nothing,AbstractMatrix{TD}}, jacx::Union{Nothing,AbstractMatrix{TD}},
        m::Model{T}, x::AbstractMatrix{T}, p::AbstractVector{TD}; batchsize=1024) where {T,TD}
    interpret_jac!(vec, jacp, jacx, code(m), x, p; batchsize)
end

function _interpret_jac!(vec, jacp, jacx, taperesults, tapediffs, tmp,
        code, x, p, batchsize)
    !isnothing(jacp) && fill!(jacp, zero(eltype(jacp)))
    !isnothing(jacx) && fill!(jacx, zero(eltype(jacx)))
    allrows = axes(x, 1)

    start = first(allrows)
    while start + batchsize - 1 <= last(allrows)
        batchrows = start:(start + batchsize - 1)
        jp = isnothing(jacp) ? nothing : @view jacp[batchrows, :]
        jx = isnothing(jacx) ? nothing : @view jacx[batchrows, :]
        _x = @view x[batchrows, :]
        _interpret_batch!(code, jp, jx, taperesults, tapediffs, tmp, _x, p, batchrows)
        vec[batchrows] .= (@view taperesults[1:length(batchrows), length(code)])
        start += batchsize
    end

    remrows = start:last(allrows)
    if length(remrows) > 0
        jp = isnothing(jacp) ? nothing : @view jacp[remrows, :]
        jx = isnothing(jacx) ? nothing : @view jacx[remrows, :]
        _x = @view x[remrows, :]
        _interpret_batch!(code, jp, jx, taperesults, tapediffs, tmp, _x, p, remrows)
        vec[remrows] .= (@view taperesults[1:length(remrows), length(code)])
    end
    nothing
end

# forward and backward pass (if necessary) for a single batch of rows.
function _interpret_batch!(code,
        jacp, jacx, taperesults, tapediffs, tmp,
        x, p, rows)
    _interpret_fwd!(code, taperesults, tmp, x, p, rows)

    if !isnothing(jacp) || !isnothing(jacx)
        _interpret_rev!(code, jacp, jacx, taperesults, tapediffs, tmp, x, p, rows)
    end
    nothing
end

# full forward pass: updates taperesults in place.
function _interpret_fwd!(code, taper, tmp, x, p, rows)
    pc = 1
    while pc <= length(code)
        res = view(taper, 1:length(rows), pc)
        _step!(code[pc], taper, tmp, x, p, res)
        pc += 1
    end
    nothing
end

# Which instructions the reverse sweep has to visit at all.
#
# `_step_backwards!` at instruction `i` propagates `i`'s adjoint to its children
# and, at a PARAM / SCALEDVAR / VAR terminal, accumulates into `grad` / `jacp` /
# `jacx`.  Visiting `i` is pointless unless the subtree rooted at `i` contains a
# terminal that one of the requested outputs actually reads. 
#
# `jacx` is additionally seeded by VAR, so a sweep that wants it must keep every
# subtree containing an input -- nearly all of them -- and the mask then prunes
# only constant folds.  `seed_var` selects that.
@inline _mask_seeds(opc::Opcode, seed_var::Bool) =
    opc == PARAM || opc == SCALEDVAR || (seed_var && opc == VAR)

function _reverse_mask(code::Vector{Instruction{T}}, seed_var::Bool) where {T}
    dep = get_vector_cache_view(Bool, :rev_mask, length(code))
    @inbounds for i in eachindex(code)
        instr = code[i]
        opc = instr.opcode
        d = degree(opc)
        dep[i] = if d == 0
            _mask_seeds(opc, seed_var)
        elseif d == 1
            dep[instr.arg1idx]
        else
            dep[instr.arg1idx] | dep[instr.arg2idx]
        end
    end
    dep
end

# full reverse pass: jacobian for p and x, no gradient
function _interpret_rev!(code, jacp, jacx, taperesults, tapediffs, tmp, x::AbstractMatrix, p::AbstractArray, rows)
    pc = length(code)
    nactive = length(rows)
    TD = eltype(tapediffs)

    # see `_seeded_rev!`: zero only the corner the sweep reads
    fill!((@view tapediffs[1:nactive, 1:length(code)]), zero(TD))
    fill!((@view tapediffs[1:nactive, length(code)]), one(TD))

    dep = _reverse_mask(code, !isnothing(jacx))
    while pc >= 1
        @inbounds dep[pc] && _step_backwards!(code[pc], jacp, nothing, jacx, taperesults, tapediffs, tmp, x, p, pc)
        pc -= 1
    end
    nothing
end

# full reverse pass: gradient for parameters only, no Jacobians
function _interpret_rev_grad!(code, grad::AbstractVector{TD}, taperesults, tapediffs, tmp, nactive, x::AbstractMatrix, p::AbstractVector) where {TD}
    pc = length(code)

    # see `_seeded_rev!`: zero only the corner the sweep reads
    fill!((@view tapediffs[1:nactive, 1:length(code)]), zero(TD))
    fill!((@view tapediffs[1:nactive, length(code)]), one(TD))

    dep = _reverse_mask(code, false)
    while pc >= 1
        @inbounds dep[pc] && _step_backwards!(code[pc], nothing, grad, nothing, taperesults, tapediffs, tmp, x, p, pc)
        pc -= 1
    end
    nothing
end

# core forward step: updates taperesults in place.
function _step!(instr, taper, tmp, x, p, res)
    opc = instr.opcode
    nactive = length(res)
    deg = degree(opc)
    if deg == 0
        if opc == VAR           copyto!(res, view(x, 1:nactive, instr.arg1idx))
        elseif opc == SCALEDVAR @inbounds for i in 1:nactive res[i] = p[instr.arg2idx] * x[i, instr.arg1idx] end
        elseif opc == PARAM     fill!(res, p[instr.arg1idx])
        elseif opc == CONSTANT  fill!(res, instr.val)
        end
    elseif deg == 1
        arg = view(taper, 1:nactive, instr.arg1idx)
        if     opc == LOG      _vec_log!(res, arg, tmp)
        elseif opc == LOG10    _vec_log10!(res, arg, tmp)
        elseif opc == LOGABS   _vec_logabs!(res, arg, tmp)
        elseif opc == EXP      _vec_exp!(res, arg, tmp)
        elseif opc == ABS      _vec_abs!(res, arg, tmp)
        elseif opc == NEG      _vec_neg!(res, arg, tmp)
        elseif opc == INV      _vec_inv!(res, arg, tmp)
        elseif opc == SIGN     _vec_sign!(res, arg, tmp)
        elseif opc == SQR      _vec_sqr!(res, arg, tmp)
        elseif opc == SQRT     _vec_sqrt!(res, arg, tmp)
        elseif opc == SQRTABS  _vec_sqrtabs!(res, arg, tmp)
        elseif opc == POWCONST _vec_powconst!(res, arg, instr.val, tmp)
        elseif opc == SIN      _vec_sin!(res, arg, tmp)
        elseif opc == SINH     _vec_sinh!(res, arg, tmp)
        elseif opc == COS      _vec_cos!(res, arg, tmp)
        elseif opc == COSH     _vec_cosh!(res, arg, tmp)
        elseif opc == ASIN     _vec_asin!(res, arg, tmp)
        elseif opc == TAN      _vec_tan!(res, arg, tmp)
        elseif opc == TANH     _vec_tanh!(res, arg, tmp)
        else throw(ArgumentError("Unsupported opcode $opc"))
        end
    elseif deg == 2
        left = view(taper, 1:nactive, instr.arg1idx)
        right = view(taper, 1:nactive, instr.arg2idx)

        if     opc == ADD    _vec_add!(res, left, right, tmp)
        elseif opc == SUB    _vec_sub!(res, left, right, tmp)
        elseif opc == MUL    _vec_mul!(res, left, right, tmp)
        elseif opc == DIV    _vec_div!(res, left, right, tmp)
        elseif opc == POW    _vec_pow!(res, left, right, tmp)
        elseif opc == POWABS _vec_powabs!(res, left, right, tmp)
        else throw(ArgumentError("Unsupported opcode $opc"))
        end
    end

    nothing
end

# core backwards step: updates tapediffs, jacp, grad, and jacx in place.
# The caller is responsible for initializing tapediffs to zero and setting the
# last column to one for the output node.
function _step_backwards!(instr, jacp, grad, jacx, taper, tapediffs, tmp, x::AbstractMatrix, p::AbstractVector, pc::Int)
    opc = instr.opcode
    nactive = size(x, 1)
    cur = view(taper, 1:nactive, pc)
    diff = view(tapediffs, 1:nactive, pc)
    deg = degree(opc)

    if deg == 0
        if opc == VAR
            # d(output)/d x[idx]: receive the accumulated adjoint from the row.
            !isnothing(jacx) && for i in 1:nactive jacx[i, instr.arg1idx] += diff[i] end
        elseif opc == SCALEDVAR
            !isnothing(jacx) && for i in 1:nactive jacx[i, instr.arg1idx] += diff[i] * p[instr.arg2idx] end
            !isnothing(jacp) && for i in 1:nactive jacp[i, instr.arg2idx] += diff[i] * x[i, instr.arg1idx] end
            if !isnothing(grad)
                s = zero(eltype(diff))
                @inbounds for i in 1:nactive s += diff[i] * x[i, instr.arg1idx] end
                grad[instr.arg2idx] += s
            end
        elseif opc == PARAM
            !isnothing(jacp) && for i in 1:nactive @inbounds jacp[i, instr.arg1idx] += diff[i] end
            !isnothing(grad) && (grad[instr.arg1idx] += sum(@view diff[1:nactive]))
        elseif opc == CONSTANT
            # nothing to do
        end
    elseif deg == 1
        arg = view(taper, 1:nactive, instr.arg1idx)
        argDiff = view(tapediffs, 1:nactive, instr.arg1idx)
        if     opc == LOG      _vec_rev_log!(argDiff, arg, cur, diff, tmp)
        elseif opc == LOG10    _vec_rev_log10!(argDiff, arg, cur, diff, tmp)
        elseif opc == LOGABS   _vec_rev_logabs!(argDiff, arg, cur, diff, tmp)
        elseif opc == EXP      _vec_rev_exp!(argDiff, arg, cur, diff, tmp)
        elseif opc == ABS      _vec_rev_abs!(argDiff, arg, cur, diff, tmp)
        elseif opc == NEG      _vec_rev_neg!(argDiff, arg, cur, diff, tmp)
        elseif opc == INV      _vec_rev_inv!(argDiff, arg, cur, diff, tmp)
        elseif opc == SIGN     # gradient is zero
        elseif opc == SQR      _vec_rev_sqr!(argDiff, arg, cur, diff, tmp)
        elseif opc == SQRT     _vec_rev_sqrt!(argDiff, arg, cur, diff, tmp)
        elseif opc == SQRTABS  _vec_rev_sqrtabs!(argDiff, arg, cur, diff, tmp)
        elseif opc == POWCONST _vec_rev_powconst!(argDiff, arg, cur, diff, instr.val, tmp)
        elseif opc == SIN      _vec_rev_sin!(argDiff, arg, cur, diff, tmp)
        elseif opc == SINH     _vec_rev_sinh!(argDiff, arg, cur, diff, tmp)
        elseif opc == COS      _vec_rev_cos!(argDiff, arg, cur, diff, tmp)
        elseif opc == COSH     _vec_rev_cosh!(argDiff, arg, cur, diff, tmp)
        elseif opc == ASIN     _vec_rev_asin!(argDiff, arg, cur, diff, tmp)
        elseif opc == TAN      _vec_rev_tan!(argDiff, arg, cur, diff, tmp)
        elseif opc == TANH     _vec_rev_tanh!(argDiff, arg, cur, diff, tmp)
        else throw(ArgumentError("Unsupported opcode $opc"))
        end
    elseif deg == 2
        left = view(taper, 1:nactive, instr.arg1idx)
        leftDiff = view(tapediffs, 1:nactive, instr.arg1idx)
        right = view(taper, 1:nactive, instr.arg2idx)
        rightDiff = view(tapediffs, 1:nactive, instr.arg2idx)

        if     opc == ADD    _vec_rev_add!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        elseif opc == SUB    _vec_rev_sub!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        elseif opc == MUL    _vec_rev_mul!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        elseif opc == DIV    _vec_rev_div!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        elseif opc == POW    _vec_rev_pow!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        elseif opc == POWABS _vec_rev_powabs!(leftDiff, rightDiff, left, right, cur, diff, tmp)
        else throw(ArgumentError("Unsupported opcode $opc"))
        end
    end

    nothing
end

# ---------------------------------------------------------------------------
# Vector kernels (primal + reverse)
# ---------------------------------------------------------------------------

for unaryfunc in (:abs, :sin, :cos, :cosh, :asin, :tan, :tanh, :sinh)
    funcsy = Symbol("_vec_$(unaryfunc)!")
    @eval function $funcsy(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
        for i in eachindex(res) @inbounds res[i] = NaNMath.$unaryfunc(arg[i]) end; nothing
    end
end

function _vec_exp!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = _exp(arg[i]) end; nothing
end
function _vec_log!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = arg[i] < zero(T) ? T(NaN) : _log(arg[i]) end; nothing
end
function _vec_log10!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = arg[i] < zero(T) ? T(NaN) : _log(arg[i]) * T(1.0/log(10.0)) end; nothing
end
function _vec_logabs!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = arg[i] == zero(T) ? T(NaN) : _log(abs(arg[i])) end; nothing
end
function _vec_sqr!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = arg[i] * arg[i] end; nothing
end
function _vec_sqrt!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = NaNMath.sqrt(arg[i]) end; nothing
end
function _vec_sqrtabs!(res::AbstractVector{T}, arg::AbstractVector{T}, ::AbstractVector{T}) where T<:Real
    for i in eachindex(res) @inbounds res[i] = NaNMath.sqrt(abs(arg[i])) end; nothing
end

function _vec_add!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = left[i] + right[i] end; nothing
end
function _vec_sub!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = left[i] - right[i] end; nothing
end
function _vec_mul!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = left[i] * right[i] end; nothing
end
function _vec_div!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = left[i] / right[i] end; nothing
end
function _vec_pow!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = _pow(left[i], right[i]) end; nothing
end
function _vec_powconst!(res, left, right::TC, ::AbstractVector{TE}) where {TE<:Real,TC<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = NaNMath.pow(left[i], right) end; nothing
end
# The platform's libm pow, exp and log (through the `llvm.pow`, `llvm.exp` and
# `llvm.log` intrinsics) for plain IEEE floats. Faster than Base implementations and comparable accuracy.
# Every other element type, e.g. ForwardDiff duals in `information_matrix`, uses
# Base's functions and ForwardDiff's rules for them.
@inline _llvm_pow(x::Float32, y::Float32) = ccall("llvm.pow.f32", llvmcall, Float32, (Float32, Float32), x, y)
@inline _llvm_pow(x::Float64, y::Float64) = ccall("llvm.pow.f64", llvmcall, Float64, (Float64, Float64), x, y)

@inline _exp(x::Float32) = ccall("llvm.exp.f32", llvmcall, Float32, (Float32,), x)
@inline _exp(x::Float64) = ccall("llvm.exp.f64", llvmcall, Float64, (Float64,), x)
@inline _exp(x) = NaNMath.exp(x)

# log for x >= 0; the kernels handle negative arguments (and Base's `log`
# would throw a DomainError for them, libm returns NaN)
@inline _log(x::Float32) = ccall("llvm.log.f32", llvmcall, Float32, (Float32,), x)
@inline _log(x::Float64) = ccall("llvm.log.f64", llvmcall, Float64, (Float64,), x)
@inline _log(x) = log(x)

# |x|^y
@inline _powabs(x::T, y::T) where {T<:Union{Float32,Float64}} = _llvm_pow(abs(x), y)
@inline _powabs(x, y) = abs(x)^y

# x^y with NaN for a negative base and a non-integer exponent, as
# `NaNMath.pow`.  The guard comes first because C's pow differs from NaNMath
# there for infinite arguments (pow(-Inf, 0.5) = Inf, pow(-2, Inf) = Inf).
@inline _pow(x::T, y::T) where {T<:Union{Float32,Float64}} =
    x < zero(T) && !isinteger(y) ? T(NaN) : _llvm_pow(x, y)
@inline _pow(x, y) = NaNMath.pow(x, y)

function _vec_powabs!(res, left, right, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(res) @inbounds res[i] = _powabs(left[i], right[i]) end; nothing
end

function _vec_neg!(res, arg, ::AbstractVector{TE}) where {TE<:Real}  @simd for i in eachindex(res) @inbounds res[i] = -arg[i] end end
function _vec_inv!(res, arg, ::AbstractVector{TE}) where {TE<:Real}  @simd for i in eachindex(res) @inbounds res[i] = inv(arg[i]) end end
function _vec_sign!(res, arg, ::AbstractVector{TE}) where {TE<:Real} @simd for i in eachindex(res) @inbounds res[i] = sign(arg[i]) end end

function _vec_rev_add!(leftDiff, rightDiff, left, right, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        @inbounds leftDiff[i] += diff[i]
        @inbounds rightDiff[i] += diff[i]
    end
end
function _vec_rev_sub!(leftDiff, rightDiff, left, right, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        @inbounds leftDiff[i] += diff[i]
        @inbounds rightDiff[i] -= diff[i]
    end
end
function _vec_rev_mul!(leftDiff, rightDiff, left, right, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        @inbounds leftDiff[i] += diff[i] * right[i]
        @inbounds rightDiff[i] += diff[i] * left[i]
    end
end
function _vec_rev_div!(leftDiff, rightDiff, left, right, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        if !iszero(right[i])
            @inbounds ftmp = diff[i] / right[i]
            @inbounds leftDiff[i] += ftmp
            @inbounds rightDiff[i] -= ftmp * left[i] / right[i]
        end
    end
end
# d|l|^r/dl = r * sign(l) * |l|^(r-1) = r * cur / l reuses the forward value
# instead of a second `^`
function _vec_rev_powabs!(leftDiff, rightDiff, left, right, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        if !iszero(left[i])
            @inbounds absleft = abs(left[i])
            @inbounds leftDiff[i] += diff[i] * right[i] * cur[i] / left[i]
            @inbounds rightDiff[i] += diff[i] * cur[i] * _log(absleft)
        end
    end
end
function _vec_rev_pow!(leftDiff, rightDiff, left, right, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        @inbounds tmpf = diff[i] * cur[i]
        if !iszero(left[i])
            @inbounds leftDiff[i] += tmpf * right[i] / left[i]
        end
        if left[i] > zero(TE)
            @inbounds rightDiff[i] += tmpf * _log(left[i])
        end
    end
end
function _vec_rev_powconst!(argDiff, arg, cur, diff, exponent::TC, tmp::AbstractVector{TE}) where {TE<:Real, TC<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += diff[i] * exponent * NaNMath.pow(arg[i], (exponent - TE(1.0))) end; nothing
end
function _vec_rev_exp!(argDiff, arg, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(argDiff) @inbounds argDiff[i] += diff[i] * cur[i] end; nothing
end
function _vec_rev_abs!(argDiff, arg, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(argDiff) @inbounds argDiff[i] += diff[i] * sign(arg[i]) end; nothing
end
function _vec_rev_neg!(argDiff, arg, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(argDiff) @inbounds argDiff[i] -= diff[i] end; nothing
end
function _vec_rev_inv!(argDiff, arg, cur, diff, ::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(argDiff) @inbounds argDiff[i] -= diff[i] * cur[i] * cur[i] end; nothing
end
function _vec_rev_log!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += arg[i] <= TE(0) ? TE(0) : diff[i] / arg[i] end; nothing
end
function _vec_rev_log10!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += arg[i] <= TE(0) ? TE(0) : TE((1/log(10))) * diff[i] / arg[i] end; nothing
end
function _vec_rev_logabs!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += arg[i] == TE(0) ? TE(0) : diff[i] / arg[i] end; nothing
end
function _vec_rev_sqr!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += diff[i] * TE(2) * arg[i] end; nothing
end
function _vec_rev_sqrt!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += !iszero(cur[i]) ? diff[i] / (TE(2) * cur[i]) : TE(0) end; nothing
end
function _vec_rev_sqrtabs!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += !iszero(cur[i]) ? sign(arg[i]) * diff[i] / (TE(2) * cur[i]) : TE(0) end; nothing
end
function _vec_rev_sin!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += diff[i] * NaNMath.cos(arg[i]) end; nothing
end
function _vec_rev_sinh!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += diff[i] * NaNMath.cosh(arg[i]) end; nothing
end
function _vec_rev_cos!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] -= diff[i] * NaNMath.sin(arg[i]) end; nothing
    nothing
end
function _vec_rev_cosh!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff) @inbounds argDiff[i] += diff[i] * NaNMath.sinh(arg[i]) end; nothing
end
function _vec_rev_asin!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        if arg[i] * arg[i] != one(TE)
            @inbounds argDiff[i] += diff[i] / sqrt(one(TE) - arg[i] * arg[i])
        end
    end
    nothing
end
function _vec_rev_tan!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        t = cos(arg[i])
        if t != zero(TE)
            @inbounds argDiff[i] += diff[i] / (t * t)
        end
    end
    nothing
end
function _vec_rev_tanh!(argDiff, arg, cur, diff, tmp::AbstractVector{TE}) where {TE<:Real}
    @simd for i in eachindex(diff)
        t = cosh(arg[i])
        if t != zero(TE) && !isinf(t)
            @inbounds argDiff[i] += diff[i] / (t * t)
        end
    end
    nothing
end
# ---------------------------------------------------------------------------
# Rational constant snapping
# ---------------------------------------------------------------------------
# Iteratively try to replace model parameters by rational constants until no
# improvement of the description length is possible.

"""
    rational_constant_snap(l, model, p) -> (model, l, p)

Replace model parameters by rational constants, one at a time with
[`find_best_snap`](@ref), as long as a replacement reduces the
[`description_length`](@ref) of `model` under the likelihood `l` at the natural
parameters `p`.  Returns the snapped model with its remaining parameters.
"""
function rational_constant_snap(l, model, p)
    # try to replace parameters by constants until no improvement of DL is possible
    idx, rational_const, delta_dl, new_param, new_model = find_best_snap(l, model, p)
    while idx > 0
        @assert delta_dl < 0
        model = new_model
        p = new_param
        idx, rational_const, delta_dl, new_param, new_model = find_best_snap(l, model, p)
    end
    @debug "snapped model" model
    (model, l, p)
end

"""
    find_best_snap(l, model, p)

Find the best replacement of a model parameter by a constant.  Returns
`(best_pidx, best_rational, best_delta_dl, best_newparam, best_newmodel)`
with `best_pidx == 0` if no replacement improves the description length.
"""
function find_best_snap(l, model, p)
    # find the best replacement of a parameter by a constant
    best_pidx = 0
    best_delta_dl = 0.0
    best_rational = 0//1
    best_newparam = copy(p)
    best_newmodel = model

    # we cannot snap likelihood parameters, so we only look at model parameters
    # TODO: use FI to iterate parameters from least important to most important
    for i in 1:n_param(model)
        ri, newmodel, delta_dl, newparam = find_best_snap_for_pidx(l, model, p, i)
        if delta_dl < best_delta_dl
            best_rational = ri
            best_delta_dl = delta_dl
            best_pidx = i
            best_newparam = newparam
            best_newmodel = newmodel
        end
    end

    isinteger(best_rational) && (best_rational = numerator(best_rational))
    return best_pidx, best_rational, best_delta_dl, best_newparam, best_newmodel
end


function find_best_snap_for_pidx(l, model, p, pidx)
    best_delta_dl = 0.0
    best_ri = 0//1
    best_newparam = copy(p)
    best_newmodel = model

    dl_start,_ = description_length(l, model, p)
    # iterate over increasingly more accurate approximations of p[i]
    rat_terms, approx = rational_approx(p[pidx], 1_000)
    @debug "starting point:" p=p[pidx] rat_terms=rat_terms
    for ri in approx
        # The likelihood is model-agnostic: the same likelihood is re-evaluated
        # against the snapped model (no need to rebuild it with a new model).
        new_model = replace_param_with_const(model, pidx, ri)
        x0 = vcat(p[1:pidx-1], p[pidx+1:end])   # always start from original parameters
        optresult = optimize(l, new_model, x0)
        p_approx = optresult.x
        dl_approx,_ = description_length(l, new_model, p_approx)
        delta_dl = dl_approx - dl_start
        if delta_dl < best_delta_dl
            best_ri = ri
            best_delta_dl = delta_dl
            best_newparam = p_approx
            best_newmodel = new_model
            @debug "new best improvement" best_delta_dl best_ri
        end
    end
    best_ri, best_newmodel, best_delta_dl, best_newparam
end


#=
/*
** find rational approximation to given real number
** David Eppstein / UC Irvine / 8 Aug 1993
**
** With corrections from Arno Formella, May 2008
**
** usage: a.out r d
**   r is real number to approx
**   d is the maximum denominator allowed
**
** based on the theory of continued fractions
** if x = a1 + 1/(a2 + 1/(a3 + 1/(a4 + ...)))
** then best approximation is found by truncating this series
** (with some adjustments in the last term).
**
** Note the fraction can be recovered as the first column of the matrix
**  ( a1 1 ) ( a2 1 ) ( a3 1 ) ...
**  ( 1  0 ) ( 1  0 ) ( 1  0 )
** Instead of keeping the sequence of continued fraction terms,
** we just keep the last partial product of these matrices.
*/
=#

"""
    rational_approx(x, maxden::Integer) -> (cont_frac, rat)

Rational approximations of `x` with denominator at most `maxden`, from the
continued-fraction expansion.  `cont_frac` holds the continued-fraction terms
and `rat` the convergents in increasing order of accuracy, so `rat[end]` is the
closest approximation found.  An exactly representable `x`, say `2.0` or
`-1.5`, appears in `rat` as itself.

Returns empty lists rather than throwing when `x` cannot be expanded: a
non-finite value, or a magnitude large enough that the convergents would not fit
in an `Int`.  A caller iterating over snap candidates then simply gets none.
"""
function rational_approx(x, maxden::Integer)
    cont_frac = Int[]
    rat = Rational{Int}[]
    maxden >= 1 || throw(ArgumentError("maxden must be >= 1, got $maxden"))

    # A convergent's numerator is about |x| * denominator and the denominator is
    # capped at maxden below, so this bounds the whole expansion to `Int`.
    (isfinite(x) && abs(x) * maxden < typemax(Int) / 2) || return cont_frac, rat

    m = Int[1 0; 0 1]
    startx = x # only for the error comparison at the end

    # loop finding terms until the denominator gets too big
    while true
        ai = floor(Int, x)
        # widened so that a huge term cannot overflow the guard itself
        Int128(m[2, 1]) * ai + m[2, 2] <= maxden || break
        push!(cont_frac, ai)
        m .= m * [ai 1; 1 0]
        # Record this term's convergent *before* any early exit.
        push!(rat, m[1, 1] // m[2, 1])
        x == ai && break            # exact, and 1/(x - ai) would divide by zero
        x = 1 / (x - ai)
        # representation failure, or a remainder too large to truncate
        (isfinite(x) && abs(x) < typemax(Int)) || break
    end

    isempty(rat) && return cont_frac, rat

    # The remaining x lies between 0 and 1/ai.  Approximate it either as 0 (the
    # convergent just recorded) or as the largest term that still fits within
    # maxden, and keep whichever is *nearer*.
    err1 = abs(startx - m[1, 1] / m[2, 1])
    ai = (maxden - m[2, 2]) ÷ m[2, 1]
    lastcol = m * [ai; 1]
    err2 = abs(startx - lastcol[1] / lastcol[2])
    if err2 < err1
        m[:, 1] .= lastcol
        push!(cont_frac, ai)
        push!(rat, m[1, 1] // m[2, 1])
    end
    return cont_frac, rat
end
# ---------------------------------------------------------------------------
# Model-selection criteria
# ---------------------------------------------------------------------------
#
# Parameters: `p` is the natural parameter vector θ = [model params;
# likelihood params] of SymRegLikelihoods, with variances (an estimated
# Gaussian `sigma²`) among the likelihood parameters.

# https://arxiv.org/pdf/2304.06333 Eq. 5, scalar contributions on the vector of
# (projected) parameters and their precisions. Adapted to always store the code 
# for the sign even when the contribution of a parameter is clamped to zero. 
# This accounts for every parameter (even sloppy ones).
# Decision 30/5/2025:
# We never actually snap to integers (neither in rotated or non-rotated space).
# The lattice sqrt(12/prec) is only used implicitly to calculate the
# parameter complexity, but we never actually snap parameters to an integer multiple.
# We have to prevent that parameter complexity becomes negative for
# uncertain parameters.
function param_complexity(p::AbstractVector, prec::AbstractVector)
    p_compl = 0
    for i in eachindex(p)
        # we still need to count the contribution of the parameter to the
        # description length, but we don't want to have a negative contribution.
        p_compl += max(0.0, log(abs(p[i]) / sqrt(12/abs(prec[i])))) + log(2) # for the sign
    end
    p_compl
end

"""
    param_complexity(p, l, model) -> Float64

Unrotated parameter complexity (https://arxiv.org/pdf/2304.06333 Eq. 5) from
the diagonal of the information matrix, after negative eigenvalues are set to
zero precision. Returns `NaN` if the information matrix has non-finite
entries.
"""
function param_complexity(p::AbstractVector, l::AbstractLikelihood, model::Model)
    length(p) > 0 || return 0.0

    FI = information_matrix(l, model, p)
    all(isfinite, FI) || return NaN

    # diag(V * Diagonal(λ) * V')
    λ, V = _nonnegative_eigen(FI)
    return param_complexity(p, (V .^ 2) * λ)
end

"""
    param_complexity_det(p, l, model) -> Float64

Parameter complexity from the *determinant* of the information matrix
(https://arxiv.org/pdf/2304.06333 Eq. 6), which discretises the parameters
jointly on a rectangular lattice of free orientation rather than separately in
each dimension, and so accounts for their degeneracy. Returns `NaN` if the
information matrix has non-finite entries.

Eq. 5 — what `param_complexity` and `param_complexity_rot` implement — carries
`Σ_α ½ log(I_αα) - (p/2) log(3)`. Eq. 6 replaces that whole group by

    ½ log det(I) + (p/2) log(v_p),        log(v_p) = 1 - log(3)

as two separate summands. `p` is the effective rank of `I` here rather than the
parameter count, and `det` the corresponding pseudo-determinant, so a rank
deficiency costs nothing instead of sending the complexity to `-Inf`.
Negative eigenvalues count as zero precision, i.e. as a rank deficiency.
"""
function param_complexity_det(p::AbstractVector, l::AbstractLikelihood, model::Model)
    length(p) > 0 || return 0.0

    FI = information_matrix(l, model, p)
    all(isfinite, FI) || return NaN

    λ = sort(first(_nonnegative_eigen(FI)); rev = true)

    r = _effective_rank(λ)
    if r < length(p)
        @debug "Fisher information matrix is rank deficient" rank = r n_param = length(p)
    end
    log_det_r = sum(log, view(λ, 1:r); init = 0.0)
    log_vp = 1 - log(3)
    log_det_r / 2 + r / 2 * log_vp
end


"""
    param_complexity_rot(p, l, model; scaled = false) -> Float64

Rotated parameter complexity (https://arxiv.org/abs/2605.22374 Eq. 7: 
eigendecomposition of the information matrix, parameter vector projected onto the 
eigenvectors, negative contributions of uncertain parameters clamped to zero. Negative eigenvalues
(directions of negative curvature, away from a minimum of the NLL) count as
zero precision, so such a direction costs only its sign. Returns `floatmax` if
the information matrix has non-finite entries. This is the default parameter
complexity of `description_length`.

With `scaled = true` the rotation is found in relative coordinates
`z_α = θ_α / |θ̂_α|`: the eigendecomposition is taken of
`diag|θ̂| · I · diag|θ̂|` and the point projected onto it is `sign(θ̂)`.  Each direction is then charged its number of
significant digits, the quantity the unrotated [`param_complexity`](@ref)
charges each parameter, and the two coincide when the parameters are
uncorrelated.  Unscaled, the rotation mixes parameters measured in different
units, so the result can change with the units of the data and with where an
optimizer lands on a ridge of equivalent optima.  Scaled, it has neither
dependence for scale-type parameters and product-type ridges, and before the
clamp it is never larger than the unrotated complexity.  See also
[`param_complexity_rot_scaled`](@ref).
"""
function param_complexity_rot(p::AbstractVector, l::AbstractLikelihood, model::Model;
                              scaled::Bool = false)
    # https://arxiv.org/abs/2605.22374 Eq. 7
    T = eltype(p)
    length(p) > 0 || return T(0.0)

    FI = information_matrix(l, model, p)
    any(f -> isnan(f) || f > 1e30, FI) && return floatmax(T)
    p_compl = zero(T)
    try
        if scaled
            D = Diagonal(abs.(p))
            prec, V = _nonnegative_eigen(D * FI * D)
            p_proj = V' * sign.(p)
        else
            prec, V = _nonnegative_eigen(FI)
            p_proj = V' * p
        end

        for i in eachindex(p_proj)
            p_compl += T(log(2)) +  # always count log(2) for the sign
                       max(zero(T), log(abs(p_proj[i])) + T(0.5) * (log(abs(prec[i]))) - T(0.5 * log(3)) - T(log(2)))  # no negative contributions for uncertain parameters
            # p_compl += max(zero(T), log(abs(p_proj[i]) / sqrt(12/abs(prec[i])))) + log(2) # for the sign
        end
    catch ex
        return floatmax(T)
    end
    isnan(p_compl) && return floatmax(T)
    p_compl
end

"""
    param_complexity_rot_scaled(p, l, model) -> Float64

[`param_complexity_rot`](@ref) with `scaled = true`, as a function of
`(p, l, model)` so that it can be passed as `paramCompFunc`.
"""
param_complexity_rot_scaled(p::AbstractVector, l::AbstractLikelihood, model::Model) =
    param_complexity_rot(p, l, model; scaled = true)

# ---------------------------------------------------------------------------
# Description length
# ---------------------------------------------------------------------------

"""
    description_length(l, model, p; paramCompFunc = param_complexity_rot,
                       funcCompFunc = func_complexity) -> (dl, nll, funcComp, paramComp)

Description length `dl = nll + funcComp + paramComp`
(https://arxiv.org/pdf/2304.06333). If the parameter complexity fails or is
`NaN`, returns `(Inf, nll, funcComp, paramComp)`.
"""
function description_length(l::AbstractLikelihood, model::Model, p::AbstractVector;
                            paramCompFunc = param_complexity_rot,
                            funcCompFunc = func_complexity)
    nll = evaluate_nll(l, model, p)
    funcComp = funcCompFunc(model)
    T = eltype(p)
    paramComp = T(NaN)
    try
        paramComp = paramCompFunc(p, l, model)
    catch e
        # An information matrix that cannot be formed is an ordinary outcome
        # for one candidate in a search, and leaves `paramComp` at `NaN`.  
        # Rethrow other exception types.
        SymRegLikelihoods._is_bug(e) && rethrow()
        @debug "parameter complexity failed" exception = e
    end

    isnan(paramComp) && return (T(Inf), nll, funcComp, paramComp)

    dl = nll + funcComp + paramComp
    return (dl, nll, funcComp, paramComp)
end

"""
    description_length_terms(l, model, p; paramCompFunc = param_complexity_rot)
        -> (nll, funcComp, paramComp)

The three components of the description length as a tuple.
"""
function description_length_terms(l::AbstractLikelihood, model::Model, p::AbstractVector;
                                  paramCompFunc = param_complexity_rot)
    nll = evaluate_nll(l, model, p)
    funcComp = func_complexity(model)
    T = eltype(p)
    paramComp = T(NaN)
    try
        paramComp = paramCompFunc(p, l, model)
    catch e
        # as above
        SymRegLikelihoods._is_bug(e) && rethrow()
        @debug "parameter complexity failed" exception = e
    end
    (nll, funcComp, paramComp)
end

"""
    description_length_complexity_penalty(l, model, p; paramCompFunc = param_complexity_rot) -> Float64

Complexity penalty `funcComp + paramComp` of the description length (used, e.g.,
as a fitness function on its own).
"""
function description_length_complexity_penalty(l::AbstractLikelihood, model::Model, p::AbstractVector;
                                               paramCompFunc = param_complexity_rot)
    func_complexity(model) + paramCompFunc(p, l, model)
end

# ---------------------------------------------------------------------------
# Fractional Bayes factor
# ---------------------------------------------------------------------------

"""
    fractional_bayes_factor(l, model, p; funcCompFunc = func_complexity) -> Float64

Fractional Bayes factor model-selection criterion
(https://arxiv.org/pdf/2304.06333 Eq. 11):
`(1 - b) * nll - k/2 * log(b) + funcComp + k/2 * log(2 * pi * nup)` with
`b = 1/sqrt(n)` and `nup = exp(1 - log(3))` (rectangular lattice).
"""
function fractional_bayes_factor(l::AbstractLikelihood, model::Model, p::AbstractVector;
                                 funcCompFunc = func_complexity)
    # https://arxiv.org/pdf/2304.06333 Eq. 11
    sum(fractional_bayes_factor_terms(l, model, p; funcCompFunc = funcCompFunc))
end

"""
    fractional_bayes_factor_terms(l, model, p) -> ((1 - b) * nll, penalty)

The two components of the fractional Bayes factor: the scaled data fit and the
complexity penalty. Their sum equals [`fractional_bayes_factor`](@ref).
"""
function fractional_bayes_factor_terms(l::AbstractLikelihood, model::Model, p::AbstractVector; 
                                                    funcCompFunc = func_complexity)
    nll = evaluate_nll(l, model, p)
    b = inv(sqrt(n_observations(l))) # default choice for fractional Bayes factor (alternative log n)
    ((1 - b) * nll, fractional_bayes_factor_complexity_penalty(l, model, p; funcCompFunc = funcCompFunc))
end

"""
    fractional_bayes_factor_complexity_penalty(l, model, p) -> Float64

Complexity penalty of the fractional Bayes factor
(https://arxiv.org/pdf/2304.06333 Eq. 11):
`- k/2 * log(b) + funcComp + k/2 * log(2 * pi * nup)`.
"""
function fractional_bayes_factor_complexity_penalty(l::AbstractLikelihood, model::Model, p::AbstractVector; 
                                                    funcCompFunc = func_complexity)
    nup = exp(1 - log(3)) # rectangular lattice
    nump = n_param(l, model)
    b   = inv(sqrt(n_observations(l))) # default choice for fractional Bayes factor (alternative: log n)

    - nump/2 * log(b) + funcCompFunc(model) + nump/2 * log(2 * pi * nup)
end

# ---------------------------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------------------------

"""
    AIC(l, model, p) -> Float64

Akaike information criterion `2 * nll + 2 * k`, where `k` is the number of
parameters (`n_param(l, model)`): the model parameters plus the likelihood
parameters, including an estimated Gaussian `sigma²`.
"""
function AIC(l::AbstractLikelihood, model::Model, p::AbstractVector)
    nll = evaluate_nll(l, model, p)
    k = n_param(l, model)
    2 * nll + 2 * k
end

"""
    BIC(l, model, p) -> Float64

Bayesian information criterion `2 * (nll + k/2 * log(n))`, where `k` is the
number of parameters (`n_param(l, model)`, see [`AIC`](@ref)) and `n` the
number of observations.
"""
function BIC(l::AbstractLikelihood, model::Model, p::AbstractVector)
    nll = evaluate_nll(l, model, p)
    k = n_param(l, model)
    n = n_observations(l)
    2 * (nll + k/2 * log(n))
end

"""
    BIC_funccompl(l, model, p) -> Float64

BIC with the additional function complexity penalty.
"""
function BIC_funccompl(l::AbstractLikelihood, model::Model, p::AbstractVector)
    nll = evaluate_nll(l, model, p)
    2 * nll + BIC_funccompl_penalty(l, model, p)
end

"""
    BIC_funccompl_penalty(l, model, p) -> Float64

Penalty term of [`BIC_funccompl`](@ref): `2 * (k/2 * log(n) + funcComp)`,
with `k = n_param(l, model)` as in [`BIC`](@ref).
"""
function BIC_funccompl_penalty(l::AbstractLikelihood, model::Model, p::AbstractVector)
    k = n_param(l, model)
    n = n_observations(l)

    2 * (k/2 * log(n) + func_complexity(model))
end

# A likelihood holds observed data but no model or parameter values; the model
# is passed to every evaluation.
#
# Every likelihood is written in its natural parameters θ = [β; φ]: the
# `n_param(model)` model parameters β followed by the `n_likelihood_params(l)`
# likelihood parameters φ.  Scales are variances (σ², b², w²), an intrinsic
# scatter σ_int² is always the first entry of φ, and profiled parameters are
# part of φ as well.
#
# The NLL is always the full NLL in θ.  Profiling and log transforms are done by
# the optimizer (optim.jl), which picks its strategy from the abstract supertype:
#
#   AbstractLikelihood                   joint: NLopt over θ, positive φ as logs
#   └─ AbstractProfiledLikelihood        profiled: NLopt over β, φ from `profile!`
#      └─ AbstractLeastSquaresLikelihood least squares: LM over β on residuals

"""
    AbstractLossFunction{T<:AbstractFloat}

Supertype of all loss functions for fitting a `Model` with element type `T`;
[`AbstractLikelihood`](@ref) is the subtype for negative log-likelihoods.
"""
abstract type AbstractLossFunction{T<:AbstractFloat} end

"""
    AbstractLikelihood{T} <: AbstractLossFunction{T}

A likelihood holds the observed data; the model and the natural parameters
`θ = [β; φ]` (the model parameters followed by the likelihood parameters) are
passed to every evaluation, e.g. `evaluate_nll(l, model, θ)`.  A likelihood
implements [`evaluate_nll`](@ref), [`evaluate_nll_grad!`](@ref) and
[`n_observations`](@ref), and [`n_likelihood_params`](@ref) and
[`positive_params`](@ref) if it has likelihood parameters.  [`optimize`](@ref)
fits it jointly over `θ` unless it is an [`AbstractProfiledLikelihood`](@ref).
"""
abstract type AbstractLikelihood{T} <: AbstractLossFunction{T} end

"""
    AbstractProfiledLikelihood{T} <: AbstractLikelihood{T}

A likelihood whose likelihood parameters have a closed-form maximum-likelihood
estimate given the model parameters, computed by [`profile!`](@ref), so that
[`optimize`](@ref) searches over the model parameters alone.
"""
abstract type AbstractProfiledLikelihood{T} <: AbstractLikelihood{T} end

"""
    AbstractLeastSquaresLikelihood{T} <: AbstractProfiledLikelihood{T}

A profiled likelihood whose negative log-likelihood is a sum of squared residuals
plus terms of the likelihood parameters; [`optimize`](@ref) fits it by
Levenberg-Marquardt on its residuals (see [`residual!`](@ref)).
"""
abstract type AbstractLeastSquaresLikelihood{T} <: AbstractProfiledLikelihood{T} end

# A capped chunk size limits the number of Dual specializations when models
# with different parameter counts are evaluated.
forward_diff_chunk(x::AbstractArray) = ForwardDiff.Chunk(x, MAX_FORWARD_DIFF_CHUNK)

"""
    parametertype(l::AbstractLossFunction{T}) -> Type{T}

The element type `T` of the data and parameters of `l`.
"""
parametertype(::AbstractLossFunction{T}) where {T} = T

"""
    n_likelihood_params(l) -> Int

Number of likelihood parameters φ, the suffix of the natural parameter vector
`θ = [β; φ]`.
"""
n_likelihood_params(::AbstractLossFunction) = 0

"""
   n_param(l,m) -> Int
The length of the parameter vector θ = [β; φ].
"""
n_param(l::AbstractLossFunction, model::Model) = n_param(model) + n_likelihood_params(l)

"""
    positive_params(l) -> indices into φ

The strictly positive likelihood parameters (the variances), which a joint
likelihood optimizes as their logarithm.
"""
positive_params(::AbstractLossFunction) = 1:0 # empty range

"""
    n_observations(l::AbstractLikelihood) -> Int

The number of observations held by `l`.
"""
n_observations(l::AbstractLikelihood) =
    error("n_observations is not implemented for $(typeof(l))")

# ---------------------------------------------------------------------------
# Intrinsic scatter
# ---------------------------------------------------------------------------
#
# An intrinsic scatter adds homoscedastic noise to the given measurement errors:
# s_i² = σ_y,i² + σ_int².  The profiled Gaussian and Laplace likelihoods report
# their estimated noise as intrinsic scatter.

"""
    has_intrinsic_scatter(l::AbstractLikelihood) -> Bool

Whether `l` estimates an intrinsic scatter `σ_int` in addition to the
measurement errors it was given.
"""
has_intrinsic_scatter(::AbstractLikelihood) = false

"""
    intrinsic_scatter(l::AbstractLikelihood, model::Model, θ) -> Real

The intrinsic scatter `σ_int = sqrt(θ[n_param(model) + 1])`, or zero when `l`
has none.
"""
intrinsic_scatter(l::AbstractLikelihood, model::Model, θ::AbstractVector) =
    sqrt(_scatter2(l, θ, n_param(model)))

@inline _scatter2(l::AbstractLikelihood, θ::AbstractVector, nmod::Int) =
    has_intrinsic_scatter(l) ? θ[nmod + 1] : zero(eltype(θ))

# ---------------------------------------------------------------------------
# Evaluation interface for loss functions
# ---------------------------------------------------------------------------

"""
    evaluate_loss(l, model, p) -> Real

Loss value for loss function `l` and model `model` at the complete
parameter vector `p`.
"""
function evaluate_loss end

"""
    evaluate_loss_grad!(grad, l, model, p) -> Real

Evaluate the loss and store its gradient in `grad` (in place); returns the loss.
"""
function evaluate_loss_grad! end

# ---------------------------------------------------------------------------
# Evaluation interface for likelihoods
# ---------------------------------------------------------------------------

"""
    evaluate_nll(l, model, θ) -> Real

Negative log-likelihood of the data under `l` for model `model` at the natural
parameter vector `θ`.
"""
function evaluate_nll end

"""
    evaluate_nll_grad!(grad, l, model, θ) -> Real

Evaluate the NLL and store its gradient w.r.t. `θ` in `grad` (in place);
returns the NLL.

The generic method uses ForwardDiff.  Analytic overrides must accept
ForwardDiff duals so that `information_matrix` can differentiate them.
"""
function evaluate_nll_grad!(grad::AbstractVector, l::AbstractLikelihood, model::Model,
        θ::AbstractVector)
    # checked here rather than in the signature so that every override is more specific
    eltype(grad) === eltype(θ) ||
        throw(MethodError(evaluate_nll_grad!, (grad, l, model, θ)))
    _check_grad_dims(grad, l, model, θ)
    f = q -> evaluate_nll(l, model, q)
    ForwardDiff.gradient!(grad, f, θ, ForwardDiff.GradientConfig(f, θ, forward_diff_chunk(θ)))
    f(θ)
end

evaluate_loss(l::AbstractLikelihood, model::Model, p::AbstractVector) =
    evaluate_nll(l, model, p)
evaluate_loss_grad!(grad::AbstractVector, l::AbstractLikelihood, model::Model, p::AbstractVector) =
    evaluate_nll_grad!(grad, l, model, p)

"""
    information_matrix(l::AbstractLikelihood, model::Model, θ) -> AbstractMatrix

Observed Fisher information matrix at `θ`: the Hessian of the NLL w.r.t. the
natural parameters, taken as the Jacobian of `evaluate_nll_grad!`.
"""
function information_matrix(l::AbstractLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    x = Vector{T}(θ)
    pl = prepare(l, model)
    g! = (g, q) -> (evaluate_nll_grad!(g, pl, q); nothing)
    g = similar(x)
    H = ForwardDiff.jacobian(g!, g, x, ForwardDiff.JacobianConfig(g!, g, x, forward_diff_chunk(x)))
    # remove the rounding asymmetry
    @inbounds for j in axes(H, 2), i in 1:(j - 1)
        avg = (H[i, j] + H[j, i]) / 2
        H[i, j] = avg
        H[j, i] = avg
    end
    H
end

"""
    observation_scatter(l::AbstractLikelihood, model, θ) -> Real

Standard deviation of a single new observation about the model prediction at
`θ`, used for prediction intervals.  Only defined for likelihoods whose noise is
the same at every input.
"""
function observation_scatter(l::AbstractLikelihood, ::Model, ::AbstractVector)
    throw(ArgumentError(
        "$(nameof(typeof(l))) does not define an observation scatter, so a " *
        "prediction interval cannot be formed from it. Its noise is not the " *
        "same at every input, and a prediction interval is evaluated at inputs " *
        "the likelihood has never seen."))
end

function _check_nll_dims(l::AbstractLossFunction, model::Model, p::AbstractVector)
    if length(p) != n_param(l, model)
        throw(DimensionMismatch("length of p ($(length(p))) != n_param(l, model) ($(n_param(l, model)))"))
    end
end
function _check_grad_dims(grad::AbstractVector, l::AbstractLossFunction, model::Model, p::AbstractVector)
    _check_nll_dims(l, model, p)
    if length(grad) != length(p)
        throw(DimensionMismatch("length(p) ($(length(p))) != length(grad) ($(length(grad)))"))
    end
end

# ---------------------------------------------------------------------------
# Prepared likelihoods
# ---------------------------------------------------------------------------
#
# A `PreparedLikelihood` binds a likelihood to a model and holds precomputed
# data that does not depend on the parameters.  Likelihoods without a fast
# evaluation path use `PreparedGeneric`, which delegates to the likelihood.
"""
    PreparedLikelihood{T}

A likelihood bound to a model by [`prepare`](@ref), evaluated with
`evaluate_nll(pl, θ)` and `evaluate_nll_grad!(grad, pl, θ)`.
"""
abstract type PreparedLikelihood{T<:AbstractFloat} end

struct PreparedGeneric{L,M,T} <: PreparedLikelihood{T}
    likelihood::L
    model::M
end

PreparedGeneric(l::AbstractLikelihood{T}, model::Model) where {T} =
    PreparedGeneric{typeof(l),typeof(model),T}(l, model)

"""
    prepare(l::AbstractLikelihood, model::Model) -> PreparedLikelihood

Bind a likelihood to a model, precomputing everything that does not depend on
the parameters.  The result is reused across parameter updates, e.g. inside
`optimize`.
"""
prepare(l::AbstractLikelihood, model::Model) = PreparedGeneric(l, model)

evaluate_nll(pl::PreparedLikelihood, p::AbstractVector) = evaluate_nll(pl.likelihood, pl.model, p)
evaluate_nll_grad!(grad::AbstractVector, pl::PreparedLikelihood, p::AbstractVector) = evaluate_nll_grad!(grad, pl.likelihood, pl.model, p)

# ---------------------------------------------------------------------------
# Profiled likelihoods
# ---------------------------------------------------------------------------
#
# A profiled likelihood has a closed-form MLE φ̂(β) (`profile!`) and is fitted
# over β alone.  By the envelope theorem, the β-gradient of the profile NLL is
# the β-block of the full gradient.
#
# The generic `evaluate_profiled_nll` / `evaluate_profiled_nll_grad!` allocate θ,
# its gradient and (for least squares) the residuals on every call, so they are
# not meant for an optimization loop.  A profiled likelihood fitted by NLopt
# must override both (see `LaplaceProfiledLikelihood`); a least-squares
# likelihood is fitted by `LevenbergMarquardtOptimizer` and should not be passed
# to `NLoptOptimizer`.

"""
    profile!(θ, l::AbstractProfiledLikelihood, model) -> θ

Write the closed-form MLE of the likelihood parameters, given the model
parameters `θ[1:n_param(model)]`, into the suffix of `θ`.
"""
function profile! end

"""
    evaluate_profiled_nll(l::AbstractProfiledLikelihood, model, β) -> Real

The profile NLL `NLL([β; φ̂(β)])` at the model parameters `β`.
"""
evaluate_profiled_nll(l::AbstractProfiledLikelihood, model::Model, β::AbstractVector) =
    evaluate_nll(l, model, _profiled_theta(l, model, β))

"""
    evaluate_profiled_nll_grad!(grad, l::AbstractProfiledLikelihood, model, β) -> Real

The profile NLL and its gradient w.r.t. the model parameters `β`.
"""
function evaluate_profiled_nll_grad!(grad::AbstractVector, l::AbstractProfiledLikelihood,
        model::Model, β::AbstractVector)
    θ = _profiled_theta(l, model, β)
    g = similar(θ)
    nll = evaluate_nll_grad!(g, l, model, θ)
    copyto!(grad, 1, g, 1, length(β))
    nll
end

evaluate_profiled_nll(pl::PreparedLikelihood, β::AbstractVector) =
    evaluate_profiled_nll(pl.likelihood, pl.model, β)
evaluate_profiled_nll_grad!(grad::AbstractVector, pl::PreparedLikelihood, β::AbstractVector) =
    evaluate_profiled_nll_grad!(grad, pl.likelihood, pl.model, β)

function _profiled_theta(l::AbstractProfiledLikelihood, model::Model, β::AbstractVector)
    length(β) == n_param(model) ||
        throw(DimensionMismatch("length of β ($(length(β))) != n_param(model) ($(n_param(model)))"))
    θ = zeros(eltype(β), n_param(l, model))
    copyto!(θ, β)
    profile!(θ, l, model)
end

# ---------------------------------------------------------------------------
# Least-squares likelihoods
# ---------------------------------------------------------------------------
#
# A least-squares likelihood is a profiled likelihood whose profile NLL is
# monotone in a weighted residual sum of squares, so it can be fitted on the
# residuals.  The residual functions read only `β = θ[1:n_param(model)]`.

"""
    residual!(F, l::AbstractLeastSquaresLikelihood, model, β) -> nothing

Write the weighted per-observation residuals of model `model` at `β` into `F`.
"""
function residual! end

"""
    residual_jacobian!(J, l::AbstractLeastSquaresLikelihood, model, β) -> nothing

Write the `n_observations × n_param(model)` residual Jacobian at `β` into `J`.
"""
function residual_jacobian! end

"""
    residual_and_jacobian!(F, J, l::AbstractLeastSquaresLikelihood, model, β) -> nothing

Compute the residuals and their Jacobian in one pass.
"""
function residual_and_jacobian! end

"""
    least_squares_complete!(θ, l::AbstractLeastSquaresLikelihood, F) -> Real

Given the weighted residuals `F` at the model parameters in `θ`, write the
profiled likelihood parameters into the suffix of `θ` and return the NLL.
"""
function least_squares_complete! end

function profile!(θ::AbstractVector, l::AbstractLeastSquaresLikelihood, model::Model)
    F = Vector{eltype(θ)}(undef, n_observations(l))
    residual!(F, l, model, view(θ, 1:n_param(model)))
    least_squares_complete!(θ, l, F)
    θ
end

"""
    evaluate_loss_grad_fd!(grad, l, model, p) -> nll

ForwardDiff gradient of `evaluate_loss`, for cross-checking analytic gradients.
"""
function evaluate_loss_grad_fd!(grad::AbstractVector{T}, l::AbstractLossFunction,
        model::Model, p::AbstractVector{T}) where {T}
    f = q -> evaluate_loss(l, model, q)
    loss = f(p)
    cfg = ForwardDiff.GradientConfig(f, p, forward_diff_chunk(p))
    ForwardDiff.gradient!(grad, f, p, cfg)
    (isnan(loss) || isinf(loss)) && (fill!(grad, zero(T)); loss = floatmax(T))
    loss
end

"""
    evaluate_nll_grad_fd!(grad, l::AbstractLikelihood, model, θ) -> nll
    evaluate_nll_grad_fd!(grad, pl::PreparedLikelihood, θ) -> nll

The gradient of [`evaluate_nll`](@ref) by ForwardDiff, written into `grad`, for
cross-checking the analytic gradient of [`evaluate_nll_grad!`](@ref).
"""
evaluate_nll_grad_fd!(grad::AbstractVector{T}, l::AbstractLikelihood, model::Model, p::AbstractVector{T}) where {T} =
    evaluate_loss_grad_fd!(grad, l, model, p)
evaluate_nll_grad_fd!(grad::AbstractVector{T}, pl::PreparedLikelihood, p::AbstractVector{T}) where {T} =
    evaluate_loss_grad_fd!(grad, pl.likelihood, pl.model, p)

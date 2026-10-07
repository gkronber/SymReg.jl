using Random
using NLopt
using LinearAlgebra: dot
import GeodesicLM

# Optimizers take and return the natural parameters θ = [β; φ] but search in
# coordinates set by the likelihood's abstract supertype:
#
#   joint     u = θ with the positive likelihood parameters as logarithms (NLopt)
#   profiled  u = β; φ follows from `profile!` (NLopt, or LM for least squares)

"""
    OptResult

The result of [`optimize`](@ref): the negative log-likelihood `f` at the fitted
natural parameters `x`, the numbers of objective and gradient evaluations
`num_fevals` and `num_gevals`, the elapsed time `t` in seconds, and the outcome
`returnvalue`, e.g. `:SUCCESS`, `:MAXEVAL_REACHED` or `:EXCEPTION`.  The
remaining fields count iterations, best results, failures and invalid starts.
"""
struct OptResult
    f::Float64
    x::Vector{Float64}   # natural parameters θ
    num_fevals::Int
    num_gevals::Int
    iterations::Int
    num_best::Int
    t::Float64
    num_fail::Int
    num_inv_start::Int
    returnvalue::Symbol
end

"""
    AbstractOptimizer

Supertype of the optimizers that [`optimize`](@ref) accepts:
[`NLoptOptimizer`](@ref) and [`LevenbergMarquardtOptimizer`](@ref).
"""
abstract type AbstractOptimizer end

"""
    NLoptOptimizer(alg::Symbol = :LD_LBFGS; autodiff = false)

Fit with the NLopt algorithm `alg`.  `autodiff = true` uses ForwardDiff instead
of the analytic gradient of the likelihood.
"""
struct NLoptOptimizer <: AbstractOptimizer
    alg::Symbol
    autodiff::Bool
end
NLoptOptimizer(alg::Symbol = :LD_LBFGS; autodiff::Bool = false) = NLoptOptimizer(alg, autodiff)

"""
    LevenbergMarquardtOptimizer(; maxiter = 100)

Fit an [`AbstractLeastSquaresLikelihood`](@ref) with the Levenberg-Marquardt
algorithm of GeodesicLM, with at most `maxiter` iterations.  It works in
`Float64`, also for models of lower precision.
"""
struct LevenbergMarquardtOptimizer <: AbstractOptimizer
    maxiter::Int
end
LevenbergMarquardtOptimizer(; maxiter::Int = 100) = LevenbergMarquardtOptimizer(maxiter)

# ---------------------------------------------------------------------------
# Optimizer coordinates
# ---------------------------------------------------------------------------

"""
    n_opt_params(l, model) -> Int

Number of coordinates the optimizer searches over: `n_param(l, model)` for a
joint likelihood, `n_param(model)` for a profiled one.
"""
n_opt_params(l::AbstractLikelihood, model::Model) = n_param(l, model)
n_opt_params(::AbstractProfiledLikelihood, model::Model) = n_param(model)

"""
    optimizer_parameters(l, model, θ) -> u

The optimizer coordinates of the natural parameters `θ`.
"""
function optimizer_parameters(l::AbstractLikelihood, model::Model, θ::AbstractVector)
    _check_nll_dims(l, model, θ)
    u = collect(float(eltype(θ)), θ)
    nmod = n_param(model)
    for k in positive_params(l)
        u[nmod + k] = log(θ[nmod + k])
    end
    u
end

function optimizer_parameters(l::AbstractProfiledLikelihood, model::Model, θ::AbstractVector)
    _check_nll_dims(l, model, θ)
    collect(float(eltype(θ)), view(θ, 1:n_param(model)))
end

"""
    natural_parameters(l, model, u) -> θ

The natural parameters at the optimizer coordinates `u`.  For a profiled
likelihood this evaluates the closed-form likelihood parameters at `u = β`.
"""
function natural_parameters(l::AbstractLikelihood, model::Model, u::AbstractVector)
    _check_opt_dims(l, model, u)
    _joint_natural!(similar(u, float(eltype(u))), l, n_param(model), u)
end

# θ has the likelihood's precision, at which `profile!` evaluates the model.
function natural_parameters(l::AbstractProfiledLikelihood{T}, model::Model,
        β::AbstractVector) where {T}
    _check_opt_dims(l, model, β)
    θ = zeros(T, n_param(l, model))
    copyto!(θ, β)
    profile!(θ, l, model)
end

function _check_opt_dims(l::AbstractLikelihood, model::Model, u::AbstractVector)
    length(u) == n_opt_params(l, model) || throw(DimensionMismatch(
        "length of u ($(length(u))) != n_opt_params(l, model) ($(n_opt_params(l, model)))"))
    nothing
end

function _joint_natural!(θ::AbstractVector, l::AbstractLikelihood, nmod::Int, u::AbstractVector)
    copyto!(θ, u)
    for k in positive_params(l)
        θ[nmod + k] = exp(u[nmod + k])
    end
    θ
end

# Scratch for θ on the hot path; duals (autodiff) get a fresh vector.
_theta_buffer(u::AbstractVector{S}) where {S<:AbstractFloat} = get_vector_cache(S, :opt_theta, length(u))
_theta_buffer(u::AbstractVector) = similar(u)

# NLL and gradient in optimizer coordinates (∂/∂log θ_k = θ_k ∂/∂θ_k).
function _opt_nll(l::AbstractLikelihood, pl::PreparedLikelihood, u::AbstractVector)
    isempty(positive_params(l)) && return evaluate_nll(pl, u)
    evaluate_nll(pl, _joint_natural!(_theta_buffer(u), l, n_param(pl.model), u))
end

function _opt_nll_grad!(g::AbstractVector, l::AbstractLikelihood, pl::PreparedLikelihood,
        u::AbstractVector)
    isempty(positive_params(l)) && return evaluate_nll_grad!(g, pl, u)
    nmod = n_param(pl.model)
    θ = _joint_natural!(_theta_buffer(u), l, nmod, u)
    nll = evaluate_nll_grad!(g, pl, θ)
    for k in positive_params(l)
        g[nmod + k] *= θ[nmod + k]
    end
    nll
end

_opt_nll(::AbstractProfiledLikelihood, pl::PreparedLikelihood, β::AbstractVector) =
    evaluate_profiled_nll(pl, β)
_opt_nll_grad!(g::AbstractVector, ::AbstractProfiledLikelihood, pl::PreparedLikelihood,
        β::AbstractVector) = evaluate_profiled_nll_grad!(g, pl, β)

function _opt_nll_grad_fd!(g::AbstractVector, l::AbstractLikelihood, pl::PreparedLikelihood,
        u::AbstractVector)
    f = q -> _opt_nll(l, pl, q)
    ForwardDiff.gradient!(g, f, u, ForwardDiff.GradientConfig(f, u, forward_diff_chunk(u)))
    f(u)
end

function _evaluate_only(l::AbstractLikelihood{T}, model::Model) where {T}
    local f, θ
    t = @elapsed begin
        θ = natural_parameters(l, model, T[])
        f = evaluate_nll(l, model, θ)
    end
    OptResult(f, θ, 1, 0, 1, 1, t, 0, 0, :GLOBAL_OPTIMUM)
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_default_optimizer(::AbstractLikelihood) = NLoptOptimizer()
_default_optimizer(::AbstractLeastSquaresLikelihood) = LevenbergMarquardtOptimizer()

"""
    optimize([opt,] l, model, θ0; maxeval) -> OptResult

Fit `model` under `l` from the natural parameters `θ0`, with the likelihood's
default optimizer unless `opt` is given.  The result holds the fitted natural
parameters.

`maxeval` limits the evaluations of the objective: the residual evaluations of
Levenberg-Marquardt (default 0, unlimited; the iterations are limited by the
optimizer's `maxiter`) and the evaluations of the objective and its gradient of
NLopt (default 300).  A fit stopped by it returns `:MAXEVAL_REACHED` and the
best point found.
"""
optimize(l::AbstractLikelihood, model::Model, θ0::AbstractVector; kwargs...) =
    optimize(_default_optimizer(l), l, model, θ0; kwargs...)

optimize(opt::AbstractOptimizer, l::AbstractLikelihood, model::Model, θ0::AbstractVector; kwargs...) =
    _optimize(opt, l, model, optimizer_parameters(l, model, θ0); kwargs...)

"""
    optimize(l, model; num_starts = 10, rng = Random.default_rng()) -> OptResult

Restart the default optimizer from `num_starts` random points (standard normal
in optimizer coordinates) and return the best result.
"""
function optimize(l::AbstractLikelihood{T}, model::Model; num_starts::Int = 10,
                  rng::Random.AbstractRNG = Random.default_rng()) where {T}
    n = n_opt_params(l, model)
    n == 0 && return _evaluate_only(l, model)
    opt = _default_optimizer(l)
    best = nothing
    for _ in 1:num_starts
        r = try
            _optimize(opt, l, model, randn(rng, T, n))
        catch ex
            # a numerically hard start is skipped, a bug or interrupt is not
            _is_bug(ex) && rethrow()
            nothing
        end
        if r !== nothing
            (isnothing(best) || r.f < best.f) && (best = r)
        end
    end
    isnothing(best) && throw(ErrorException("optimization failed for all starts"))
    best
end

# ---------------------------------------------------------------------------
# Levenberg--Marquardt
# ---------------------------------------------------------------------------

# GeodesicLM exit code -> `OptResult.returnvalue`; positive codes mean convergence.
function _lm_returnvalue(code::Int)
    code > 0 && return :SUCCESS
    code == -1 && return :NOT_CONVERGED   # maxiter exhausted
    code == -2 && return :MAXEVAL_REACHED # maxfev exhausted
    code == -11 && return :NAN_PRODUCED
    return :NOT_CONVERGED
end

# GeodesicLM's Float64 point in the likelihood's element type `T`, since the
# interpreter buffers are keyed by `T`.
_lm_point(::Type{Float64}, p::AbstractVector{Float64}) = p
_lm_point(::Type{T}, p::AbstractVector) where {T} =
    copyto!(get_vector_cache(T, :lm_optimizer_p, length(p)), p)

function _optimize(opt::LevenbergMarquardtOptimizer, l::AbstractLeastSquaresLikelihood{T},
                   model::Model, β0::AbstractVector; maxeval::Int = 0) where {T}
    _check_opt_dims(l, model, β0)
    nx = length(β0)
    nx == 0 && return _evaluate_only(l, model)
    nobs = n_observations(l)

    # geodesiclm returns its best point in `x` and `fvec`.  Both must be
    # `Vector{Float64}`; anything else is converted to a copy and the result lost.
    F = get_vector_cache(Float64, :lm_optimizer_f, nobs)
    xbuf = get_vector_cache(Float64, :lm_optimizer_x, nx)
    copyto!(xbuf, β0)
    # The optimizer's scratch, including its `nobs × nx` Jacobian, comes from a
    # task-local workspace.  `geodesiclm` grows it to the largest `(m, n)` seen.
    ws = get_cached_object(:geodesiclm_workspace, Float64) do
        GeodesicLM.GeodesicLMWorkspace{Float64}()
    end::GeodesicLM.GeodesicLMWorkspace{Float64}

    resid! = (p, r) -> residual!(r, l, model, _lm_point(T, p))
    jac! = (p, j) -> residual_jacobian!(j, l, model, _lm_point(T, p))

    # the starting point, completed by the likelihood parameters its residuals imply
    θ0 = zeros(Float64, n_param(l, model))
    copyto!(θ0, β0)
    resid!(xbuf, F)
    initial_loss = Float64(least_squares_complete!(θ0, l, F))
    t = 0.0

    # iaccel = 0: geodesic acceleration would need a finite-difference Avv.
    # ibroyden = 1: Broyden updates instead of a new Jacobian after each accepted
    # step;
    # damp_mode = 0: damp with the identity.
    # Only the relative criterion `frtol` is used, since the scales of the cost
    # and the parameters come from the data.
    try
        local nfev, njev, code
        t = @elapsed begin
            (_, _, _, nfev, njev, _, code) = GeodesicLM.geodesiclm(
                resid!, jac!, nothing;
                x = xbuf, fvec = F, n = nx, m = nobs, workspace = ws,
                print_unit = devnull,
                analytic_jac = true, iaccel = 0,
                ibroyden = 1, incremental_jtj = true, damp_mode = 0,
                Cgoal = 0.0, gtol = 0.0, xtol = 0.0, ftol = 0.0,
                # the residuals are only as precise as `T`
                frtol = Float64(sqrt(eps(T))),
                maxiter = opt.maxiter, maxfev = maxeval)
        end
        # geodesicLM leaves the residuals at its best point in `F`.
        θ = copy(θ0)
        copyto!(θ, xbuf)
        fitted_loss = Float64(least_squares_complete!(θ, l, F))

        if !isfinite(initial_loss) || fitted_loss < initial_loss
            return OptResult(fitted_loss, θ, nfev, njev, 1, 1, t, 0, 0, _lm_returnvalue(code))
        end

        result = fitted_loss > initial_loss ? :NOT_IMPROVED : _lm_returnvalue(code)
        return OptResult(initial_loss, θ0, nfev, njev, 1, 1, t, 0, 0, result)
    catch ex
        # a numerically hard fit is reported as failed, a bug is rethrown
        _is_bug(ex) && rethrow()
        @debug "LM optimizer failed" exception=ex
        return OptResult(initial_loss, θ0, 0, 0, 1, 0, t, 1, 0, :EXCEPTION)
    end
end

# Exceptions that indicate a bug rather than a numerically hard problem, which
# must not be swallowed.
_is_bug(ex) = ex isa UndefVarError || ex isa MethodError || ex isa TypeError ||
              ex isa UndefKeywordError || ex isa InterruptException ||
              ex isa EltypeMismatchError
# NLopt rethrows exceptions from the objective wrapped in a `CapturedException`.
_is_bug(ex::CapturedException) = _is_bug(ex.ex)

# ---------------------------------------------------------------------------
# NLopt
# ---------------------------------------------------------------------------

# NLopt also returns `:FAILURE` when the line search is stopped by the
# objective's precision (e.g. a Float32 model) or a kink (Laplace).  A
# `:FAILURE` that improved on the starting point is a converged fit.
_nlopt_returnvalue(ret::Symbol, f, f0) =
    (ret === :FAILURE && isfinite(f) && isfinite(f0) && f < f0) ? :ROUNDOFF_LIMITED : ret

# NLopt works in Float64; the objective converts parameters and gradient to and
# from the likelihood's element type `T`.
function _optimize(opt::NLoptOptimizer, l::AbstractLikelihood{T}, model::Model,
                   u0::AbstractVector; maxeval::Int = 300,
                   maxtime::Float64 = 300.0,
                   ftol_rel::Float64 = Float64(sqrt(eps(T))),
                   xtol_rel::Float64 = Float64(sqrt(eps(T)))) where {T}
    _check_opt_dims(l, model, u0)
    nx = length(u0)
    nx == 0 && return _evaluate_only(l, model)

    pl = prepare(l, model)
    autodiff = opt.autodiff
    native = T === Float64

    # loss at the starting point, which NLopt evaluates first
    f0 = Ref(NaN)
    _obj = (p, grad) -> begin
        q = native ? p : copyto!(get_vector_cache(T, :nlopt_p, nx), p)
        nll = if isempty(grad)
            Float64(_opt_nll(l, pl, q))
        else
            g = native ? grad : get_vector_cache(T, :nlopt_g, nx)
            v = autodiff ? _opt_nll_grad_fd!(g, l, pl, q) : _opt_nll_grad!(g, l, pl, q)
            native || copyto!(grad, g)
            Float64(v)
        end
        isnan(f0[]) && (f0[] = nll)
        nll
    end

    nlo = NLopt.Opt(opt.alg, nx)
    NLopt.min_objective!(nlo, _obj)
    NLopt.maxeval!(nlo, maxeval)
    NLopt.maxtime!(nlo, maxtime)
    NLopt.ftol_rel!(nlo, ftol_rel)
    NLopt.xtol_rel!(nlo, xtol_rel)
    x = collect(Float64, u0)
    t = @elapsed ((f, xf, ret) = NLopt.optimize!(nlo, x))
    # NLopt turns an `InterruptException` in the objective into `:FORCED_STOP`.
    ret === :FORCED_STOP && throw(InterruptException())
    # with LBFGS every evaluation includes the gradient
    nev = NLopt.numevals(nlo)
    OptResult(f, natural_parameters(l, model, xf), nev, opt.alg === :LD_LBFGS ? nev : 0, 0, 1,
              t, 0, 0, _nlopt_returnvalue(ret, f, f0[]))
end

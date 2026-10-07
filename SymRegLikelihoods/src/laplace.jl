# Laplace noise likelihoods
#
#   y = f(X, β) + ε,   ε_i ~ Laplace(0, b_i)
#   NLL = Σ [ |e_i| / b_i + log(2 b_i) ],   e_i = f_i - y_i
#
# An intrinsic scale b_int is added in quadrature, b_i = sqrt(b_y,i² + b_int²),
# which keeps the Laplace shape and matches the total variance.  Scales are
# parameters as their squares.  One type per way the scale is known:
#
#   LaplaceLikelihood(X, y, b_y)           b_i = b_y,i                    θ = β
#       known scales
#   LaplaceProfiledLikelihood(X, y)        b_i = b                        θ = [β; b²]
#       unknown homoscedastic scale; b̂ = max(mean |e|, b_min), fitted over β
#   LaplaceScatterLikelihood(X, y, b_y)    b_i = sqrt(b_y,i² + b_int²)    θ = [β; b_int²]
#       known scales plus an intrinsic scale, fitted jointly
#
# Gradients: the adjoints λ_i = sign(e_i) / b_i seed a reverse sweep over the
# model, and v = b² inside b_i contributes ∂NLL/∂v = Σ (1/b_i - |e_i|/b_i²) / (2 b_i).

_laplace_pred(::Type{T}, n, ::Type{TD}) where {T,TD} =
    get_buffer(get_vector_diffcache(T, :laplace_pred, n), TD)
_laplace_lam(::Type{T}, n, ::Type{TD}) where {T,TD} =
    get_buffer(get_vector_diffcache(T, :laplace_lam, n), TD)

# ---------------------------------------------------------------------------
# Known scales
# ---------------------------------------------------------------------------

"""
    LaplaceLikelihood(X, y, scale)

Laplace likelihood with known measurement scales `scale` (a scalar or a
per-observation vector).  It has no likelihood parameters.
"""
struct LaplaceLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    scale2::Vector{T}   # squared measurement scales b_y²
    w::Vector{T}        # 1/b_y
    c::T                # Σ log(2 b_y), the constant of the NLL

    function LaplaceLikelihood(X::Matrix{T}, y::Vector{T}, scale) where {T}
        _check_data(X, y)
        b2 = _variances(T, scale, length(y))
        b = sqrt.(b2)
        new{T}(X, y, b2, inv.(b), sum(bi -> log(2 * bi), b))
    end
end

n_observations(l::LaplaceLikelihood) = length(l.y)

function evaluate_nll(l::LaplaceLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    TD = eltype(θ)
    f = _laplace_pred(T, length(l.y), TD)
    interpret_vec!(f, code(model), l.X, θ)
    acc = zero(TD)
    @inbounds for i in eachindex(l.y)
        acc += abs(f[i] - l.y[i]) * l.w[i]
    end
    _finite_loss(acc + l.c, T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::LaplaceLikelihood{T}, model::Model,
        θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    f = _laplace_pred(T, n, TD)
    λ = _laplace_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    acc = zero(TD)
    @inbounds for i in 1:n
        e = f[i] - l.y[i]
        λ[i] = sign(e) * l.w[i]
        acc += abs(e) * l.w[i]
    end
    fill!(grad, zero(TD))
    _prediction_sweep!(grad, model, l.X, θ, λ, acc + l.c, T)
end

# A Laplace(0, b) deviate has standard deviation `b * sqrt(2)`.
function observation_scatter(l::LaplaceLikelihood, ::Model, ::AbstractVector)
    allequal(l.scale2) ||
        throw(ArgumentError("LaplaceLikelihood has a per-observation scale, which has " *
                            "no value at a new input; a prediction interval needs a " *
                            "scalar scale"))
    sqrt(2 * first(l.scale2))
end

# ---------------------------------------------------------------------------
# Known scales plus an intrinsic scale
# ---------------------------------------------------------------------------

"""
    LaplaceScatterLikelihood(X, y, scale)

Laplace likelihood with known measurement scales `scale` (a scalar or a
per-observation vector) plus an intrinsic scale `b_int`, added in quadrature.
`b_int²` is the single likelihood parameter.
"""
struct LaplaceScatterLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    scale2::Vector{T}   # squared measurement scales b_y²

    function LaplaceScatterLikelihood(X::Matrix{T}, y::Vector{T}, scale) where {T}
        _check_data(X, y)
        new{T}(X, y, _variances(T, scale, length(y)))
    end
end

n_observations(l::LaplaceScatterLikelihood) = length(l.y)
n_likelihood_params(::LaplaceScatterLikelihood) = 1
positive_params(::LaplaceScatterLikelihood) = 1:1
has_intrinsic_scatter(::LaplaceScatterLikelihood) = true

function evaluate_nll(l::LaplaceScatterLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    TD = eltype(θ)
    v = θ[n_param(model) + 1]
    v >= 0 || return _worst_loss(TD, T)
    f = _laplace_pred(T, length(l.y), TD)
    interpret_vec!(f, code(model), l.X, θ)
    acc = zero(TD)
    @inbounds for i in eachindex(l.y)
        b = sqrt(l.scale2[i] + v)
        acc += abs(f[i] - l.y[i]) / b + log(2 * b)
    end
    _finite_loss(acc, T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::LaplaceScatterLikelihood{T},
        model::Model, θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    nmod = n_param(model)
    fill!(grad, zero(TD))
    v = θ[nmod + 1]
    v >= 0 || return _worst_loss(TD, T)
    f = _laplace_pred(T, n, TD)
    λ = _laplace_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    nll = zero(TD)
    dv = zero(TD)
    @inbounds for i in 1:n
        b = sqrt(l.scale2[i] + v)
        invb = inv(b)
        e = f[i] - l.y[i]
        λ[i] = sign(e) * invb
        nll += abs(e) * invb + log(2 * b)
        dv += (invb - abs(e) * invb * invb) * invb
    end
    grad[nmod + 1] = dv / 2
    _prediction_sweep!(grad, model, l.X, θ, λ, nll, T)
end

function observation_scatter(l::LaplaceScatterLikelihood, model::Model, θ::AbstractVector)
    allequal(l.scale2) ||
        throw(ArgumentError("LaplaceScatterLikelihood has a per-observation scale, which " *
                            "has no value at a new input; a prediction interval needs a " *
                            "scalar scale"))
    sqrt(2 * (first(l.scale2) + θ[n_param(model) + 1]))
end

# ---------------------------------------------------------------------------
# Unknown homoscedastic scale
# ---------------------------------------------------------------------------

"""
    LaplaceProfiledLikelihood(X, y)

Laplace likelihood with an unknown scale `b`; `b²` is the single likelihood
parameter.  Its MLE is the mean absolute residual, so the likelihood is fitted
over the model parameters alone.

The estimate is bounded below by `b_min = eps(T) · rms(y)`, the resolution of
the data: an exact fit would otherwise have an NLL of `-Inf`.
"""
struct LaplaceProfiledLikelihood{T} <: AbstractProfiledLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    bmin::T     # lower bound b_min of the profiled scale

    function LaplaceProfiledLikelihood(X::Matrix{T}, y::Vector{T}) where {T}
        _check_data(X, y)
        new{T}(X, y, _min_scale(y))
    end
end

n_observations(l::LaplaceProfiledLikelihood) = length(l.y)
n_likelihood_params(::LaplaceProfiledLikelihood) = 1
positive_params(::LaplaceProfiledLikelihood) = 1:1
has_intrinsic_scatter(::LaplaceProfiledLikelihood) = true

# Sum of absolute residuals; reads only the model parameters of `p`.
function _laplace_sae(l::LaplaceProfiledLikelihood{T}, model::Model, p::AbstractVector) where {T}
    TD = eltype(p)
    f = _laplace_pred(T, length(l.y), TD)
    interpret_vec!(f, code(model), l.X, p)
    sae = zero(TD)
    @inbounds for i in eachindex(l.y)
        sae += abs(f[i] - l.y[i])
    end
    sae
end

function evaluate_nll(l::LaplaceProfiledLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    v = θ[n_param(model) + 1]
    v > 0 || return _worst_loss(eltype(θ), T)
    b = sqrt(v)
    _finite_loss(_laplace_sae(l, model, θ) / b + length(l.y) * log(2 * b), T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::LaplaceProfiledLikelihood{T},
        model::Model, θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    nmod = n_param(model)
    fill!(grad, zero(TD))
    v = θ[nmod + 1]
    v > 0 || return _worst_loss(TD, T)
    b = sqrt(v)
    f = _laplace_pred(T, n, TD)
    λ = _laplace_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    sae = zero(TD)
    @inbounds for i in 1:n
        e = f[i] - l.y[i]
        λ[i] = sign(e) / b
        sae += abs(e)
    end
    grad[nmod + 1] = (n / b - sae / (b * b)) / (2 * b)
    _prediction_sweep!(grad, model, l.X, θ, λ, sae / b + n * log(2 * b), T)
end

function profile!(θ::AbstractVector, l::LaplaceProfiledLikelihood, model::Model)
    b = max(_laplace_sae(l, model, θ) / length(l.y), l.bmin)
    θ[n_param(model) + 1] = b * b
    θ
end

# b̂ = max(sae/n, b_min) and the NLL in one pass.
function evaluate_profiled_nll(l::LaplaceProfiledLikelihood{T}, model::Model,
        β::AbstractVector) where {T}
    n = length(l.y)
    sae = _laplace_sae(l, model, β)
    b = max(sae / n, l.bmin)
    _finite_loss(sae / b + n * log(2 * b), T)
end

function evaluate_profiled_nll_grad!(grad::AbstractVector{TD}, l::LaplaceProfiledLikelihood{T},
        model::Model, β::AbstractVector{TD}) where {T,TD}
    n = length(l.y)
    length(β) == length(grad) == n_param(model) ||
        throw(DimensionMismatch("length of β, grad != n_param(model)"))
    f = _laplace_pred(T, n, TD)
    λ = _laplace_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, β)
    sae = zero(TD)
    @inbounds for i in 1:n
        e = f[i] - l.y[i]
        λ[i] = sign(e)
        sae += abs(e)
    end
    # at the bound b̂ does not depend on β, so the gradient is still that at fixed b
    b = max(sae / n, l.bmin)
    λ ./= b
    fill!(grad, zero(TD))
    _prediction_sweep!(grad, model, l.X, β, λ, sae / b + n * log(2 * b), T)
end

observation_scatter(::LaplaceProfiledLikelihood, model::Model, θ::AbstractVector) =
    sqrt(2 * θ[n_param(model) + 1])

# ---------------------------------------------------------------------------
# Expected Fisher information
# ---------------------------------------------------------------------------
# The NLL is piecewise linear in the predictions, so its Hessian is zero in the
# model parameters almost everywhere: the curvature sits in the kinks at
# e_i = 0, which differentiation does not see.  The Laplace likelihoods
# therefore use the expected Fisher information, the curvature of the NLL
# averaged over the kinks.  Per observation the score sign(e)/b of the
# location has variance 1/b², the score |e|/b² - 1/b of the scale b has
# variance 1/b², and the two are uncorrelated.  With v = b² inside b_i
# (∂b/∂v = 1/(2b)):
#
#   I_ββ = Σ J_iᵀ J_i / b_i²,   I_vv = Σ 1/(4 b_i⁴),   I_βv = 0,
#
# with J_i = ∂f_i/∂β.  An invalid scale gives a NaN matrix.

# `H[1:nmod, 1:nmod] = Jᵀ diag(1 ./ b2) J`, the rest of `H` zero; `b2` holds the
# squared scales, a vector or one scalar for all observations.
function _laplace_information!(H::AbstractMatrix{T}, X::AbstractMatrix{T}, model::Model,
        θ::AbstractVector{T}, b2) where {T}
    n = size(X, 1)
    nmod = n_param(model)
    fill!(H, zero(T))
    nmod == 0 && return H
    f = _laplace_pred(T, n, T)
    J = get_matrix_cache_view(T, :laplace_fisher_jac, n, nmod)
    fill!(J, zero(T))
    interpret_jac!(f, J, nothing, model, X, view(θ, 1:nmod))
    @inbounds for j in 1:nmod, k in 1:j
        s = zero(T)
        for i in 1:n
            s += J[i, j] * J[i, k] / _scale2_at(b2, i)
        end
        H[j, k] = s
        H[k, j] = s
    end
    H
end

_scale2_at(b2::Number, _) = b2
_scale2_at(b2::AbstractVector, i) = b2[i]

function information_matrix(l::LaplaceLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    H = Matrix{T}(undef, length(θ), length(θ))
    _laplace_information!(H, l.X, model, Vector{T}(θ), l.scale2)
end

function information_matrix(l::LaplaceScatterLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    k = length(θ)
    H = Matrix{T}(undef, k, k)
    v = T(θ[k])
    v >= 0 || return fill!(H, T(NaN))
    b2 = get_vector_cache_view(T, :laplace_fisher_b2, length(l.y))
    b2 .= l.scale2 .+ v
    _laplace_information!(H, l.X, model, Vector{T}(θ), b2)
    H[k, k] = sum(s2 -> inv(4 * s2 * s2), b2)
    H
end

function information_matrix(l::LaplaceProfiledLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    k = length(θ)
    n = length(l.y)
    H = Matrix{T}(undef, k, k)
    v = T(θ[k])
    v > 0 || return fill!(H, T(NaN))
    _laplace_information!(H, l.X, model, Vector{T}(θ), v)
    H[k, k] = n / (4 * v * v)
    H
end

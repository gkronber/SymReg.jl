# Gaussian noise likelihoods
#
#   y = f(X, β) + ε,   ε_i ~ N(0, s_i²)
#   NLL = ½ Σ [ r_i² / s_i² + log(2π s_i²) ],   r_i = f_i - y_i
#
# One type per way the noise is known:
#
#   GaussianLikelihood(X, y, σ_y)          s_i² = σ_y,i²             θ = β
#       known noise; weighted least squares with weights 1/σ_y
#   GaussianProfiledLikelihood(X, y)       s_i² = σ²                 θ = [β; σ²]
#       unknown homoscedastic noise; σ̂² = max(RSS/n, σ_min²), least squares over β
#   GaussianScatterLikelihood(X, y, σ_y)   s_i² = σ_y,i² + σ_int²    θ = [β; σ_int²]
#       known noise plus an intrinsic scatter, fitted jointly
#
# Gradients: the adjoints λ_i = r_i / s_i² seed a reverse sweep over the model,
# and a variance v in s_i² contributes ∂NLL/∂v = ½ Σ (1/s_i² - r_i²/s_i⁴).

function _check_data(X::AbstractMatrix, y::AbstractVector)
    length(y) == size(X, 1) || throw(DimensionMismatch("length(y) != size(X,1)"))
    nothing
end

# Per-observation variances from a scalar or a vector of standard deviations.
function _variances(::Type{T}, sigma, n::Int) where {T}
    s2 = sigma isa AbstractVector ? T.(sigma) .^ 2 : fill(T(sigma)^2, n)
    length(s2) == n || throw(DimensionMismatch("sigma length mismatch"))
    s2
end

_gauss_pred(::Type{T}, n, ::Type{TD}) where {T,TD} =
    get_buffer(get_vector_diffcache(T, :gauss_pred, n), TD)
_gauss_lam(::Type{T}, n, ::Type{TD}) where {T,TD} =
    get_buffer(get_vector_diffcache(T, :gauss_lam, n), TD)

# Residual Jacobian with NaN entries set to zero.
function _residual_jacobian(l, model::Model, β::AbstractVector)
    ypred, J = interpret_jac(model, l.X, β)
    replace!(x -> isnan(x) ? zero(x) : x, J)
    ypred, J
end

# ---------------------------------------------------------------------------
# Known noise
# ---------------------------------------------------------------------------

"""
    GaussianLikelihood(X, y, sigma)

Gaussian likelihood with known measurement errors `sigma` (a scalar or a
per-observation vector of standard deviations).  It has no likelihood
parameters and is fitted by weighted least squares.
"""
struct GaussianLikelihood{T} <: AbstractLeastSquaresLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    sigma2::Vector{T}   # measurement variances σ_y²
    w::Vector{T}        # weights 1/σ_y
    c::T                # ½ Σ log(2π σ_y²), the constant of the NLL

    function GaussianLikelihood(X::Matrix{T}, y::Vector{T}, sigma) where {T}
        _check_data(X, y)
        s2 = _variances(T, sigma, length(y))
        new{T}(X, y, s2, inv.(sqrt.(s2)), sum(v -> log(2 * T(pi) * v), s2) / 2)
    end
end

n_observations(l::GaussianLikelihood) = length(l.y)

function evaluate_nll(l::GaussianLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    TD = eltype(θ)
    f = _gauss_pred(T, length(l.y), TD)
    interpret_vec!(f, code(model), l.X, θ)
    wrss = zero(TD)
    @inbounds for i in eachindex(l.y)
        r = (f[i] - l.y[i]) * l.w[i]
        wrss += r * r
    end
    _finite_loss(wrss / 2 + l.c, T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::GaussianLikelihood{T}, model::Model,
        θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    f = _gauss_pred(T, n, TD)
    λ = _gauss_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    wrss = zero(TD)
    @inbounds for i in 1:n
        r = f[i] - l.y[i]
        w2 = l.w[i] * l.w[i]
        λ[i] = r * w2
        wrss += r * r * w2
    end
    fill!(grad, zero(TD))
    _prediction_sweep!(grad, model, l.X, θ, λ, wrss / 2 + l.c, T)
end

function residual!(F, l::GaussianLikelihood, model::Model, β::AbstractVector)
    F .= l.w .* (interpret_vec(model, l.X, β) .- l.y)
    nothing
end

function residual_jacobian!(J, l::GaussianLikelihood, model::Model, β::AbstractVector)
    _, Jf = _residual_jacobian(l, model, β)
    J .= Jf .* l.w
    nothing
end

function residual_and_jacobian!(F, J, l::GaussianLikelihood, model::Model, β::AbstractVector)
    ypred, Jf = _residual_jacobian(l, model, β)
    F .= l.w .* (ypred .- l.y)
    J .= Jf .* l.w
    nothing
end

# no profiled parameters to complete
least_squares_complete!(::AbstractVector, l::GaussianLikelihood{T}, F::AbstractVector) where {T} =
    _finite_loss(dot(F, F) / 2 + l.c, T)

function observation_scatter(l::GaussianLikelihood, ::Model, ::AbstractVector)
    allequal(l.sigma2) ||
        throw(ArgumentError("GaussianLikelihood has a per-observation sigma, which has " *
                            "no value at a new input; a prediction interval needs a " *
                            "scalar sigma"))
    sqrt(first(l.sigma2))
end

# ---------------------------------------------------------------------------
# Unknown homoscedastic noise
# ---------------------------------------------------------------------------

"""
    GaussianProfiledLikelihood(X, y)

Gaussian likelihood with an unknown noise variance `σ²`, the single
likelihood parameter.  Its MLE is `RSS/n`, so the likelihood is fitted by
least squares over the model parameters.

The estimate is bounded below by `σ_min² = (eps(T) · rms(y))²`, the resolution
of the data: an exact fit (`RSS = 0`) would otherwise have an NLL of `-Inf`.
"""
struct GaussianProfiledLikelihood{T} <: AbstractLeastSquaresLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    s2min::T    # lower bound σ_min² of the profiled variance

    function GaussianProfiledLikelihood(X::Matrix{T}, y::Vector{T}) where {T}
        _check_data(X, y)
        new{T}(X, y, _min_scale(y)^2)
    end
end

n_observations(l::GaussianProfiledLikelihood) = length(l.y)
n_likelihood_params(::GaussianProfiledLikelihood) = 1
positive_params(::GaussianProfiledLikelihood) = 1:1
has_intrinsic_scatter(::GaussianProfiledLikelihood) = true

function evaluate_nll(l::GaussianProfiledLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    TD = eltype(θ)
    n = length(l.y)
    s2 = θ[n_param(model) + 1]
    s2 > 0 || return _worst_loss(TD, T)
    f = _gauss_pred(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    rss = zero(TD)
    @inbounds for i in 1:n
        r = f[i] - l.y[i]
        rss += r * r
    end
    _finite_loss(rss / (2 * s2) + n * log(2 * T(pi) * s2) / 2, T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::GaussianProfiledLikelihood{T},
        model::Model, θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    nmod = n_param(model)
    fill!(grad, zero(TD))
    s2 = θ[nmod + 1]
    s2 > 0 || return _worst_loss(TD, T)
    f = _gauss_pred(T, n, TD)
    λ = _gauss_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    rss = zero(TD)
    @inbounds for i in 1:n
        r = f[i] - l.y[i]
        λ[i] = r / s2
        rss += r * r
    end
    grad[nmod + 1] = (n - rss / s2) / (2 * s2)
    _prediction_sweep!(grad, model, l.X, θ, λ, rss / (2 * s2) + n * log(2 * T(pi) * s2) / 2, T)
end

function residual!(F, l::GaussianProfiledLikelihood, model::Model, β::AbstractVector)
    F .= interpret_vec(model, l.X, β) .- l.y
    nothing
end

function residual_jacobian!(J, l::GaussianProfiledLikelihood, model::Model, β::AbstractVector)
    J .= _residual_jacobian(l, model, β)[2]
    nothing
end

function residual_and_jacobian!(F, J, l::GaussianProfiledLikelihood, model::Model, β::AbstractVector)
    ypred, Jf = _residual_jacobian(l, model, β)
    F .= ypred .- l.y
    J .= Jf
    nothing
end

# σ̂² = max(RSS/n, σ_min²)
function least_squares_complete!(θ::AbstractVector, l::GaussianProfiledLikelihood{T},
        F::AbstractVector) where {T}
    n = length(F)
    rss = dot(F, F)
    s2 = max(rss / n, l.s2min)
    θ[end] = s2
    _finite_loss(rss / (2 * s2) + n * log(2 * T(pi) * s2) / 2, T)
end

observation_scatter(::GaussianProfiledLikelihood, model::Model, θ::AbstractVector) =
    sqrt(θ[n_param(model) + 1])

# ---------------------------------------------------------------------------
# Known noise plus intrinsic scatter
# ---------------------------------------------------------------------------

"""
    GaussianScatterLikelihood(X, y, sigma)

Gaussian likelihood with known measurement errors `sigma` (a scalar or a
per-observation vector of standard deviations) plus an intrinsic scatter
`σ_int`, added in quadrature.  `σ_int²` is the single likelihood parameter.
"""
struct GaussianScatterLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    sigma2::Vector{T}   # measurement variances σ_y²

    function GaussianScatterLikelihood(X::Matrix{T}, y::Vector{T}, sigma) where {T}
        _check_data(X, y)
        new{T}(X, y, _variances(T, sigma, length(y)))
    end
end

n_observations(l::GaussianScatterLikelihood) = length(l.y)
n_likelihood_params(::GaussianScatterLikelihood) = 1
positive_params(::GaussianScatterLikelihood) = 1:1
has_intrinsic_scatter(::GaussianScatterLikelihood) = true

function evaluate_nll(l::GaussianScatterLikelihood{T}, model::Model, θ::AbstractVector) where {T}
    _check_nll_dims(l, model, θ)
    TD = eltype(θ)
    v = θ[n_param(model) + 1]
    v >= 0 || return _worst_loss(TD, T)
    f = _gauss_pred(T, length(l.y), TD)
    interpret_vec!(f, code(model), l.X, θ)
    acc = zero(TD)
    @inbounds for i in eachindex(l.y)
        r = f[i] - l.y[i]
        s2 = l.sigma2[i] + v
        acc += r * r / s2 + log(2 * T(pi) * s2)
    end
    _finite_loss(acc / 2, T)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::GaussianScatterLikelihood{T},
        model::Model, θ::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, θ)
    n = length(l.y)
    nmod = n_param(model)
    fill!(grad, zero(TD))
    v = θ[nmod + 1]
    v >= 0 || return _worst_loss(TD, T)
    f = _gauss_pred(T, n, TD)
    λ = _gauss_lam(T, n, TD)
    interpret_vec!(f, code(model), l.X, θ)
    nll = zero(TD)
    dv = zero(TD)
    @inbounds for i in 1:n
        r = f[i] - l.y[i]
        s2 = l.sigma2[i] + v
        invs2 = inv(s2)
        λ[i] = r * invs2
        nll += r * r * invs2 + log(2 * T(pi) * s2)
        dv += (1 - r * r * invs2) * invs2
    end
    grad[nmod + 1] = dv / 2
    _prediction_sweep!(grad, model, l.X, θ, λ, nll / 2, T)
end

function observation_scatter(l::GaussianScatterLikelihood, model::Model, θ::AbstractVector)
    allequal(l.sigma2) ||
        throw(ArgumentError("GaussianScatterLikelihood has a per-observation sigma, which " *
                            "has no value at a new input; a prediction interval needs a " *
                            "scalar sigma"))
    sqrt(first(l.sigma2) + θ[n_param(model) + 1])
end

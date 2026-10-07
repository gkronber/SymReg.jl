# Uniform-prior marginal likelihood ("unif", arXiv:2309.00948, Sec. 2.2) for
# error covariances that are banded in time (`BandedCovariance`, see
# banded_covariance.jl).  The true inputs are marginalized under an infinitely
# wide uniform prior, which to first order in the model propagates the input
# errors through the input Jacobian J (n x n*d; J[i, (k-1)*n+i] = g[i,k] =
# ∂f_i/∂X[i,k], at the observed inputs):
#   C   = D + σ²I,   D = Σyy + J Σxx J' - J Σxy - (J Σxy)'
#   NLL = ½ (log det(2π C) + r' C⁻¹ r),   r = f(X) - y
# Up to terms that depend on neither the model nor σ², this is the w² → ∞ limit
# of `MNRGeneralBandedLikelihood` (the latent block decouples), and for a single
# input with diagonal covariances it equals `XUniformDiagonalLikelihood`.  Without
# the latent prior there is no w² that can collapse (with w² → 0 the MNR
# likelihood can explain the outputs of a nearly constant input by its noise,
# when inputs and outputs are derived from the same observations).
#
# C has the half bandwidth h of the covariances, so the NLL costs one band LDLᵀ
# of an n x n matrix, O(n (h + 1)² + n d² (h + 1)).  The gradient follows from
# the band of C⁻¹ (Takahashi's recurrence) as for the MNR likelihood:
#   ∂NLL/∂C_ij = Z_ij - v_i v_j (halved on the diagonal),  Z = C⁻¹, v = C⁻¹ r
# and C depends quadratically on g.
#
# Noise inference: the covariance may contain patterns P_r with unknown weights,
# Σ = Σ_fixed + Σ_r θ_r P_r (e.g. the raw-noise stencil of the target variable
# with weight τ² for an unknown noise level τ).  D is linear in Σ, so
#   C = D(Σ_fixed) + Σ_r θ_r D(P_r) + σ²I,   ∂NLL/∂θ_r = Σ_ij ∂NLL/∂C_ij D(P_r)_ij
# The likelihood parameters are [σ² (with `intrinsic_scatter = true`); θ_1; …; θ_R],
# all non-negative.

"""
    XUniformBandedLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = true)
    XUniformBandedLikelihood(X, y, cov::BandedCovariance; intrinsic_scatter = true)
    XUniformBandedLikelihood(X, y, patterns, weights; intrinsic_scatter = true)

Uniform-prior marginal likelihood ("unif", arXiv:2309.00948) for error
covariances of the inputs `X` (ordered variable-major, `vec(X)`) and the outputs
`y` that are banded in time (see [`BandedCovariance`](@ref)): the input errors
are propagated to the outputs through the input Jacobian of the model.  The
banded generalization of [`XUniformDiagonalLikelihood`](@ref) and the limit of
[`MNRGeneralBandedLikelihood`](@ref) for an infinitely wide prior on the true
inputs.  With `intrinsic_scatter = true`, σ² is the single likelihood parameter.
"""
struct XUniformBandedLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    cov::BandedCovariance{T}             # Σ_fixed
    free::Vector{BandedCovariance{T}}    # patterns P_r with weights θ_r (likelihood parameters)
    intrinsic_scatter::Bool
end

function XUniformBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        cov::BandedCovariance{T}; intrinsic_scatter::Bool = true,
        free_patterns::AbstractVector{<:BandedCovariance{T}} = BandedCovariance{T}[]) where {T}
    n, d = size(X, 1), size(X, 2)
    length(y) == n || throw(DimensionMismatch("length(y) != size(X,1)"))
    all(c -> (c.n, c.d) == (n, d), [cov; free_patterns]) ||
        throw(DimensionMismatch("covariance dimensions do not match data"))
    # a common half bandwidth for Σ_fixed and the free patterns
    h = maximum(c -> c.h, [cov; free_patterns])
    XUniformBandedLikelihood{T}(Matrix{T}(X), Vector{T}(y), _widen(cov, h),
                                BandedCovariance{T}[_widen(c, h) for c in free_patterns], intrinsic_scatter)
end

function XUniformBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        Σxx::AbstractMatrix{T}, Σxy::AbstractMatrix{T}, Σyy::AbstractMatrix{T}; kwargs...) where {T}
    n, d = size(X, 1), size(X, 2)
    (size(Σxx) == (n * d, n * d) && size(Σxy) == (n * d, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("covariance dimensions do not match data"))
    XUniformBandedLikelihood(X, y, BandedCovariance(Σxx, Σxy, Σyy); kwargs...)
end

function XUniformBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        patterns::AbstractVector{<:BandedCovariance{T}}, weights::AbstractVector; kwargs...) where {T}
    length(patterns) == length(weights) >= 1 ||
        throw(ArgumentError("one weight per pattern is needed"))
    XUniformBandedLikelihood(X, y, sum(weights .* patterns); kwargs...)
end

XUniformBandedLikelihood(x::AbstractVector, y::AbstractVector, args...; kwargs...) =
    XUniformBandedLikelihood(reshape(x, :, 1), y, args...; kwargs...)

n_observations(l::XUniformBandedLikelihood) = size(l.X, 1)
has_intrinsic_scatter(l::XUniformBandedLikelihood) = l.intrinsic_scatter
# [σ²]; θ_1, …, θ_R
n_likelihood_params(l::XUniformBandedLikelihood) = Int(l.intrinsic_scatter) + length(l.free)
positive_params(l::XUniformBandedLikelihood) = 1:n_likelihood_params(l)

# σ² (zero without intrinsic scatter) and the weights θ of the free patterns, or
# `nothing` when a variance is negative
function _xub_suffix(l::XUniformBandedLikelihood, nmod::Int, p::AbstractVector)
    ns = Int(l.intrinsic_scatter)
    sig2 = _scatter2(l, p, nmod)
    θ = view(p, nmod + ns + 1:nmod + ns + length(l.free))
    (sig2 >= 0 && all(>=(0), θ)) || return nothing
    (sig2, θ)
end

# Fill the lower band of C into `B` (B[r + 1, i] = C[i + r, i]) and r into `u`.
function _xub_build!(B, u, l::XUniformBandedLikelihood, f, g, sig2, θ)
    n = size(l.X, 1)
    cov = l.cov
    h = cov.h
    @inbounds for i in 1:n
        u[i] = f[i] - l.y[i]
        for r in 0:h
            if i + r <= n
                v = _banded_D(cov, g, i, r)
                for (q, P) in enumerate(l.free)
                    v += θ[q] * _banded_D(P, g, i, r)
                end
                r == 0 && (v += sig2)
                B[r + 1, i] = v
            else
                B[r + 1, i] = zero(eltype(B))
            end
        end
    end
    nothing
end

function _xub_nll(l::XUniformBandedLikelihood{T}, f, g, sig2, θ, ::Type{TE}) where {T,TE}
    n = size(l.X, 1)
    b = l.cov.h
    B = get_buffer(get_matrix_diffcache(T, :xub_B, b + 1, n), TE)
    u = get_buffer(get_vector_diffcache(T, :xub_u, n), TE)
    _xub_build!(B, u, l, f, g, sig2, θ)
    _band_ldlt!(B, b, n) || return _worst_loss(TE, T)
    logdet, quad = _band_logdet_quadratic!(u, B, b, n)
    nll = T(1/2) * (logdet + n * log(2 * T(pi)) + quad)
    (isnan(nll) || isinf(nll)) ? _worst_loss(TE, T) : nll
end

function evaluate_nll(l::XUniformBandedLikelihood{T}, model::Model, p::AbstractVector) where {T}
    _check_nll_dims(l, model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _xub_suffix(l, n_param(model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, θ = suffix
    f = get_buffer(get_vector_diffcache(T, :xub_f, n), TE)
    jacx = get_buffer(get_matrix_diffcache(T, :xub_jacx, n, d), TE)
    interpret_jac!(f, nothing, jacx, model, l.X, p)
    _xub_nll(l, f, jacx, sig2, θ, TE)
end

struct PreparedXUniformBanded{T,L<:XUniformBandedLikelihood{T}} <: PreparedLikelihood{T}
    likelihood::L
    model::Model{T}
    code::Vector{Instruction{T}}   # f and all d input partials in one code vector
    cols::Vector{Int}              # [fidx; dfidxs...]: positions in `code`
end

function prepare(l::XUniformBandedLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.X, 2))
    PreparedXUniformBanded(l, model, dcode, [Int(fidx); Int.(dfidxs)])
end

# Uses the same differentiated code as the gradient, so that both agree.
function evaluate_nll(pl::PreparedXUniformBanded{T}, p::AbstractVector) where {T}
    l = pl.likelihood
    _check_nll_dims(l, pl.model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _xub_suffix(l, n_param(pl.model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, θ = suffix
    outs = get_buffer(get_matrix_diffcache(T, :xub_outs, n, d + 1), TE)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    _xub_nll(l, view(outs, :, 1), view(outs, :, 2:d + 1), sig2, θ, TE)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedXUniformBanded{T},
        p::AbstractVector{TD}) where {T,TD}
    l = pl.likelihood
    _check_grad_dims(grad, l, pl.model, p)
    nmod = n_param(pl.model)
    n, d = size(l.X)
    cov = l.cov
    b = cov.h
    ns = Int(l.intrinsic_scatter)
    suffix = _xub_suffix(l, nmod, p)
    suffix === nothing && (fill!(grad, zero(TD)); return _worst_loss(TD, T))
    sig2, θ = suffix

    outs = get_buffer(get_matrix_diffcache(T, :xub_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    f = view(outs, :, 1)
    g = view(outs, :, 2:d + 1)

    B = get_buffer(get_matrix_diffcache(T, :xub_B, b + 1, n), TD)
    Zb = get_buffer(get_matrix_diffcache(T, :xub_Z, b + 1, n), TD)
    u = get_buffer(get_vector_diffcache(T, :xub_u, n), TD)
    _xub_build!(B, u, l, f, g, sig2, θ)
    if !_band_ldlt!(B, b, n)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    logdet, quad = _band_logdet_quadratic!(u, B, b, n)
    nll = T(1/2) * (logdet + n * log(2 * T(pi)) + quad)
    if isnan(nll) || isinf(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    _band_backsolve!(u, B, b, n)                 # u = v = C⁻¹ r
    _band_selected_inverse!(Zb, B, b, n)

    λf = get_buffer(get_vector_diffcache(T, :xub_lf, n), TD)
    λg = get_buffer(get_matrix_diffcache(T, :xub_lg, n, d), TD)
    fill!(λg, zero(TD))
    fill!(grad, zero(TD))
    dsig = zero(TD)
    @inbounds for i in 1:n
        λf[i] = u[i]
        for r in 0:b
            i + r <= n || break
            G = _band_G(Zb, u, i + r, i)
            r == 0 && (dsig += G)
            _banded_D_adjoint!(λg, cov, g, i, r, G)
            for (q, P) in enumerate(l.free)
                _banded_D_adjoint!(λg, P, g, i, r, θ[q] * G)
                grad[nmod + ns + q] += G * _banded_D(P, g, i, r)
            end
        end
    end
    l.intrinsic_scatter && (grad[nmod + 1] = dsig)
    _grad_seeded!(grad, pl.code, l.X, p, pl.cols, λf, λg)
    nll
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::XUniformBandedLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    evaluate_nll_grad!(grad, prepare(l, model), p)
end

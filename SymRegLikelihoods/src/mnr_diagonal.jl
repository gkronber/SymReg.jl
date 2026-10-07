# MNR (marginalized normal regression, arXiv:2309.00948) likelihood with
# diagonal error covariance: independent x and y errors, given as standard
# deviations, and no correlation between observations.  The special case of
# `MNRBandedLikelihood` with zero `Σxy` and diagonal `Σyy`; the system
# factorizes into one (d+1) x (d+1) block per observation, inverted in closed form.
#
# θ = [β; σ²; mu_1; …; mu_d; w_1²; …; w_d²], without σ² when
# `intrinsic_scatter = false`.  Each latent input has the prior
# x_k ~ N(mu_k, w_k²).  The model f and its derivatives a_k = ∂f/∂x_k are
# evaluated at `xobs`;  With per-observation quantities
#   zx_k = xobs_k - mu_k             D_k = w_k² + xerr2_k
#   s2   = yerr2 + σ²                S   = s2 + Σ_k w_k² a_k² xerr2_k/D_k
#   α    = -Σ_k w_k² a_k zx_k/D_k    zy  = f - Σ_k a_k zx_k - yobs
#   c    = α - zy = yobs - f + Σ_k a_k zx_k xerr2_k/D_k
# (the last form is used: α and the a_k zx_k in zy cancel for a wide prior,
# w_k² ≫ xerr2_k, which costs digits in the derivatives of the NLL)
# the marginalized NLL is
#   nll = 1/2 [ Σ_k log D_k + log S + Σ_k zx_k²/D_k + c²/S ] + (d+1)/2 log(2π)
# For d == 1 it equals ROXY's `nll_mnr`, the MNR
# likelihood with a Gaussian prior on the true x.  The uniform-prior "unif"
# variant is `XUniformDiagonalLikelihood`.

"""
    MNRDiagonalLikelihood(xobs, xerr, yobs, yerr; intrinsic_scatter = true)

Marginalized normal regression likelihood (MNR, arXiv:2309.00948) for
independent errors `xerr` (one column per input) and `yerr`, given as standard
deviations.  The true inputs are marginalized under Gaussian priors
`N(mu_k, w_k²)`; the likelihood parameters are `[σ²; mu_1, …, mu_d; w_1², …, w_d²]`,
without the intrinsic scatter `σ²` when `intrinsic_scatter = false`.
"""
struct MNRDiagonalLikelihood{T} <: AbstractLikelihood{T}
    xobs::Matrix{T}   # (n x d) observed x; also the model evaluation point
    xerr2::Matrix{T}  # (n x d) xerr^2
    yobs::Vector{T}
    yerr2::Vector{T}
    intrinsic_scatter::Bool

    # `xerr` and `yerr` are standard deviations
    function MNRDiagonalLikelihood(xobs::AbstractMatrix{T}, xerr::AbstractMatrix{T},
            yobs::AbstractVector{T}, yerr::AbstractVector{T};
            intrinsic_scatter::Bool = true) where {T}
        n, d = size(xobs)
        size(xerr) == (n, d) ||
            throw(DimensionMismatch("MNR x matrices must have matching size"))
        (length(yobs) == n && length(yerr) == n) ||
            throw(DimensionMismatch("MNR y vectors must have length = size(xobs, 1)"))
        new{T}(Matrix{T}(xobs), Matrix{T}(xerr .^ 2),
               Vector{T}(yobs), Vector{T}(yerr .^ 2), intrinsic_scatter)
    end
    function MNRDiagonalLikelihood(xobs::AbstractVector{T}, xerr::AbstractVector{T},
            yobs::AbstractVector{T}, yerr::AbstractVector{T};
            intrinsic_scatter::Bool = true) where {T}
        MNRDiagonalLikelihood(reshape(xobs, :, 1), reshape(xerr, :, 1), yobs, yerr;
                              intrinsic_scatter)
    end
end

n_observations(l::MNRDiagonalLikelihood) = size(l.xobs, 1)
has_intrinsic_scatter(l::MNRDiagonalLikelihood) = l.intrinsic_scatter
# [sigma²], mu_k, w_k²
n_likelihood_params(l::MNRDiagonalLikelihood) = Int(l.intrinsic_scatter) + 2 * size(l.xobs, 2)
positive_params(l::MNRDiagonalLikelihood) = _mnr_positive_params(Int(l.intrinsic_scatter), size(l.xobs, 2))

# The variances among `[sigma² (with intrinsic scatter); mu_1..mu_d; w_1²..w_d²]`.
_mnr_positive_params(ns::Int, d::Int) = Iterators.flatten((1:ns, ns + d + 1:ns + 2d))

# Variances must be non-negative and below 1/eps.
_mnr_valid(sig2, w2, ::Type{TD}) where {TD} =
    inv(sig2) > eps(TD) && all(w2k -> inv(w2k) > eps(TD), w2)

# NLL of one observation.  Writes the adjoints w.r.t. a_k into `λg` and
# ∂nll/∂mu_k, ∂nll/∂w_k² into `dmu`, `dw2` of `scratch`; returns
# (nll, ∂nll/∂f, ∂nll/∂σ²).
function _mnr_row!(λg, scratch, f, gvals, xobsv, xerr2v, yerr2, yobs,
        sig2, w2, mu, ::Type{T}) where {T}
    a, zx, D, dmu, dw2 = scratch
    d = length(gvals)
    G = zero(T); Tc = zero(T); B = zero(T); LgD = zero(T)
    @inbounds for k in 1:d
        xerr2 = xerr2v[k]; w2k = w2[k]
        ak = gvals[k]
        zx[k] = xobsv[k] - mu[k]
        D[k] = w2k + xerr2
        a[k] = ak
        invD = inv(D[k])
        G += w2k * ak * ak * xerr2 * invD
        Tc += ak * zx[k] * xerr2 * invD
        B += zx[k] * zx[k] * invD
        LgD += log(D[k])
    end
    s2 = yerr2 + sig2
    S = s2 + G
    c = yobs - f + Tc
    invS = inv(S)
    nll = T(1 / 2) * (LgD + log(S) + B + c * c * invS) + T((d + 1) / 2) * log(2 * T(pi))
    dsig2 = (invS - c * c * invS * invS) / 2
    @inbounds for k in 1:d
        ak = a[k]; zxk = zx[k]; Dk = D[k]; xerr2 = xerr2v[k]; w2k = w2[k]
        invD = inv(Dk); invD2 = invD * invD
        λg[k] = (xerr2 * invD) * (w2k * ak + c * zxk - c * c * w2k * ak * invS) * invS
        dmu[k] = -zxk * invD - c * ak * xerr2 * invD * invS
        dSdw2 = ak * ak * xerr2 * xerr2 * invD2
        dcdw2 = -ak * zxk * xerr2 * invD2
        dw2[k] = (invD - zxk * zxk * invD2 +
                  dSdw2 * (invS - c * c * invS * invS) +
                  2 * c * dcdw2 * invS) / 2
    end
    (nll, -c * invS, dsig2)
end

# σ² (zero without intrinsic scatter) and views of mu and w² in `p`.
function _smnr_suffix(l::MNRDiagonalLikelihood, p::AbstractVector, nmod::Int)
    d = size(l.xobs, 2)
    ns = Int(l.intrinsic_scatter)
    (_scatter2(l, p, nmod), view(p, nmod + ns + 1:nmod + ns + d),
     view(p, nmod + ns + d + 1:nmod + ns + 2d))
end

# Total NLL from the model values `f` and the input partials `gmat` (n x d).
function _smnr_nll(l::MNRDiagonalLikelihood{T}, f, gmat, sig2, mu, w2,
        ::Type{TD}) where {T,TD}
    n, d = size(l.xobs)
    λg = get_buffer(get_vector_diffcache(T, :smnr_lambdag, d), TD)
    a = get_buffer(get_vector_diffcache(T, :smnr_a, d), TD)
    zx = get_buffer(get_vector_diffcache(T, :smnr_zx, d), TD)
    D = get_buffer(get_vector_diffcache(T, :smnr_D, d), TD)
    dmu = get_buffer(get_vector_diffcache(T, :smnr_dmu, d), TD)
    dw2 = get_buffer(get_vector_diffcache(T, :smnr_dw2, d), TD)
    scratch = (a, zx, D, dmu, dw2)
    total = zero(TD)
    for i in 1:n
        row = _mnr_row!(λg, scratch, f[i], view(gmat, i, :), view(l.xobs, i, :),
                        view(l.xerr2, i, :), l.yerr2[i], l.yobs[i], sig2, w2, mu, TD)[1]
        total += isnan(row) || isinf(row) ? _worst_loss(TD, T) : row
    end
    isnan(total) || isinf(total) ? _worst_loss(TD, T) : total
end

function evaluate_nll(l::MNRDiagonalLikelihood{T}, model::Model, p::AbstractVector) where {T}
    n, d = size(l.xobs)
    nmod = n_param(model)
    _check_nll_dims(l, model, p)
    TD = eltype(p)
    sig2, mu, w2 = _smnr_suffix(l, p, nmod)
    _mnr_valid(sig2, w2, TD) || return _worst_loss(TD, T)
    f = get_buffer(get_vector_diffcache(T, :smnr_f, n), TD)
    jacx = get_buffer(get_matrix_diffcache(T, :smnr_jacx, n, d), TD)
    interpret_jac!(f, nothing, jacx, model, l.xobs, p)
    _smnr_nll(l, f, jacx, sig2, mu, w2, TD)
end

struct PreparedMNRDiagonal{T} <: PreparedLikelihood{T}
    likelihood::MNRDiagonalLikelihood{T}
    model::Model{T}
    dcode::Vector{Instruction{T}}  # differentiate(model.code, 1:d); f at cols[1], df/dx_k at cols[1+k]
    cols::Vector{Int}              # [fidx; dfidxs...]: positions in `dcode`
end

function prepare(l::MNRDiagonalLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.xobs, 2))
    PreparedMNRDiagonal(l, model, dcode, [Int(fidx); Int.(dfidxs)])
end

function evaluate_nll(pl::PreparedMNRDiagonal{T}, p::AbstractVector) where {T}
    l = pl.likelihood
    n, d = size(l.xobs)
    nmod = n_param(pl.model)
    _check_nll_dims(l, pl.model, p)
    TD = eltype(p)
    sig2, mu, w2 = _smnr_suffix(l, p, nmod)
    _mnr_valid(sig2, w2, TD) || return _worst_loss(TD, T)
    outs = get_buffer(get_matrix_diffcache(T, :smnr_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.dcode, l.xobs, p, pl.cols)
    _smnr_nll(l, view(outs, :, 1), view(outs, :, 2:d + 1), sig2, mu, w2, TD)
end

# One forward pass over the differentiated code gives f and a_k = ∂f/∂x_k; the
# row adjoints w.r.t. f and a_k seed the reverse sweep at `cols[1]` and
# `cols[1+k]`.
function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedMNRDiagonal{T},
        p::AbstractVector{TD}) where {T,TD}
    l = pl.likelihood
    n, d = size(l.xobs)
    nmod = n_param(pl.model)
    ns = Int(l.intrinsic_scatter)
    _check_grad_dims(grad, l, pl.model, p)
    sig2, mu, w2 = _smnr_suffix(l, p, nmod)
    _mnr_valid(sig2, w2, TD) || (fill!(grad, zero(TD)); return _worst_loss(TD, T))

    outs = get_buffer(get_matrix_diffcache(T, :smnr_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.dcode, l.xobs, p, pl.cols)

    a = get_buffer(get_vector_diffcache(T, :smnr_a, d), TD)
    zx = get_buffer(get_vector_diffcache(T, :smnr_zx, d), TD)
    D = get_buffer(get_vector_diffcache(T, :smnr_D, d), TD)
    dmu = get_buffer(get_vector_diffcache(T, :smnr_dmu, d), TD)
    dw2 = get_buffer(get_vector_diffcache(T, :smnr_dw2, d), TD)
    λf = get_buffer(get_vector_diffcache(T, :smnr_lf, n), TD)
    λg = get_buffer(get_matrix_diffcache(T, :smnr_lg, n, d), TD)
    scratch = (a, zx, D, dmu, dw2)
    fill!(grad, zero(TD))
    total = zero(TD)
    for i in 1:n
        row, lf, dsig2 = _mnr_row!(view(λg, i, :), scratch,
            outs[i, 1], view(outs, i, 2:d + 1),
            view(l.xobs, i, :), view(l.xerr2, i, :), l.yerr2[i], l.yobs[i],
            sig2, w2, mu, TD)
        total += isnan(row) || isinf(row) ? _worst_loss(TD, T) : row
        λf[i] = lf
        l.intrinsic_scatter && (grad[nmod + 1] += dsig2)
        @inbounds for k in 1:d
            grad[nmod + ns + k] += dmu[k]
            grad[nmod + ns + d + k] += dw2[k]
        end
    end
    if isnan(total) || isinf(total)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    _grad_seeded!(grad, pl.dcode, l.xobs, p, pl.cols, λf, λg)
    total
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::MNRDiagonalLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    evaluate_nll_grad!(grad, prepare(l, model), p)
end


# MNR (marginalized normal regression, arXiv:2309.00948) likelihood for error
# covariances with a general band structure in time (`BandedCovariance`, see
# banded_covariance.jl): Σxx, Σxy and Σyy may couple observations i and j with
# |i - j| <= h, and Σxx may couple the input variables.  This covers derivatives
# from any finite-difference stencil (forward differences paired with x_i or with
# the midpoint (x_i + x_{i+1})/2, central differences, …).  θ = [β; σ²; mu_1; …;
# mu_d; w_1²; …; w_d²], without σ² when `intrinsic_scatter = false`, as for the
# other MNR likelihoods.
#
# The latent inputs are ordered variable-major (`vec(X)`).  With the input
# Jacobian J (n x n*d; J[i, (k-1)*n+i] = g[i,k] = ∂f_i/∂X[i,k]), the
# transformation T = [I 0; -J I] (det T = 1) of the MNR system M, z (see
# mnr_banded.jl) gives
#   T M T' = [Σxx + W   E;  E'   D + σ²I],   T z = [μ - vec(X); f - y]
#   E = Σxy - Σxx J',   D = Σyy + J Σxx J' - J Σxy - (J Σxy)'
# in which W appears only on the diagonal of the latent block, so nothing
# cancels for a wide latent prior (w² → ∞, see mnr_dense.jl).  Ordered by time,
# (x_1,i, …, x_d,i, y_i) for i = 1, …, n, T M T' is a band matrix with half
# bandwidth b = (h + 1)(d + 1) - 1, and with its LDLᵀ factorization (band, no
# pivoting) and u = L⁻¹ T z
#   NLL = ½ (Σ_c log D_c + N log 2π + Σ_c u_c² / D_c),   N = n (d + 1)
# in O(n (d + 1)³ (h + 1)²).  T M T' is positive definite for a valid covariance;
# a non-positive pivot gives the worst loss.
#
# Gradient: with Z = (T M T')⁻¹ and v = Z T z,
#   dNLL = ½ tr(Z dN) - ½ v' dN v + v' d(T z),   N = T M T'
# The entries of Z within the band follow from the factors by Takahashi's
# recurrence, so the gradient w.r.t. every band entry of N, G_pq = Z_pq - v_p v_q
# (halved on the diagonal), costs as much as the factorization.  N depends on w²
# (diagonal of the latent block), σ² (diagonal of D) and the input partials g
# (E linearly, D quadratically); T z on mu and f.  The adjoints w.r.t. f and g
# seed the reverse sweep of the interpreter, as for `MNRBandedLikelihood`.

"""
    MNRGeneralBandedLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = true)
    MNRGeneralBandedLikelihood(X, y, cov::BandedCovariance; intrinsic_scatter = true)
    MNRGeneralBandedLikelihood(X, y, patterns, weights; intrinsic_scatter = true)

Marginalized normal regression likelihood (MNR, arXiv:2309.00948) for error
covariances of the inputs `X` (ordered variable-major, `vec(X)`) and the outputs
`y` that are banded in time: every entry coupling observations i and j must have
|i - j| <= h for some small h, which is detected from the matrices.  The
covariance can also be given as a [`BandedCovariance`](@ref), or as the weighted
sum Σ_r weights[r] patterns[r] of known patterns (noise variances `weights`).
Unlike [`MNRBandedLikelihood`](@ref), Σxx need not be diagonal and Σxy may have
bands on both sides of the diagonal, so it covers derivatives from any
finite-difference stencil.  The likelihood parameters are those of
[`MNRDiagonalLikelihood`](@ref).
"""
struct MNRGeneralBandedLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    cov::BandedCovariance{T}
    intrinsic_scatter::Bool
end

function MNRGeneralBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        cov::BandedCovariance{T}; intrinsic_scatter::Bool = true) where {T}
    n, d = size(X, 1), size(X, 2)
    length(y) == n || throw(DimensionMismatch("length(y) != size(X,1)"))
    (cov.n, cov.d) == (n, d) || throw(DimensionMismatch("MNR covariance dimensions do not match data"))
    MNRGeneralBandedLikelihood{T}(Matrix{T}(X), Vector{T}(y), cov, intrinsic_scatter)
end

function MNRGeneralBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        Σxx::AbstractMatrix{T}, Σxy::AbstractMatrix{T}, Σyy::AbstractMatrix{T};
        intrinsic_scatter::Bool = true) where {T}
    n, d = size(X, 1), size(X, 2)
    (size(Σxx) == (n * d, n * d) && size(Σxy) == (n * d, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("MNR covariance dimensions do not match data"))
    MNRGeneralBandedLikelihood(X, y, BandedCovariance(Σxx, Σxy, Σyy); intrinsic_scatter)
end

function MNRGeneralBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        patterns::AbstractVector{<:BandedCovariance{T}}, weights::AbstractVector;
        intrinsic_scatter::Bool = true) where {T}
    length(patterns) == length(weights) >= 1 ||
        throw(ArgumentError("one weight per pattern is needed"))
    MNRGeneralBandedLikelihood(X, y, sum(weights .* patterns); intrinsic_scatter)
end

MNRGeneralBandedLikelihood(x::AbstractVector, y::AbstractVector, args...; kwargs...) =
    MNRGeneralBandedLikelihood(reshape(x, :, 1), y, args...; kwargs...)

n_observations(l::MNRGeneralBandedLikelihood) = size(l.X, 1)
has_intrinsic_scatter(l::MNRGeneralBandedLikelihood) = l.intrinsic_scatter
# [sigma²], mu_k, w_k²
n_likelihood_params(l::MNRGeneralBandedLikelihood) = Int(l.intrinsic_scatter) + 2 * size(l.X, 2)
positive_params(l::MNRGeneralBandedLikelihood) = _mnr_positive_params(Int(l.intrinsic_scatter), size(l.X, 2))

# Fill the lower band of T M T' (time-interleaved order) into `B`, with
# B[p - q + 1, q] = (T M T')[p, q] for p >= q, and T z into `u`.
function _mnr_gb_build!(B, u, l::MNRGeneralBandedLikelihood, f, g, sig2, mu, w2)
    n, d = size(l.X)
    cov = l.cov
    h = cov.h
    m = d + 1
    sxx, sxy = cov.sxx, cov.sxy
    fill!(B, zero(eltype(B)))
    @inbounds for i in 1:n
        oi = (i - 1) * m              # position of x_1,i is oi + 1, of y_i is oi + m
        for k in 1:d
            u[oi + k] = mu[k] - l.X[i, k]
        end
        u[oi + m] = f[i] - l.y[i]
        for r in 0:h
            j = i + r
            j <= n || break
            oj = (j - 1) * m
            c = r + h + 1             # band index of (i, j)
            cm = h + 1 - r            # band index of (j, i)
            # latent block: Σxx + W
            for k in 1:d, ll in 1:d
                (r == 0 && ll > k) && continue
                v = sxx[k, ll, c, i]
                (r == 0 && ll == k) && (v += w2[k])
                p, q = r == 0 ? (oi + k, oi + ll) : (oj + ll, oi + k)   # p >= q
                B[p - q + 1, q] = v
            end
            # E[(k, i), j] = Σxy[(k, i), j] - Σ_l Σxx[(k, i), (l, j)] g[j, l]
            for k in 1:d
                v = sxy[k, c, i]
                for ll in 1:d
                    v -= sxx[k, ll, c, i] * g[j, ll]
                end
                p, q = oj + m, oi + k
                B[p - q + 1, q] = v
            end
            # E[(l, j), i] for j > i
            if r > 0
                for ll in 1:d
                    v = sxy[ll, cm, j]
                    for k in 1:d
                        v -= sxx[ll, k, cm, j] * g[i, k]
                    end
                    p, q = oj + ll, oi + m
                    B[p - q + 1, q] = v
                end
            end
            # D[i, j] + σ² δ_ij
            v = _banded_D(cov, g, i, r)
            r == 0 && (v += sig2)
            p, q = oj + m, oi + m
            B[p - q + 1, q] = v
        end
    end
    nothing
end

function _mnr_gb_nll(l::MNRGeneralBandedLikelihood{T}, f, g, sig2, mu, w2, ::Type{TE}) where {T,TE}
    n, d = size(l.X)
    m = d + 1
    N = n * m
    b = (l.cov.h + 1) * m - 1
    B = get_buffer(get_matrix_diffcache(T, :mnrgb_B, b + 1, N), TE)
    u = get_buffer(get_vector_diffcache(T, :mnrgb_u, N), TE)
    _mnr_gb_build!(B, u, l, f, g, sig2, mu, w2)
    _band_ldlt!(B, b, N) || return _worst_loss(TE, T)
    logdet, quad = _band_logdet_quadratic!(u, B, b, N)
    nll = T(1/2) * (logdet + N * log(2 * T(pi)) + quad)
    (isnan(nll) || isinf(nll)) ? _worst_loss(TE, T) : nll
end

# σ² (zero without intrinsic scatter) and views of mu and w² in `p`, or
# `nothing` when the variances are invalid.
function _mnr_gb_suffix(l::MNRGeneralBandedLikelihood, nmod::Int, p::AbstractVector)
    d = size(l.X, 2)
    ns = Int(l.intrinsic_scatter)
    sig2 = _scatter2(l, p, nmod)
    mu = view(p, nmod + ns + 1:nmod + ns + d)
    w2 = view(p, nmod + ns + d + 1:nmod + ns + 2d)
    _mnr_valid(sig2, w2, eltype(p)) || return nothing
    (sig2, mu, w2)
end

function evaluate_nll(l::MNRGeneralBandedLikelihood{T}, model::Model, p::AbstractVector) where {T}
    _check_nll_dims(l, model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _mnr_gb_suffix(l, n_param(model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, mu, w2 = suffix
    f = get_buffer(get_vector_diffcache(T, :mnrgb_f, n), TE)
    jacx = get_buffer(get_matrix_diffcache(T, :mnrgb_jacx, n, d), TE)
    interpret_jac!(f, nothing, jacx, model, l.X, p)
    _mnr_gb_nll(l, f, jacx, sig2, mu, w2, TE)
end

struct PreparedMNRGeneralBanded{T,L<:MNRGeneralBandedLikelihood{T}} <: PreparedLikelihood{T}
    likelihood::L
    model::Model{T}
    code::Vector{Instruction{T}}   # f and all d input partials in one code vector
    cols::Vector{Int}              # [fidx; dfidxs...]: positions in `code`
end

function prepare(l::MNRGeneralBandedLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.X, 2))
    PreparedMNRGeneralBanded(l, model, dcode, [Int(fidx); Int.(dfidxs)])
end

# Uses the same differentiated code as the gradient, so that both agree.
function evaluate_nll(pl::PreparedMNRGeneralBanded{T}, p::AbstractVector) where {T}
    l = pl.likelihood
    _check_nll_dims(l, pl.model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _mnr_gb_suffix(l, n_param(pl.model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, mu, w2 = suffix
    outs = get_buffer(get_matrix_diffcache(T, :mnrgb_outs, n, d + 1), TE)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    _mnr_gb_nll(l, view(outs, :, 1), view(outs, :, 2:d + 1), sig2, mu, w2, TE)
end

# Accumulate the adjoints of the band entries of N into λg (n x d), dw2 (d) and
# the derivative w.r.t. σ²; walks the entries as `_mnr_gb_build!` does.
function _mnr_gb_adjoints!(λg, dw2, l::MNRGeneralBandedLikelihood, Zb, v, g)
    n, d = size(l.X)
    cov = l.cov
    h = cov.h
    m = d + 1
    sxx = cov.sxx
    fill!(λg, zero(eltype(λg)))
    fill!(dw2, zero(eltype(dw2)))
    dsig = zero(eltype(λg))
    @inbounds for i in 1:n
        oi = (i - 1) * m
        for k in 1:d
            dw2[k] += _band_G(Zb, v, oi + k, oi + k)
        end
        for r in 0:h
            j = i + r
            j <= n || break
            oj = (j - 1) * m
            c = r + h + 1
            cm = h + 1 - r
            # E[(k, i), j] = Σxy[(k, i), j] - Σ_l Σxx[(k, i), (l, j)] g[j, l]
            for k in 1:d
                G = _band_G(Zb, v, oj + m, oi + k)
                for ll in 1:d
                    λg[j, ll] -= G * sxx[k, ll, c, i]
                end
            end
            # E[(l, j), i] for j > i
            if r > 0
                for ll in 1:d
                    G = _band_G(Zb, v, oj + ll, oi + m)
                    for k in 1:d
                        λg[i, k] -= G * sxx[ll, k, cm, j]
                    end
                end
            end
            # D[i, j]
            G = _band_G(Zb, v, oj + m, oi + m)
            r == 0 && (dsig += G)
            _banded_D_adjoint!(λg, cov, g, i, r, G)
        end
    end
    dsig
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedMNRGeneralBanded{T},
        p::AbstractVector{TD}) where {T,TD}
    l = pl.likelihood
    _check_grad_dims(grad, l, pl.model, p)
    nmod = n_param(pl.model)
    n, d = size(l.X)
    m = d + 1
    N = n * m
    b = (l.cov.h + 1) * m - 1
    ns = Int(l.intrinsic_scatter)
    suffix = _mnr_gb_suffix(l, nmod, p)
    suffix === nothing && (fill!(grad, zero(TD)); return _worst_loss(TD, T))
    sig2, mu, w2 = suffix

    outs = get_buffer(get_matrix_diffcache(T, :mnrgb_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    f = view(outs, :, 1)
    g = view(outs, :, 2:d + 1)

    B = get_buffer(get_matrix_diffcache(T, :mnrgb_B, b + 1, N), TD)
    Zb = get_buffer(get_matrix_diffcache(T, :mnrgb_Z, b + 1, N), TD)
    u = get_buffer(get_vector_diffcache(T, :mnrgb_u, N), TD)
    _mnr_gb_build!(B, u, l, f, g, sig2, mu, w2)
    if !_band_ldlt!(B, b, N)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    logdet, quad = _band_logdet_quadratic!(u, B, b, N)
    nll = T(1/2) * (logdet + N * log(2 * T(pi)) + quad)
    if isnan(nll) || isinf(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    _band_backsolve!(u, B, b, N)                 # u = v = N⁻¹ T z
    _band_selected_inverse!(Zb, B, b, N)

    λf = get_buffer(get_vector_diffcache(T, :mnrgb_lf, n), TD)
    λg = get_buffer(get_matrix_diffcache(T, :mnrgb_lg, n, d), TD)
    dw2 = get_buffer(get_vector_diffcache(T, :mnrgb_dw2, d), TD)
    dsig = _mnr_gb_adjoints!(λg, dw2, l, Zb, u, g)
    fill!(grad, zero(TD))
    @inbounds for i in 1:n
        oi = (i - 1) * m
        λf[i] = u[oi + m]
        for k in 1:d
            grad[nmod + ns + k] += u[oi + k]       # ∂/∂mu_k
        end
    end
    @inbounds for k in 1:d
        grad[nmod + ns + d + k] = dw2[k]
    end
    l.intrinsic_scatter && (grad[nmod + 1] = dsig)
    _grad_seeded!(grad, pl.code, l.X, p, pl.cols, λf, λg)
    nll
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::MNRGeneralBandedLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    evaluate_nll_grad!(grad, prepare(l, model), p)
end

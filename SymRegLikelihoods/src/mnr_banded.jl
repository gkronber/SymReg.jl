# MNR (marginalized normal regression, arXiv:2309.00948) likelihood with banded
# error covariance: diagonal `Σxx`, per-variable bidiagonal `Σxy` and symmetric
# tridiagonal `Σyy`.  θ = [β; σ²; mu_1; …; mu_d; w_1²; …; w_d²],
# without σ² when `intrinsic_scatter = false`.
#
# The latent inputs are ordered variable-major (`vec(X)`), so `Σxx` has d
# blocks of size n x n.  With W the diagonal matrix holding each w_k² once per
# observation and J the input Jacobian (n x n*d; J[i, (k-1)*n+i] = ∂f_i/∂X[i,k]):
#   M = [Σxx + W          Σxy + WJ';
#        Σxy' + JW        Σyy + JWJ' + σ²I]
#   z = [μ - vec(X); f + J(μ - vec(X)) - y]
# where μ stacks each mu_k n times.  M is never formed and the cost is O(n·d);
# `MNRDenseLikelihood` handles general covariances.

# Concrete band storage avoids dynamic dispatch in the fast path.  For d > 1,
# `Σxy` is not a `Bidiagonal`.
"""
    MNRBandedLikelihood(X, y, Σxx::Diagonal, Σxy, Σyy::SymTridiagonal; intrinsic_scatter = true)

Marginalized normal regression likelihood (MNR, arXiv:2309.00948) for banded
error covariances: diagonal `Σxx`, bidiagonal blocks of `Σxy` per input and
tridiagonal `Σyy`, with the inputs `X` (or a single input vector) ordered
variable-major.  The likelihood parameters are those of
[`MNRDiagonalLikelihood`](@ref); the cost is linear in the number of observations.
"""
struct MNRBandedLikelihood{T,SXY<:AbstractMatrix{T}} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    Σxx::Diagonal{T,Vector{T}}   # (n*d) x (n*d), block-diagonal over variables
    Σxy::SXY                     # (n*d) x n, bidiagonal-per-variable blocks
    Σyy::SymTridiagonal{T,Vector{T}}
    # Bands of `Σxy` per variable block k: main diagonal `sxy0[:, k]` and the
    # off-diagonal `sxy1[i, k] = block[i, i + sxyside[k]]`, where `sxyside[k]`
    # is -1 (below), +1 (above) or 0 (diagonal block).
    sxy0::Matrix{T}       # n x d
    sxy1::Matrix{T}       # n x d
    sxyside::Vector{Int}  # d
    intrinsic_scatter::Bool
end

# Each n x n block of `Σxy` must be bidiagonal on one side, or the Schur
# complement below would not be tridiagonal.
function _mnr_split_sxy(Σxy::AbstractMatrix{T}, n::Int, d::Int) where {T}
    sxy0 = zeros(T, n, d)
    sxy1 = zeros(T, n, d)
    sides = zeros(Int, d)
    for k in 1:d
        off = (k - 1) * n
        side = 0
        for j in 1:n, i in 1:n
            v = Σxy[off + i, j]
            (iszero(v) || j == i) && continue
            abs(j - i) == 1 || throw(ArgumentError(
                "MNRBandedLikelihood requires each n x n block of Σxy to be bidiagonal, but " *
                "block $k has a nonzero entry at ($i, $j), $(abs(j - i)) places off the " *
                "diagonal. Use MNRDenseLikelihood for a general Σxy."))
            if side == 0
                side = j - i
            elseif side != j - i
                throw(ArgumentError(
                    "MNRBandedLikelihood requires each n x n block of Σxy to be bidiagonal on " *
                    "one side, but block $k has entries both above and below the " *
                    "diagonal (at ($i, $j) among them). Use MNRDenseLikelihood for a " *
                    "general Σxy."))
            end
        end
        sides[k] = side
        for i in 1:n
            sxy0[i, k] = Σxy[off + i, i]
            j = i + side
            (side != 0 && 1 <= j <= n) && (sxy1[i, k] = Σxy[off + i, j])
        end
    end
    sxy0, sxy1, sides
end

function MNRBandedLikelihood(x::AbstractVector{T}, y::AbstractVector{T},
        Σxx::Diagonal{T}, Σxy::AbstractMatrix{T}, Σyy::SymTridiagonal{T};
        intrinsic_scatter::Bool = true) where {T}
    MNRBandedLikelihood(reshape(x, :, 1), y, Σxx, Σxy, Σyy; intrinsic_scatter)
end
function MNRBandedLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        Σxx::Diagonal{T}, Σxy::AbstractMatrix{T}, Σyy::SymTridiagonal{T};
        intrinsic_scatter::Bool = true) where {T}
    n, d = size(X, 1), size(X, 2)
    nd = n * d
    length(y) == n ||
        throw(DimensionMismatch("length(y) != size(X,1)"))
    (size(Σxx) == (nd, nd) && size(Σxy) == (nd, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("MNR covariance dimensions do not match data"))
    sxy0, sxy1, sides = _mnr_split_sxy(Σxy, n, d)
    MNRBandedLikelihood{T,typeof(Σxy)}(
        Matrix{T}(X), Vector{T}(y),
        Diagonal(Vector{T}(Σxx.diag)),
        Σxy,
        SymTridiagonal(Vector{T}(Σyy.dv), Vector{T}(Σyy.ev)),
        sxy0, sxy1, sides, intrinsic_scatter)
end

n_observations(l::MNRBandedLikelihood) = size(l.X, 1)
has_intrinsic_scatter(l::MNRBandedLikelihood) = l.intrinsic_scatter
# [sigma²], mu_k, w_k²
n_likelihood_params(l::MNRBandedLikelihood) = Int(l.intrinsic_scatter) + 2 * size(l.X, 2)
positive_params(l::MNRBandedLikelihood) = _mnr_positive_params(Int(l.intrinsic_scatter), size(l.X, 2))

# ---------------------------------------------------------------------------
# The fast path: solving M without forming it
# ---------------------------------------------------------------------------
# A11 = Σxx + W is diagonal and is eliminated exactly, leaving the n x n Schur
# complement
#
#   S = Σyy + J W J' + σ²I - B' A11⁻¹ B,     B = Σxy + W J'
#
# S is symmetric tridiagonal: J W J' is diagonal (Σ_k w_k² g_ik²), and row
# (k-1)n+i of B has nonzeros only in columns i and i ± 1.  Then
#
#   log|det M| = Σ_a log|A11[a,a]| + Σ_i log|d_i|   (d from the LDL' of S)
#   z'M⁻¹z     = z1' A11⁻¹ z1 + r' S⁻¹ r,   r = z2 - B' A11⁻¹ z1
#
# and the entries of M⁻¹ needed for the gradient follow from the tridiagonal
# part of S⁻¹.  `log|det|` matches `logabsdet` of the dense likelihood.
#
# With B[a,i] = b0 = sxy0 + w² g and A11[a,a] = D = Σxx + w², the terms of S
# and r that carry the latent input variance w² cancel analytically:
#
#   w² g² - b0²/D = (w² g (g Σxx - 2 sxy0) - sxy0²) / D
#   g - b0/D      = (g Σxx - sxy0) / D =: c / D
#
# Evaluated as written on the left, both lose all digits when w² g² ≫ Σyy
# (e.g. w² = 1e13), and the NLL is then arbitrarily wrong (even -1e200), which
# an optimizer or a GP readily exploits.  The same c appears in the gradient.

# Fill the diagonal of A11 (`Dv`), the Schur complement S (`Sd`, `Se`), the
# latent vector `z1 = mu - x` and the reduced right-hand side
# `r = z2 - B' A11⁻¹ z1` (z2 = f + J z1 - y).  `g` is the n x d matrix of input
# adjoints.  Returns z1' A11⁻¹ z1.
function _mnr_fast_build!(Dv, Sd, Se, z1, r, l, w2, sig2, mu, f, g, ::Type{TE}) where {TE}
    n, d = size(l.X)
    Σxxd = l.Σxx.diag
    sxy0, sxy1, sides = l.sxy0, l.sxy1, l.sxyside

    @inbounds for i in 1:n
        Sd[i] = l.Σyy.dv[i] + sig2
        r[i] = f[i] - l.y[i]
    end
    @inbounds for m in 1:n-1
        Se[m] = l.Σyy.ev[m]
    end

    qx = zero(TE)
    @inbounds for k in 1:d
        off = (k - 1) * n
        w2k = w2[k]
        muk = mu[k]
        side = sides[k]
        for i in 1:n
            a = off + i
            gik = g[i, k]
            sxx = Σxxd[a]
            s0 = sxy0[i, k]
            Da = sxx + w2k
            Dv[a] = Da
            z1a = muk - l.X[i, k]
            z1[a] = z1a
            qx += z1a * z1a / Da
            # J W J' minus the Schur term, and g z1 - B[a,i] z1 / D, cancelled
            Sd[i] += (w2k * gik * (gik * sxx - 2 * s0) - s0 * s0) / Da
            r[i] += (gik * sxx - s0) * z1a / Da
            j = i + side
            if side != 0 && 1 <= j <= n
                b0 = s0 + w2k * gik              # B[a, i]
                b1 = sxy1[i, k]                  # B[a, j], the only other entry
                q1 = b1 / Da
                Sd[j] -= q1 * b1
                Se[min(i, j)] -= b0 * q1
                r[j] -= q1 * z1a
            end
        end
    end
    qx
end

# Unpivoted LDL' of a symmetric tridiagonal matrix, `ll[i] = L[i+1, i]` and
# `ld = diag(D)`.  Returns `false` for a zero or non-finite pivot.
function _sym_tridiag_ldlt!(ld, ll, Sd, Se)
    n = length(Sd)
    @inbounds begin
        ld[1] = Sd[1]
        for i in 1:n-1
            di = ld[i]
            (isfinite(di) && !iszero(di)) || return false
            ll[i] = Se[i] / di
            ld[i+1] = Sd[i+1] - ll[i] * Se[i]
        end
        (isfinite(ld[n]) && !iszero(ld[n])) || return false
    end
    true
end

# Solve S v = b from the LDL' factors.  `v` may alias `b`.  Returns b' S⁻¹ b,
# accumulated from the forward sweep u = L⁻¹ b as Σ u_i² / d_i.
function _sym_tridiag_solve!(v, ld, ll, b)
    n = length(ld)
    q = zero(eltype(v))
    @inbounds begin
        v[1] = b[1]
        for i in 2:n
            v[i] = b[i] - ll[i-1] * v[i-1]
        end
        for i in 1:n
            q += v[i] * v[i] / ld[i]
            v[i] = v[i] / ld[i]
        end
        for i in n-1:-1:1
            v[i] -= ll[i] * v[i+1]
        end
    end
    q
end

# Diagonal `Zd` and first off-diagonal `Z1` of Z = S⁻¹ by Takahashi's recurrence:
#   Z[i, i+1] = -ll[i] Z[i+1, i+1]
#   Z[i, i]   = 1/ld[i] + ll[i]² Z[i+1, i+1]
function _tridiag_selected_inverse!(Zd, Z1, ld, ll)
    n = length(ld)
    @inbounds begin
        Zd[n] = inv(ld[n])
        for i in n-1:-1:1
            znext = Zd[i+1]
            Z1[i] = -ll[i] * znext
            Zd[i] = inv(ld[i]) + ll[i] * ll[i] * znext
        end
    end
    nothing
end

# Z[i, j] for |i - j| == 1, from the stored first off-diagonal.
@inline _z_off(Z1, i, j) = @inbounds Z1[min(i, j)]

# σ² (zero without intrinsic scatter) and views of mu and w² in `p`, or
# `nothing` when the variances are invalid.
function _mnr_banded_suffix(l::MNRBandedLikelihood, nmod::Int, p::AbstractVector)
    d = size(l.X, 2)
    ns = Int(l.intrinsic_scatter)
    sig2 = _scatter2(l, p, nmod)
    mu = view(p, nmod + ns + 1:nmod + ns + d)
    w2 = view(p, nmod + ns + d + 1:nmod + ns + 2d)
    _mnr_valid(sig2, w2, eltype(p)) || return nothing
    (sig2, mu, w2)
end

# NLL from the model values `f` and the input partials `g` (n x d).
function _mnr_banded_nll(l::MNRBandedLikelihood{T}, f, g, sig2, mu, w2, ::Type{TE}) where {T,TE}
    n, d = size(l.X)
    nd = n * d
    ntot = nd + n

    Dv = get_vector_scratch_view(T, :mnrmv_D, nd, TE)
    Sd = get_vector_scratch_view(T, :mnrmv_Sd, n, TE)
    Se = get_vector_scratch_view(T, :mnrmv_Se, n - 1, TE)
    z1 = get_vector_scratch_view(T, :mnrmv_z1, nd, TE)
    v2 = get_vector_scratch_view(T, :mnrmv_v2, n, TE)
    ld = get_vector_scratch_view(T, :mnrmv_ld, n, TE)
    ll = get_vector_scratch_view(T, :mnrmv_ll, n - 1, TE)

    # v2: reduced right-hand side in, y block of the solution out
    qx = _mnr_fast_build!(Dv, Sd, Se, z1, v2, l, w2, sig2, mu, f, g, TE)
    _sym_tridiag_ldlt!(ld, ll, Sd, Se) || return _worst_loss(TE, T)
    qy = _sym_tridiag_solve!(v2, ld, ll, v2)

    nll = T(1/2) * (_mnr_logabsdet(Dv, ld) + ntot * log(2 * T(pi)) + qx + qy)
    (isnan(nll) || isinf(nll)) ? _worst_loss(TE, T) : nll
end

# log|det M| = log|det A11| + log|det S|
function _mnr_logabsdet(Dv, ld)
    acc = zero(eltype(Dv))
    @inbounds for a in eachindex(Dv)
        acc += log(abs(Dv[a]))
    end
    @inbounds for i in eachindex(ld)
        acc += log(abs(ld[i]))
    end
    acc
end

function evaluate_nll(l::MNRBandedLikelihood{T}, model::Model, p::AbstractVector) where {T}
    _check_nll_dims(l, model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _mnr_banded_suffix(l, n_param(model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, mu, w2 = suffix
    f = get_buffer(get_vector_diffcache(T, :mnrmv_f, n), TE)
    jacx = get_buffer(get_matrix_diffcache(T, :mnrmv_jacx, n, d), TE)
    interpret_jac!(f, nothing, jacx, model, l.X, p)
    _mnr_banded_nll(l, f, jacx, sig2, mu, w2, TE)
end

struct PreparedMNRBanded{T,L<:MNRBandedLikelihood{T}} <: PreparedLikelihood{T}
    likelihood::L
    model::Model{T}
    code::Vector{Instruction{T}}   # f and all d input partials in one code vector
    cols::Vector{Int}              # [fidx; dfidxs...]: positions in `code`
end

function prepare(l::MNRBandedLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.X, 2))
    PreparedMNRBanded(l, model, dcode, [Int(fidx); Int.(dfidxs)])
end

# Uses the same differentiated code as the gradient, so that both agree.
function evaluate_nll(pl::PreparedMNRBanded{T}, p::AbstractVector) where {T}
    l = pl.likelihood
    _check_nll_dims(l, pl.model, p)
    TE = eltype(p)
    n, d = size(l.X)
    suffix = _mnr_banded_suffix(l, n_param(pl.model), p)
    suffix === nothing && return _worst_loss(TE, T)
    sig2, mu, w2 = suffix
    outs = get_buffer(get_matrix_diffcache(T, :mnrmv_outs, n, d + 1), TE)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    _mnr_banded_nll(l, view(outs, :, 1), view(outs, :, 2:d + 1), sig2, mu, w2, TE)
end

# With v = M⁻¹z, E = M⁻¹ and a = (k-1)*n + i:
#   dNLL/df_i    = v[nd+i]
#   dNLL/dg_{ik} = w_k² (E[a, nd+i] + g[i,k] E[nd+i, nd+i])
#                  - w_k² (v[a] v[nd+i] + g[i,k] v[nd+i]²)
#                  + (mu_k - X[i,k]) v[nd+i]
#   dNLL/dw_k²   = ½ Σ_i (E[a,a] + 2 g E[a,nd+i] + g² E[nd+i,nd+i] - (v[a] + g v[nd+i])²)
#   dNLL/dσ²     = ½ Σ_i (E[nd+i, nd+i] - v[nd+i]²)
# The entries of E follow from
#   E22 = S⁻¹,  E12 = -A11⁻¹ B S⁻¹,  E11 = A11⁻¹ + A11⁻¹ B S⁻¹ B' A11⁻¹
# but are only needed in combinations in which the w² terms cancel (see the
# fast path above).  With c = g Σxx - sxy0 = g D - b0, b1 = B[a,j], Z = S⁻¹ and
# z1 = mu - x:
#   u = v[a] + g v[nd+i]                 = (z1 + c v[nd+i] - b1 v[nd+j]) / D
#   E[a,nd+i] + g E[nd+i,nd+i]           = (c Z[i,i] - b1 Z[i,j]) / D
#   E[a,a] + 2g E[a,nd+i] + g² E[nd+i,nd+i]
#                                        = 1/D + (c² Z[i,i] - 2c b1 Z[i,j] + b1² Z[j,j]) / D²
# so that v[a] = A11⁻¹(z1 - B v_y) is never formed.
function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedMNRBanded{T},
        p::AbstractVector{TD}) where {T,TD}
    l = pl.likelihood
    nmod = n_param(pl.model)
    n, d = size(l.X)
    nd = n * d
    ntot = nd + n
    ns = Int(l.intrinsic_scatter)
    _check_grad_dims(grad, l, pl.model, p)
    suffix = _mnr_banded_suffix(l, nmod, p)
    suffix === nothing && (fill!(grad, zero(TD)); return _worst_loss(TD, T))
    sig2, mu, w2 = suffix
    X = l.X

    outs = get_buffer(get_matrix_diffcache(T, :mnrmv_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.code, X, p, pl.cols)
    f = view(outs, :, 1)
    g = view(outs, :, 2:d + 1)

    Dv = get_vector_scratch_view(T, :mnrmv_D, nd, TD)
    Sd = get_vector_scratch_view(T, :mnrmv_Sd, n, TD)
    Se = get_vector_scratch_view(T, :mnrmv_Se, n - 1, TD)
    z1 = get_vector_scratch_view(T, :mnrmv_z1, nd, TD)
    v2 = get_vector_scratch_view(T, :mnrmv_v2, n, TD)
    ld = get_vector_scratch_view(T, :mnrmv_ld, n, TD)
    ll = get_vector_scratch_view(T, :mnrmv_ll, n - 1, TD)
    Zd = get_vector_scratch_view(T, :mnrmv_Zd, n, TD)
    Z1 = get_vector_scratch_view(T, :mnrmv_Z1, n - 1, TD)

    qx = _mnr_fast_build!(Dv, Sd, Se, z1, v2, l, w2, sig2, mu, f, g, TD)
    if !_sym_tridiag_ldlt!(ld, ll, Sd, Se)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    qy = _sym_tridiag_solve!(v2, ld, ll, v2)
    _tridiag_selected_inverse!(Zd, Z1, ld, ll)

    nll = T(1/2) * (_mnr_logabsdet(Dv, ld) + ntot * log(2 * T(pi)) + qx + qy)

    λf = get_buffer(get_vector_diffcache(T, :mnrmv_lf, n), TD)
    λg = get_buffer(get_matrix_diffcache(T, :mnrmv_lg, n, d), TD)
    fill!(grad, zero(TD))
    dsig = zero(TD)
    @inbounds for i in 1:n
        vy = v2[i]
        λf[i] = vy
        dsig += Zd[i] - vy * vy
    end
    Σxxd = l.Σxx.diag
    sxy0, sxy1, sides = l.sxy0, l.sxy1, l.sxyside
    @inbounds for k in 1:d
        off = (k - 1) * n
        w2k = w2[k]
        side = sides[k]
        for i in 1:n
            a = off + i
            gik = g[i, k]
            vy = v2[i]
            eyy = Zd[i]
            Da = Dv[a]
            z1a = z1[a]
            c = gik * Σxxd[a] - sxy0[i, k]
            j = i + side
            if side != 0 && 1 <= j <= n
                b1 = sxy1[i, k]
                zoff = _z_off(Z1, i, j)
                u = (z1a + c * vy - b1 * v2[j]) / Da            # v[a] + g v[nd+i]
                exyg = (c * eyy - b1 * zoff) / Da               # E[a,nd+i] + g E[nd+i,nd+i]
                exxg = (one(T) + (c * c * eyy - 2 * c * b1 * zoff + b1 * b1 * Zd[j]) / Da) / Da
            else
                u = (z1a + c * vy) / Da
                exyg = c * eyy / Da
                exxg = (one(T) + c * c * eyy / Da) / Da
            end
            λg[i, k] = w2k * (exyg - u * vy) + z1a * vy
            grad[nmod + ns + k] += u
            grad[nmod + ns + d + k] += (exxg - u * u) / 2
        end
    end
    l.intrinsic_scatter && (grad[nmod + 1] = dsig / 2)
    if isnan(nll) || isinf(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    _grad_seeded!(grad, pl.code, X, p, pl.cols, λf, λg)
    nll
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::MNRBandedLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    evaluate_nll_grad!(grad, prepare(l, model), p)
end


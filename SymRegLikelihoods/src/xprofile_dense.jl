# Profile likelihood ("prof", arXiv:2309.00948, Sec. 2.3 and Eqs. 37-39) with
# general dense covariances (slow), as in the ROXY package.  Instead of
# marginalizing the true inputs like MNR, it replaces them by their
# maximum-likelihood values.  To first order in the model, its quadratic form is
# that of the uniform-prior marginal likelihood ("unif", Eq. 30), which is the
# limit of the MNR likelihoods for an infinitely wide prior on the true inputs
# (w_k² → ∞); only the normalization differs, logdet(2πΣ) instead of logdet(2πD).
#
# The latent inputs are ordered variable-major (`vec(X)`), as for MNR.  With the
# model f and the input Jacobian, both evaluated at the observed inputs,
#   J = [Diag(g_1) ... Diag(g_d)]   (n x n*d),  g_k = ∂f/∂X[:,k]
# the NLL is
#   nll = 1/2 * logdet2piS + 1/2 * z' D⁻¹ z,   z = f - y
#   D   = Σyy + J*Σxx*J' - (J*Σxy + (J*Σxy)')
#   logdet2piS = logabsdet(2π * [Σxx Σxy; Σxy' Σyy])
#
# With `intrinsic_scatter = true`, Σyy -> Σyy + σ²I and σ² is the single
# likelihood parameter.  The log-determinant is then split using the
# eigenvalues λ of C = Σyy - Σxy' Σxx⁻¹ Σxy (symmetric covariances, nonsingular Σxx):
#   logabsdet(2π Σ(σ²)) = logabsdet(2π Σxx) + n log(2π) + Σ_i log(λ_i + σ²)
# C is positive semidefinite (Σ is a covariance).  For a (nearly) singular Σ,
# roundoff makes some λ_i slightly negative, and log|λ_i + σ²| would have
# spurious poles at σ² = -λ_i > 0 that an optimizer of σ² runs into, so these
# eigenvalues are set to zero.

"""
    XProfileDenseLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = false)

Profile likelihood ("prof" of arXiv:2309.00948, as in ROXY) for general dense
error covariances of the inputs `X` (ordered variable-major) and the outputs `y`:
the true inputs are replaced by their maximum-likelihood values instead of being
marginalized.  With `intrinsic_scatter = true` the intrinsic scatter `σ²` is
added to `Σyy` as the single likelihood parameter.
"""
struct XProfileDenseLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}          # n x d model-variable matrix
    y::Vector{T}
    Σxx::Matrix{T}        # (n*d) x (n*d), block-diagonal over variables
    Σxy::Matrix{T}        # (n*d) x n
    Σyy::Matrix{T}
    logdet2piS::T         # logabsdet(2π Σ); only its σ-independent part with intrinsic scatter
    schur_eigvals::Vector{T}  # eigenvalues of C with intrinsic scatter, empty otherwise
    intrinsic_scatter::Bool
end
function XProfileDenseLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        Σxx::AbstractMatrix{T}, Σxy::AbstractMatrix{T}, Σyy::AbstractMatrix{T};
        intrinsic_scatter::Bool = false) where {T}
    n, d = size(X, 1), size(X, 2)
    length(y) == n ||
        throw(DimensionMismatch("length(y) != size(X,1)"))
    (size(Σxx) == (n * d, n * d) && size(Σxy) == (n * d, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("data/covariance dimensions do not match"))
    logdet2piS, λC = if intrinsic_scatter
        _xprofile_logdet_parts(Matrix{T}(Σxx), Matrix{T}(Σxy), Matrix{T}(Σyy))
    else
        Σfull = [Matrix(Σxx) Matrix(Σxy); Matrix(transpose(Σxy)) Matrix(Σyy)]
        logabsdet(T(2) * pi .* Σfull)[1], T[]
    end
    XProfileDenseLikelihood{T}(Matrix{T}(X), Vector{T}(y),
        Matrix{T}(Σxx), Matrix{T}(Σxy), Matrix{T}(Σyy), logdet2piS, λC, intrinsic_scatter)
end

has_intrinsic_scatter(l::XProfileDenseLikelihood) = l.intrinsic_scatter
n_likelihood_params(l::XProfileDenseLikelihood) = Int(l.intrinsic_scatter)
positive_params(l::XProfileDenseLikelihood) = 1:Int(l.intrinsic_scatter)

# The σ-independent part of logabsdet(2π Σ(σ²)) and the eigenvalues of C.
function _xprofile_logdet_parts(Σxx::Matrix{T}, Σxy::Matrix{T}, Σyy::Matrix{T}) where {T}
    nd, n = size(Σxy)
    F = _try_lu!(copy(Σxx))
    F === nothing && throw(ArgumentError(
        "XProfileDenseLikelihood with intrinsic scatter needs a nonsingular Σxx"))
    C = Σyy - transpose(Σxy) * (F \ Σxy)
    λC = eigvals(Symmetric((C + transpose(C)) / 2))
    # eigenvalues up to `tol` are taken as roundoff of a zero eigenvalue; far
    # below zero, Σ is not a covariance
    scale = maximum(abs, diag(Σyy))
    tol = n * eps(T) * scale
    minimum(λC) < -sqrt(eps(T)) * scale && throw(ArgumentError(
        "XProfileDenseLikelihood: the covariance [Σxx Σxy; Σxy' Σyy] is not positive " *
        "semidefinite (its Schur complement has the eigenvalue $(minimum(λC)))"))
    nzero = count(<=(tol), λC)
    nzero > 0 && @warn "XProfileDenseLikelihood: the covariance [Σxx Σxy; Σxy' Σyy] is " *
        "singular ($nzero zero eigenvalues), so the NLL is unbounded below as σ² → 0. " *
        "Consider adding a small jitter to the diagonal of Σyy."
    λC .= max.(λC, zero(T))
    logabsdet(F)[1] + (nd + n) * log(2 * T(pi)), λC
end

# Σ_i log(λ_i + σ²); zero without intrinsic scatter.
function _xprofile_scatter_logdet(l::XProfileDenseLikelihood, s2int)
    acc = zero(s2int)
    @inbounds for λ in l.schur_eigvals
        acc += log(λ + s2int)
    end
    acc
end

# D = Σyy + JΣxxJ' - JΣxy - (JΣxy)' + σ²I from the input partials `g` (n x d).
# J (n x n*d) has the single nonzero J[i, (k - 1) * n + i] = g[i, k] per row and
# variable, so it is applied through `g` without being formed:
#   (JΣxxJ')[i, j] = Σ_k Σ_l g[i, k] g[j, l] Σxx[(k - 1) * n + i, (l - 1) * n + j]
#   (JΣxy)[i, j]   = Σ_k g[i, k] Σxy[(k - 1) * n + i, j]
function _xprofile_D!(l::XProfileDenseLikelihood{T}, g, s2int, ::Type{TE}) where {T,TE}
    n, d = size(l.X)
    Σxx, Σxy, Σyy = l.Σxx, l.Σxy, l.Σyy
    JΣxy = get_buffer(get_matrix_diffcache(T, :xprofile_Jxy, n, n), TE)
    @inbounds for j in 1:n
        for i in 1:n
            JΣxy[i, j] = zero(TE)
        end
        for k in 1:d
            off = (k - 1) * n
            for i in 1:n
                JΣxy[i, j] += g[i, k] * Σxy[off + i, j]
            end
        end
    end
    Dm = get_buffer(get_matrix_diffcache(T, :xprofile_D, n, n), TE)
    @inbounds for j in 1:n
        for i in 1:n
            Dm[i, j] = Σyy[i, j] - JΣxy[i, j] - JΣxy[j, i]
        end
        for l in 1:d
            cj = (l - 1) * n + j
            gj = g[j, l]
            for k in 1:d
                off = (k - 1) * n
                for i in 1:n
                    Dm[i, j] += g[i, k] * gj * Σxx[off + i, cj]
                end
            end
        end
        Dm[j, j] += s2int
    end
    Dm
end

# NLL from the model values `f` and the input partials `jacx` (n x d).
function _xprofile_nll(l::XProfileDenseLikelihood{T}, f, jacx, s2int, ::Type{TE}) where {T,TE}
    s2int >= 0 || return _worst_loss(TE, T)
    n = size(l.X, 1)
    Dm = _xprofile_D!(l, jacx, s2int, TE)
    z = get_buffer(get_vector_diffcache(T, :xprofile_z, n), TE)
    @inbounds for i in 1:n
        z[i] = f[i] - l.y[i]
    end
    v = get_buffer(get_vector_diffcache(T, :xprofile_v, n), TE)
    D_lu = _try_lu!(Dm)
    D_lu === nothing && return _worst_loss(TE, T)
    ldiv!(v, D_lu, z)

    nll = T(1/2) * (l.logdet2piS + _xprofile_scatter_logdet(l, s2int)) + T(1/2) * dot(z, v)
    (isnan(nll) || isinf(nll)) ? _worst_loss(TE, T) : nll
end


function evaluate_nll(l::XProfileDenseLikelihood{T}, model::Model, p::AbstractVector) where {T}
    _check_nll_dims(l, model, p)
    n, d = size(l.X)
    TE = eltype(p)
    f = get_buffer(get_vector_diffcache(T, :xprofile_f, n), TE)
    jacx = get_buffer(get_matrix_diffcache(T, :xprofile_jacx, n, d), TE)
    interpret_jac!(f, nothing, jacx, model, l.X, p)
    _xprofile_nll(l, f, jacx, _scatter2(l, p, n_param(model)), TE)
end

struct PreparedXProfileDense{T} <: PreparedLikelihood{T}
    likelihood::XProfileDenseLikelihood{T}
    model::Model{T}
    code::Vector{Instruction{T}}   # f and all d input partials in one code vector
    cols::Vector{Int}              # [fidx; dfidxs...]: positions in `code`
end

function prepare(l::XProfileDenseLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.X, 2))
    PreparedXProfileDense(l, model, dcode, [Int(fidx); Int.(dfidxs)])
end

# Uses the same differentiated code as the gradient, so that both agree.
function evaluate_nll(pl::PreparedXProfileDense{T}, p::AbstractVector) where {T}
    l = pl.likelihood
    _check_nll_dims(l, pl.model, p)
    n, d = size(l.X)
    TE = eltype(p)
    outs = get_buffer(get_matrix_diffcache(T, :xprofile_outs, n, d + 1), TE)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    _xprofile_nll(l, view(outs, :, 1), view(outs, :, 2:d + 1),
                  _scatter2(l, p, n_param(pl.model)), TE)
end

# With v = D⁻¹z, u = J'v, s = Σxy v and c = (k−1)n + i:
#   dnll/df_i  = v_i
#   dnll/dg_ik = −½ v_i [ (Σxx u)_c + (Σxx' u)_c − 2 s_c ]
#   dnll/dσ²   = ½ (Σ_i 1/(λ_i + σ²) - v'v)
function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedXProfileDense{T},
        p::AbstractVector{TD}) where {T,TD}
    l = pl.likelihood
    _check_grad_dims(grad, l, pl.model, p)
    n, d = size(l.X)
    nd = n * d
    nmod = n_param(pl.model)
    s2int = _scatter2(l, p, nmod)
    s2int >= 0 || (fill!(grad, zero(TD)); return _worst_loss(TD, T))

    outs = get_buffer(get_matrix_diffcache(T, :xprofile_outs, n, d + 1), TD)
    interpret_vecmat!(outs, pl.code, l.X, p, pl.cols)
    f = view(outs, :, 1)

    g = view(outs, :, 2:d + 1)
    Dm = _xprofile_D!(l, g, s2int, TD)
    z = get_buffer(get_vector_diffcache(T, :xprofile_z, n), TD)
    @inbounds for i in 1:n
        z[i] = f[i] - l.y[i]
    end
    v = get_buffer(get_vector_diffcache(T, :xprofile_v, n), TD)
    D_lu = _try_lu!(Dm)
    if D_lu === nothing
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    ldiv!(v, D_lu, z)
    nll = T(1/2) * (l.logdet2piS + _xprofile_scatter_logdet(l, s2int)) + T(1/2) * dot(z, v)
    if isnan(nll) || isinf(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end

    u = get_buffer(get_vector_diffcache(T, :xprofile_u, nd), TD)
    q = get_buffer(get_vector_diffcache(T, :xprofile_q, nd), TD)
    qs = get_buffer(get_vector_diffcache(T, :xprofile_qs, nd), TD)
    s = get_buffer(get_vector_diffcache(T, :xprofile_s, nd), TD)
    @inbounds for k in 1:d, i in 1:n
        u[(k - 1) * n + i] = g[i, k] * v[i]
    end
    mul!(q, l.Σxx, u)
    mul!(qs, transpose(l.Σxx), u)
    mul!(s, l.Σxy, v)
    λf = get_buffer(get_vector_diffcache(T, :xprofile_lf, n), TD)
    λg = get_buffer(get_matrix_diffcache(T, :xprofile_lg, n, d), TD)
    fill!(grad, zero(TD))
    if l.intrinsic_scatter
        tr = zero(TD)
        @inbounds for λ in l.schur_eigvals
            tr += inv(λ + s2int)
        end
        grad[nmod + 1] = (tr - dot(v, v)) / 2
    end
    @inbounds for i in 1:n
        λf[i] = v[i]
        for k in 1:d
            c = (k-1)*n + i
            λg[i, k] = -T(1/2) * v[i] * (q[c] + qs[c] - 2 * s[c])
        end
    end
    _grad_seeded!(grad, pl.code, l.X, p, pl.cols, λf, λg)
    nll
end

# Differentiates the model on every call; use `prepare` in hot loops.
function evaluate_nll_grad!(grad::AbstractVector{TD}, l::XProfileDenseLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    dcode, fidx, dfidxs = differentiate(code(model), 1:size(l.X, 2))
    evaluate_nll_grad!(grad, PreparedXProfileDense(l, model, dcode, [Int(fidx); Int.(dfidxs)]), p)
end


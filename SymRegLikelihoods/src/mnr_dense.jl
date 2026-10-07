# MNR (marginalized normal regression, arXiv:2309.00948) likelihood with general
# dense error covariances; the system M, z is the same as for
# `MNRBandedLikelihood` (see mnr_banded.jl).
# θ = [β; σ²; mu_1; …; mu_d; w_1²; …; w_d²], without σ² when
# `intrinsic_scatter = false`.  The gradient is the generic ForwardDiff one.

using LinearAlgebra

"""
    MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = true)

Marginalized normal regression likelihood (MNR, arXiv:2309.00948) for general
dense error covariances of the inputs `X` (ordered variable-major, `vec(X)`) and
the outputs `y`.  The likelihood parameters are those of
[`MNRDiagonalLikelihood`](@ref); [`MNRBandedLikelihood`](@ref) is much faster for
banded covariances.
"""
struct MNRDenseLikelihood{T} <: AbstractLikelihood{T}
    X::Matrix{T}
    y::Vector{T}
    Σxx::Matrix{T}   # (n*d) x (n*d), block-diagonal over variables
    Σxy::Matrix{T}   # (n*d) x n
    Σyy::Matrix{T}
    intrinsic_scatter::Bool
end
function MNRDenseLikelihood(X::AbstractMatrix{T}, y::AbstractVector{T},
        Σxx::AbstractMatrix{T}, Σxy::AbstractMatrix{T}, Σyy::AbstractMatrix{T};
        intrinsic_scatter::Bool = true) where {T}
    n, d = size(X, 1), size(X, 2)
    length(y) == n || throw(DimensionMismatch("length(y) != size(X,1)"))
    (size(Σxx) == (n * d, n * d) && size(Σxy) == (n * d, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("MNR covariance dimensions do not match data"))
    MNRDenseLikelihood{T}(Matrix{T}(X), Vector{T}(y), Matrix{T}(Σxx), Matrix{T}(Σxy),
                          Matrix{T}(Σyy), intrinsic_scatter)
end

n_observations(l::MNRDenseLikelihood) = length(l.y)
has_intrinsic_scatter(l::MNRDenseLikelihood) = l.intrinsic_scatter
# [sigma²], mu_k, w_k²
n_likelihood_params(l::MNRDenseLikelihood) = Int(l.intrinsic_scatter) + 2 * size(l.X, 2)
positive_params(l::MNRDenseLikelihood) = _mnr_positive_params(Int(l.intrinsic_scatter), size(l.X, 2))

# The system M, z (see mnr_banded.jl) is never assembled: adding W J' and
# J W J' = w² g² to the covariances would round Σxy and Σyy away for a wide
# latent prior (w² g² ≫ Σyy), and the NLL would then be arbitrarily wrong.
# Instead the latent block A = Σxx + W is eliminated with W cancelled
# analytically.  With J the input Jacobian (n x n*d; J[i, (k-1)*n+i] = g[i,k]),
# K = A⁻¹ and
#   E  = Σxy - Σxx J'                       (n*d x n)
#   D  = Σyy + J Σxx J' - J Σxy - (J Σxy)'  (n x n, the profile-likelihood matrix)
#   z1 = μ - vec(X)
# the Schur complement and the reduced right-hand side are
#   S = D + σ²I - E' K E,      r = f - y - E' K z1
# and
#   log|det M| = log|det A| + log|det S|,    z'M⁻¹z = z1' K z1 + r' S⁻¹ r.
# W enters only through K, which stays bounded (K → 0 and S → D for w² → ∞).
# Neither J nor W is formed: J is applied through `jacx`, W through `w2`.
function _mnr_dense_nll(l::MNRDenseLikelihood{T}, f, jacx, sig2, mu, w2,
        ::Type{TE}) where {T,TE}
    n, d = size(l.X)
    nd = n * d
    ntot = nd + n
    Σxx, Σxy, Σyy = l.Σxx, l.Σxy, l.Σyy

    A = get_buffer(get_matrix_diffcache(T, :mnr_A, nd, nd), TE)
    E = get_buffer(get_matrix_diffcache(T, :mnr_E, nd, n), TE)
    KE = get_buffer(get_matrix_diffcache(T, :mnr_KE, nd, n), TE)
    S = get_buffer(get_matrix_diffcache(T, :mnr_S, n, n), TE)
    z1 = get_buffer(get_vector_diffcache(T, :mnr_z1, nd), TE)
    Kz = get_buffer(get_vector_diffcache(T, :mnr_Kz, nd), TE)
    r = get_buffer(get_vector_diffcache(T, :mnr_r, n), TE)
    v = get_buffer(get_vector_diffcache(T, :mnr_v, n), TE)

    @inbounds for b in 1:nd, a in 1:nd
        A[a, b] = Σxx[a, b]
    end
    @inbounds for k in 1:d, i in 1:n
        a = (k - 1) * n + i
        A[a, a] += w2[k]
        z1[a] = mu[k] - l.X[i, k]
    end
    # (Σxx J')[a, j] = Σ_k Σxx[a, (k-1)*n+j] g[j, k]
    @inbounds for j in 1:n, a in 1:nd
        acc = convert(TE, Σxy[a, j])
        for k in 1:d
            acc -= Σxx[a, (k - 1) * n + j] * jacx[j, k]
        end
        E[a, j] = acc
    end
    # D = Σyy - J E - (J Σxy)', with (J E)[i, j] = Σ_k g[i,k] E[(k-1)*n+i, j]
    @inbounds for j in 1:n, i in 1:n
        acc = convert(TE, Σyy[i, j])
        for k in 1:d
            acc -= jacx[i, k] * E[(k - 1) * n + i, j] + jacx[j, k] * Σxy[(k - 1) * n + j, i]
        end
        S[i, j] = acc
    end
    @inbounds for i in 1:n
        S[i, i] += sig2
        r[i] = f[i] - l.y[i]
    end

    A_lu = _try_lu!(A)
    A_lu === nothing && return _worst_loss(TE, T)
    copyto!(KE, E)
    ldiv!(A_lu, KE)
    copyto!(Kz, z1)
    ldiv!(A_lu, Kz)
    mul!(S, transpose(E), KE, -one(TE), one(TE))
    mul!(r, transpose(E), Kz, -one(TE), one(TE))

    S_lu = _try_lu!(S)
    S_lu === nothing && return _worst_loss(TE, T)
    ldiv!(v, S_lu, r)

    nll = T(1/2) * (logabsdet(A_lu)[1] + logabsdet(S_lu)[1] + ntot * log(2 * T(pi)) +
                    dot(z1, Kz) + dot(r, v))
    (isnan(nll) || isinf(nll)) ? _worst_loss(TE, T) : nll
end

function evaluate_nll(l::MNRDenseLikelihood{T}, model::Model, p::AbstractVector) where {T}
    n, d = size(l.X)
    n_model_param = n_param(model)
    ns = Int(l.intrinsic_scatter)
    _check_nll_dims(l, model, p)
    TE = eltype(p)
    sig2 = _scatter2(l, p, n_model_param)
    mu = view(p, n_model_param + ns + 1:n_model_param + ns + d)
    w2 = view(p, n_model_param + ns + d + 1:n_model_param + ns + 2d)
    _mnr_valid(sig2, w2, TE) || return _worst_loss(TE, T)

    f = get_buffer(get_vector_diffcache(T, :mnrd_f, n), TE)
    jacx = get_buffer(get_matrix_diffcache(T, :mnrd_jacx, n, d), TE)
    interpret_jac!(f, nothing, jacx, model, l.X, p)
    _mnr_dense_nll(l, f, jacx, sig2, mu, w2, TE)
end

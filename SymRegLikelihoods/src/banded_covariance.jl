# Error covariances that are banded in time, for errors-in-variables likelihoods
# of data derived from time series (e.g. finite-difference derivatives), and the
# band LDLᵀ kernels shared by `MNRGeneralBandedLikelihood` and
# `XUniformBandedLikelihood`.
#
# The inputs X (n x d) are ordered variable-major (`vec(X)`) and every entry of
# Σxx, Σxy or Σyy that couples observations i and j must have |i - j| <= h.
# Covariances combine linearly, Σ = Σ_r θ_r P_r, which describes a noise model
# with known patterns P_r (e.g. the stencils of the raw noise of each variable)
# and scale parameters θ_r (noise variances).

"""
    BandedCovariance(Σxx, Σxy, Σyy)

Error covariances of the inputs (n x d, ordered variable-major) and of the n
outputs that are banded in time; the half bandwidth h (in observations) is
detected from the matrices.  Covariances can be combined linearly
(`θ1 * P1 + θ2 * P2`), e.g. to build Σ = Σ_r θ_r P_r from noise patterns.
"""
struct BandedCovariance{T}
    n::Int
    d::Int
    h::Int                     # half bandwidth in time
    sxx::Array{T,4}            # sxx[k, l, r + h + 1, i] = Σxx[(k-1)*n + i, (l-1)*n + i + r]
    sxy::Array{T,3}            # sxy[k, r + h + 1, i]    = Σxy[(k-1)*n + i, i + r]
    syy::Matrix{T}             # syy[r + h + 1, i]       = Σyy[i, i + r]
end

function BandedCovariance(Σxx::AbstractMatrix{T}, Σxy::AbstractMatrix{T},
        Σyy::AbstractMatrix{T}) where {T}
    n = size(Σyy, 1)
    nd = size(Σxx, 1)
    (n > 0 && nd % n == 0) || throw(DimensionMismatch("size(Σxx, 1) must be a multiple of size(Σyy, 1)"))
    d = nd ÷ n
    (size(Σxx) == (nd, nd) && size(Σxy) == (nd, n) && size(Σyy) == (n, n)) ||
        throw(DimensionMismatch("covariance dimensions do not match"))
    time(a) = (a - 1) % n + 1
    h = 0
    for b in 1:nd, a in 1:nd
        iszero(Σxx[a, b]) || (h = max(h, abs(time(a) - time(b))))
    end
    for j in 1:n, a in 1:nd
        iszero(Σxy[a, j]) || (h = max(h, abs(time(a) - j)))
    end
    for j in 1:n, i in 1:n
        iszero(Σyy[i, j]) || (h = max(h, abs(i - j)))
    end
    nr = 2h + 1
    sxx = zeros(T, d, d, nr, n)
    sxy = zeros(T, d, nr, n)
    syy = zeros(T, nr, n)
    for i in 1:n, r in -h:h
        j = i + r
        1 <= j <= n || continue
        syy[r + h + 1, i] = Σyy[i, j]
        for k in 1:d
            sxy[k, r + h + 1, i] = Σxy[(k - 1) * n + i, j]
            for l in 1:d
                sxx[k, l, r + h + 1, i] = Σxx[(k - 1) * n + i, (l - 1) * n + j]
            end
        end
    end
    BandedCovariance{T}(n, d, h, sxx, sxy, syy)
end

# the same covariance stored with a larger half bandwidth
function _widen(c::BandedCovariance{T}, h::Int) where {T}
    h == c.h && return c
    off = h - c.h
    sxx = zeros(T, c.d, c.d, 2h + 1, c.n)
    sxy = zeros(T, c.d, 2h + 1, c.n)
    syy = zeros(T, 2h + 1, c.n)
    sxx[:, :, off+1:off+2c.h+1, :] .= c.sxx
    sxy[:, off+1:off+2c.h+1, :] .= c.sxy
    syy[off+1:off+2c.h+1, :] .= c.syy
    BandedCovariance{T}(c.n, c.d, h, sxx, sxy, syy)
end

function Base.:+(a::BandedCovariance{T}, b::BandedCovariance{T}) where {T}
    (a.n, a.d) == (b.n, b.d) || throw(DimensionMismatch("covariances of different data"))
    h = max(a.h, b.h)
    a, b = _widen(a, h), _widen(b, h)
    BandedCovariance{T}(a.n, a.d, h, a.sxx .+ b.sxx, a.sxy .+ b.sxy, a.syy .+ b.syy)
end

Base.:*(θ::Real, c::BandedCovariance{T}) where {T} =
    BandedCovariance{T}(c.n, c.d, c.h, T(θ) .* c.sxx, T(θ) .* c.sxy, T(θ) .* c.syy)

# ---------------------------------------------------------------------------
# The profile matrix D = Σyy + J Σxx J' - J Σxy - (J Σxy)' (the covariance of the
# output errors minus the propagated input errors), J[i, (k-1)*n+i] = g[i,k]
# ---------------------------------------------------------------------------

# D[i, i + r] for r >= 0
@inline function _banded_D(c::BandedCovariance, g, i, r)
    h, d = c.h, c.d
    j = i + r
    b = r + h + 1                # band index of (i, j)
    bm = h + 1 - r               # band index of (j, i)
    @inbounds begin
        v = c.syy[b, i] + zero(eltype(g))
        for k in 1:d
            v -= g[i, k] * c.sxy[k, b, i] + g[j, k] * c.sxy[k, bm, j]
            for l in 1:d
                v += g[i, k] * g[j, l] * c.sxx[k, l, b, i]
            end
        end
    end
    v
end

# Add G ∂D[i, i + r]/∂g to λg (n x d).
@inline function _banded_D_adjoint!(λg, c::BandedCovariance, g, i, r, G)
    h, d = c.h, c.d
    j = i + r
    b = r + h + 1
    bm = h + 1 - r
    @inbounds for k in 1:d
        gi = -c.sxy[k, b, i]
        gj = -c.sxy[k, bm, j]
        for l in 1:d
            gi += g[j, l] * c.sxx[k, l, b, i]
            gj += g[i, l] * c.sxx[l, k, b, i]
        end
        λg[i, k] += G * gi
        λg[j, k] += G * gj
    end
    nothing
end

# ---------------------------------------------------------------------------
# Band LDLᵀ kernels; the lower band of a symmetric N x N matrix with half
# bandwidth b is stored as B[p - q + 1, q] = A[p, q] for p >= q.
# ---------------------------------------------------------------------------

# In-place band LDLᵀ: afterwards B[1, c] = D_c and B[p - c + 1, c] = L[p, c].
# Returns `false` for a non-positive or non-finite pivot.
function _band_ldlt!(B, b, N)
    @inbounds for c in 1:N
        dc = B[1, c]
        for q in max(1, c - b):c - 1
            lcq = B[c - q + 1, q]
            dc -= lcq * lcq * B[1, q]
        end
        (isfinite(dc) && dc > 0) || return false
        B[1, c] = dc
        for p in c + 1:min(N, c + b)
            v = B[p - c + 1, c]
            for q in max(1, p - b):c - 1
                v -= B[p - q + 1, q] * B[c - q + 1, q] * B[1, q]
            end
            B[p - c + 1, c] = v / dc
        end
    end
    true
end

# log det and u' (L D Lᵀ)⁻¹ u from the factors, with u = L⁻¹ u in place.
function _band_logdet_quadratic!(u, B, b, N)
    logdet = zero(eltype(u))
    quad = zero(eltype(u))
    @inbounds for c in 1:N
        v = u[c]
        for q in max(1, c - b):c - 1
            v -= B[c - q + 1, q] * u[q]
        end
        u[c] = v
        logdet += log(B[1, c])
        quad += v * v / B[1, c]
    end
    logdet, quad
end

# v = (L D Lᵀ)⁻¹ z from u = L⁻¹ z (in place: `u` becomes v).
function _band_backsolve!(u, B, b, N)
    @inbounds for c in N:-1:1
        v = u[c] / B[1, c]
        for p in c + 1:min(N, c + b)
            v -= B[p - c + 1, c] * u[p]
        end
        u[c] = v
    end
    nothing
end

# Entries of Z = (L D Lᵀ)⁻¹ within the band (Takahashi's recurrence), stored like
# the factors: Zb[p - q + 1, q] = Z[p, q] for p >= q.
function _band_selected_inverse!(Zb, B, b, N)
    fill!(Zb, zero(eltype(Zb)))
    @inbounds for c in N:-1:1
        pmax = min(N, c + b)
        for p in pmax:-1:c + 1
            acc = zero(eltype(Zb))
            for q in c + 1:pmax
                zpq = p >= q ? Zb[p - q + 1, q] : Zb[q - p + 1, p]
                acc -= zpq * B[q - c + 1, c]
            end
            Zb[p - c + 1, c] = acc
        end
        acc = inv(B[1, c])
        for q in c + 1:pmax
            acc -= B[q - c + 1, c] * Zb[q - c + 1, c]
        end
        Zb[1, c] = acc
    end
    nothing
end

# ∂NLL/∂A_pq of NLL = ½ (log det A + z' A⁻¹ z) for the stored entry p >= q, from
# the band of Z = A⁻¹ and v = A⁻¹ z
@inline _band_G(Zb, v, p, q) = p == q ? (Zb[1, q] - v[p] * v[q]) / 2 : Zb[p - q + 1, q] - v[p] * v[q]

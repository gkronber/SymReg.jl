# ---------------------------------------------------------------------------
# Symmetric eigendecomposition of small information matrices
# ---------------------------------------------------------------------------
# The parameter complexity needs all eigenvalues and eigenvectors of the
# information matrix, a small (n ≲ 20) symmetric matrix that is often strongly
# graded: badly parameterized models give precisions that span many orders of
# magnitude.
#
# A cyclic Jacobi method is used instead of LAPACK's `eigen(Symmetric(A))`:
#
#   * Accuracy.  Standard solvers only determine eigenvalues to an absolute
#     error of about `eps * opnorm(A)`, so the small precisions of a graded
#     matrix come out wrong by orders of magnitude (in Float32 and in Float64).
#     Jacobi with the relative stopping criterion below computes them to high
#     relative accuracy (Demmel & Veselić, SIAM J. Matrix Anal. Appl. 13, 1992;
#     guaranteed for positive definite `D * C * D` with well-conditioned `C`).
#   * Robustness.  The MRRR path of `ssyevr` (OpenBLAS 0.3.30 and current
#     Reference-LAPACK) overflows in `slarre` when it forms `L^2 * D` as
#     `(L * L) * D` for such a matrix, passes `Inf` to dqds and then loops
#     forever in `slarrb` on the resulting NaN; a thread stuck in a `ccall`
#     also blocks the garbage collector of the whole process.  Here every loop
#     is bounded.
#   * Threads.  No BLAS/LAPACK call, so concurrent evaluations do not serialize
#     on OpenBLAS's buffer-allocation lock.
#
# The computation stays in the element type of the matrix.

const _JACOBI_MAX_SWEEPS = 50


# Eigendecomposition `(λ, V)` of the information matrix with negative
# eigenvalues set to zero.  Away from a minimum of the NLL (e.g. parameters
# that are not fully optimized) the information matrix can be indefinite; a
# direction of negative curvature carries no information about the
# parameters, so it counts as zero precision.  The Jacobi solver is used 
# for its relative accuracy on graded matrices.
function _nonnegative_eigen(FI::AbstractMatrix)
    λ, V = _symmetric_eigen(FI)
    max.(λ, zero(eltype(λ))), V
end


# Number of (nonnegative) eigenvalues `λ` above the numerical-rank tolerance.
function _effective_rank(λ::AbstractVector)
    rtol = length(λ)*eps(eltype(λ))
    tol = rtol*maximum(λ)
    count(>(tol), λ)
end

"""
    _symmetric_eigen(A::AbstractMatrix) -> (λ, V)

Eigenvalues `λ` (ascending) and orthonormal eigenvectors `V` (columns) of the
symmetric matrix `A`, computed by the cyclic Jacobi method in the element type
of `A`.  Only the upper triangle of `A` is read.

An off-diagonal entry is annihilated unless it is negligible relative to its
two diagonal entries, `|a_pq| <= eps * sqrt(|a_pp * a_qq|)`; the sweeps end
when no rotation was applied, or after `$(_JACOBI_MAX_SWEEPS)` sweeps (which
only matters for non-finite input).
"""
function _symmetric_eigen(A::AbstractMatrix)
    n = LinearAlgebra.checksquare(A)
    T = float(eltype(A))
    W = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:j
        W[i, j] = A[i, j]
        W[j, i] = A[i, j]
    end
    V = Matrix{T}(I, n, n)
    _jacobi_eigen!(W, V)
    λ = diag(W)
    perm = sortperm(λ)
    return λ[perm], V[:, perm]
end

# Diagonalizes the symmetric `A` in place by plane rotations accumulated into
# `V`.  `A` must hold both triangles; on return its diagonal holds the
# eigenvalues (unsorted) and the off-diagonal entries are negligible.
function _jacobi_eigen!(A::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    tol = eps(T)
    for _ in 1:_JACOBI_MAX_SWEEPS
        rotated = false
        for p in 1:(n - 1), q in (p + 1):n
            apq = A[p, q]
            iszero(apq) && continue
            app = A[p, p]
            aqq = A[q, q]
            abs(apq) <= tol * sqrt(abs(app) * abs(aqq)) && continue
            rotated = true

            # Rotation that annihilates a_pq, with t = tan of the angle chosen as
            # the smaller root (|angle| <= π/4); `hypot` avoids overflow of θ².
            θ = (aqq - app) / (2 * apq)
            t = θ >= 0 ? inv(θ + hypot(one(T), θ)) : -inv(-θ + hypot(one(T), θ))
            c = inv(hypot(one(T), t))
            s = t * c

            # Rutishauser's update: the diagonal changes by t * a_pq, which
            # avoids the cancellation of c²a_pp - 2cs a_pq + s²a_qq on graded
            # matrices; only the off-diagonal entries of rows/columns p, q rotate.
            A[p, p] = app - t * apq
            A[q, q] = aqq + t * apq
            A[p, q] = zero(T)
            A[q, p] = zero(T)
            @inbounds for k in 1:n
                (k == p || k == q) && continue
                akp = A[k, p]
                akq = A[k, q]
                bkp = c * akp - s * akq
                bkq = s * akp + c * akq
                A[k, p] = bkp
                A[p, k] = bkp
                A[k, q] = bkq
                A[q, k] = bkq
            end
            @inbounds for k in 1:n
                vkp = V[k, p]
                vkq = V[k, q]
                V[k, p] = c * vkp - s * vkq
                V[k, q] = s * vkp + c * vkq
            end
        end
        rotated || break
    end
    return A, V
end

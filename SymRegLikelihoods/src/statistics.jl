# ---------------------------------------------------------------------------
# Confidence / prediction intervals for the MLE (Laplace approximation)
# ---------------------------------------------------------------------------

using ForwardDiff
using LinearAlgebra
using Distributions

# `NaN` for the negative variances of an indefinite Hessian.
_std_err(v::Real) = v < zero(v) ? oftype(float(v), NaN) : sqrt(float(v))

"""
    confidence_intervals(likelihood, model, opt_coeff; alpha = 0.01)
        -> (low, high, paramStdErr, z, cov, corr)

Laplace-approximation `1 - alpha` confidence intervals `[low, high]` of the
fitted natural parameters `opt_coeff`, using Student's t distribution with
`n_observations - length(opt_coeff)` degrees of freedom.  Also returns the
standard errors, the z-scores `|opt_coeff| ./ paramStdErr`, the
[`covariance`](@ref) and the correlation matrix.
"""
function confidence_intervals(likelihood::AbstractLikelihood, model::Model, opt_coeff::AbstractArray; alpha = 0.01)
    cov = covariance(likelihood, model, opt_coeff)

    m = n_observations(likelihood)
    n = length(opt_coeff)
    m > n ||
        throw(ArgumentError("t interval needs more observations ($m) than parameters ($n)"))

    paramStdErr = _std_err.(diag(cov))
    z = abs.(opt_coeff ./ paramStdErr)

    corr = cov ./ (paramStdErr * paramStdErr')

    t = -quantile(TDist(m - n), alpha / 2.0)
    low = opt_coeff .- t .* paramStdErr
    high = opt_coeff .+ t .* paramStdErr
    return  low, high, paramStdErr, z, cov, corr
end

"""
    prediction_intervals(likelihood, model, X, opt_coeff; alpha = 0.01)
        -> (y_pred, low, high, resStdErr)

Laplace-approximation prediction intervals of `model` at the inputs `X`.
`resStdErr` is the delta-method standard error of the fitted mean,
`sqrt(diag(J * cov * J'))`, with `J = d(model)/d(opt_coeff)` evaluated at the
rows of `X`.  `X` is independent of the data held by `likelihood`, so the
interval can be evaluated on a prediction grid.

The interval covers a new observation, with standard error

    sqrt(resStdErr^2 + observation_scatter(likelihood, model, opt_coeff)^2)

and the `t` quantile with `n_observations - length(opt_coeff)` degrees of
freedom.
"""
function prediction_intervals(likelihood::AbstractLikelihood, model::Model, X::AbstractMatrix,
        opt_coeff::AbstractVector; alpha = 0.01)
    cov = covariance(likelihood, model, opt_coeff)

    nrows = size(X, 1)
    ncols = length(opt_coeff)
    size(cov) == (ncols, ncols) ||
        throw(DimensionMismatch("covariance is $(size(cov)), expected ($ncols, $ncols)"))

    # the likelihood-parameter columns of the Jacobian are zero
    TD = eltype(opt_coeff)
    y_pred = Vector{TD}(undef, nrows)
    jac = Matrix{TD}(undef, nrows, ncols)
    interpret_jac!(y_pred, jac, nothing, model, X, opt_coeff)

    # diag(jac * cov * jac') without materializing the nrows x nrows product
    resStdErr = _std_err.(vec(sum((jac * cov) .* jac; dims = 2)))

    m = n_observations(likelihood)
    m > ncols ||
        throw(ArgumentError("t interval needs more observations ($m) than parameters ($ncols)"))
    t = -quantile(TDist(m - ncols), alpha / 2.0)

    sig = observation_scatter(likelihood, model, opt_coeff)
    predStdErr = sqrt.(resStdErr .^ 2 .+ sig^2)

    low = y_pred .- t .* predStdErr
    high = y_pred .+ t .* predStdErr
    return   y_pred, low, high, resStdErr
end

"""
    covariance(likelihood, model, θ) -> AbstractMatrix

Covariance of the natural parameters at the MLE `θ`: the pseudo-inverse of the
observed Fisher information matrix (see [`information_matrix`](@ref)).
"""
function covariance(likelihood::AbstractLikelihood, model::Model, opt_coeff::AbstractVector)
    hess = information_matrix(likelihood, model, opt_coeff)

    # tolerance for numeric rank determination
    tol = sqrt(eps(real(float(oneunit(eltype(hess)))))) # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.pinv
    pinv(hess; rtol = tol)  # inverse of the observed Fisher information matrix
end

# Principal square root of a symmetric PSD matrix; negative eigenvalues are clamped to zero.
function _psd_sqrt(S::AbstractMatrix)
    e = eigen(Symmetric(Matrix(S)))
    e.vectors * Diagonal(sqrt.(max.(e.values, zero(eltype(e.values))))) * e.vectors'
end

"""
    pairwise_confidence_regions_laplace(likelihood, model, opt_coeff; alpha = 0.95)
        -> Vector{Matrix}

For every pair `(i, j)` of parameters, the boundary of the `alpha` confidence
region as a `npoints x 2` matrix, in the order `(1,2), (1,3), …, (d-1,d)`.

Each region is the shadow of the joint confidence ellipsoid on the `(i, j)`
plane — the *marginal* region for that pair, with the other parameters
integrated out.
"""
function pairwise_confidence_regions_laplace(likelihood::AbstractLikelihood, model::Model, opt_coeff::AbstractVector; alpha=0.95)
    d = length(opt_coeff)
    n = n_observations(likelihood)
    d >= 2 ||
        throw(ArgumentError("pairwise regions need at least two parameters, got $d"))
    n > d ||
        throw(ArgumentError("F region needs more observations ($n) than parameters ($d)"))

    cov = covariance(likelihood, model, opt_coeff)

    # points on the unit circle; first and last coincide so the boundary closes
    circle = reduce(hcat, [sin(a), cos(a)] for a in range(0, 2 * pi; length = 65))

    # The joint region is the ellipsoid
    #   (q - q̂)' cov⁻¹ (q - q̂) <= d * F_{d, n-d}(alpha),
    # whose shadow on the (i, j) plane is q̂ + f * sqrt(cov_ij) * u for u on the
    # unit circle, with `cov_ij` the 2x2 sub-block of `cov`.
    f = sqrt(d * quantile(FDist(d, n - d), alpha))

    regions = Matrix{float(eltype(cov))}[]
    for i in 1:d-1
        for j in i+1:d
            S = [cov[i, i] cov[i, j]
                 cov[j, i] cov[j, j]]
            conf = [opt_coeff[i], opt_coeff[j]] .+ f .* (_psd_sqrt(S) * circle)
            push!(regions, Matrix(conf'))
        end
    end

    regions
end

using BenchmarkTools
using SymRegLikelihoods
using SymRegLikelihoods.SymRegInterpreter   # not directly resolvable in the AirspeedVelocity env
using SymRegLikelihoods.SymRegInterpreter: to_logspace_xy_model, with_fit_buffer_cache
using SymRegLikelihoods: prepare, evaluate_nll_grad_fd!
using LinearAlgebra
using Random

const SUITE = BenchmarkGroup()

# Compatibility with revisions before the natural parameters and the separate
# Gaussian/Laplace types, so that timings can be compared across them.
const NATURAL = isdefined(SymRegLikelihoods, :GaussianProfiledLikelihood)
# a variance `v` as a likelihood parameter: `v` itself, or `log(sqrt(v))`
var_param(v) = NATURAL ? v : log(v) / 2
# the parameters of a profiled likelihood: with its variance, or without
profiled(p, v) = NATURAL ? [p; v] : p
gaussian_profiled(X, y) = NATURAL ? SymRegLikelihoods.GaussianProfiledLikelihood(X, y) : GaussianLikelihood(X, y)
laplace_profiled(X, y) = NATURAL ? SymRegLikelihoods.LaplaceProfiledLikelihood(X, y) : LaplaceLikelihood(X, y)
gaussian_scatter(X, y, s) = NATURAL ? SymRegLikelihoods.GaussianScatterLikelihood(X, y, s) :
    GaussianLikelihood(X, y, s; intrinsic_scatter = true)
laplace_scatter(X, y, s) = NATURAL ? SymRegLikelihoods.LaplaceScatterLikelihood(X, y, s) :
    LaplaceLikelihood(X, y, s; intrinsic_scatter = true)

# ---------------------------------------------------------------------------
# Gaussian likelihood: evaluation, gradient, and optimisation (NLopt + LM)
# ---------------------------------------------------------------------------
for T in (Float32, Float64), n in (1024, 4096)
    rng = MersenneTwister(0)
    X = reshape(rand(rng, T, n) .* T(4), :, 1) .- T(2)
    y = T(2) .* vec(X) .- T(1) .+ T(0.1) .* randn(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    l_fix = GaussianLikelihood(X, y, T(0.2))   # known sigma (least-squares)
    l_est = gaussian_profiled(X, y)            # profiled sigma
    p0 = [T(1.0), T(0.0)]
    p0e = profiled(p0, T(0.01))
    g = zeros(T, 2)
    case = "$T / $n obs"

    SUITE["gaussian"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l_fix, $model, p) end setup=(p=copy($p0))
    SUITE["gaussian"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l_fix, $model, p) end setup=(p=copy($p0))
    SUITE["gaussian"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l_fix, $model, p) end setup=(p=copy($p0))
    # NLopt needs a Float64 gradient
    if T === Float64
        SUITE["gaussian"]["optimize NLopt"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.optimize(NLoptOptimizer(), $l_fix, $model, p; maxeval=100) end setup=(p=copy($p0)) evals=5
    end
    SUITE["gaussian"]["optimize LM"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.optimize(LevenbergMarquardtOptimizer(), $l_est, $model, p) end setup=(p=copy($p0e)) evals=5
end

# ---------------------------------------------------------------------------
# Gaussian likelihoods with a model built from the transcendental kernels that
# dominate typical GP models: f = p1*|x1|^p2 + p3*log(x2) + p4*exp(p5*x1)
# (POWABS, LOG, EXP, with a non-integer exponent).  Profiled noise (LM) and
# known noise plus intrinsic scatter (NLopt) are the two NeoGP paths.
# ---------------------------------------------------------------------------
for T in (Float32, Float64), n in (1024,)
    rng = MersenneTwister(5)
    X = rand(rng, T, n, 2) .* T(3) .+ T(0.5)   # positive inputs for log(x2)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p),
        :(p[1] * abs(x[1])^p[2] + p[3] * log(x[2]) + p[4] * exp(p[5] * x[1]))), 5)
    y = copy(SymRegInterpreter.interpret_vec(model, X, T[1.5, 1.7, 0.8, 0.3, -0.6])) .+
        T(0.1) .* randn(rng, T, n)
    l_est = gaussian_profiled(X, y)                 # profiled sigma
    l_sc = gaussian_scatter(X, y, fill(T(0.05), n)) # known sigma + intrinsic scatter
    p0 = T[1.2, 1.5, 1.0, 0.2, -0.5]                # away from the optimum
    p0e = profiled(p0, T(0.01))
    p0s = [p0; var_param(T(0.1)^2)]
    g = zeros(T, 6)
    case = "$T / $n obs"

    SUITE["pow-log-exp"]["gaussian profiled nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l_est, $model, p) end setup=(p=copy($p0e))
    SUITE["pow-log-exp"]["gaussian profiled nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l_est, $model, p) end setup=(p=copy($p0e))
    SUITE["pow-log-exp"]["gaussian profiled optimize LM"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.optimize(LevenbergMarquardtOptimizer(), $l_est, $model, p) end setup=(p=copy($p0e)) evals=5
    SUITE["pow-log-exp"]["gaussian profiled fim"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.information_matrix($l_est, $model, p) end setup=(p=copy($p0e))
    SUITE["pow-log-exp"]["gaussian scatter nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l_sc, $model, p) end setup=(p=copy($p0s))
    SUITE["pow-log-exp"]["gaussian scatter optimize NLopt"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.optimize(NLoptOptimizer(), $l_sc, $model, p; maxeval=100) end setup=(p=copy($p0s)) evals=5
    SUITE["pow-log-exp"]["gaussian scatter fim"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.information_matrix($l_sc, $model, p) end setup=(p=copy($p0s))
end

# ---------------------------------------------------------------------------
# Laplace likelihood: evaluation, gradient, and optimisation (NLopt)
# ---------------------------------------------------------------------------
for T in (Float32, Float64), n in (1024, 4096)
    rng = MersenneTwister(4)
    X = reshape(rand(rng, T, n) .* T(4), :, 1) .- T(2)
    y = T(2) .* vec(X) .- T(1) .+ T(0.1) .* randn(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    l_fix = LaplaceLikelihood(X, y, T(0.2))   # known scale
    l_est = laplace_profiled(X, y)            # profiled scale
    p0 = [T(1.0), T(0.0)]
    p0e = profiled(p0, T(0.01))
    g = zeros(T, 2)
    case = "$T / $n obs"

    SUITE["laplace"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l_fix, $model, p) end setup=(p=copy($p0))
    SUITE["laplace"]["nll (profiled scale)"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l_est, $model, p) end setup=(p=copy($p0e))
    SUITE["laplace"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l_fix, $model, p) end setup=(p=copy($p0))
    SUITE["laplace"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l_fix, $model, p) end setup=(p=copy($p0))
    # NLopt needs a Float64 gradient
    if T === Float64
        SUITE["laplace"]["optimize NLopt"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.optimize(NLoptOptimizer(), $l_fix, $model, p; maxeval=100) end setup=(p=copy($p0)) evals=5
    end
end

# ---------------------------------------------------------------------------
# Uniform-prior ("unif") likelihood with diagonal errors, on the RAR data
# ---------------------------------------------------------------------------
for T in (Float64,), n in (1024, 4096)
    rng = MersenneTwister(1)
    gbar = rand(rng, T, n) .+ T(0.1)
    body = :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), body), 3)
    model = SymRegInterpreter.to_logspace_xy_model(model)
    l = XUniformDiagonalLikelihood(log10.(gbar), # x
                      T(0.05)*rand(rng, T, n),  # xerr
                      log10.(rand(rng, T, n) .+ T(0.1)), # y
                      T(0.05)*rand(rng, T, n)) # yerr
    p = [T(0.8), T(-0.02), T(0.38)]
    g = zeros(T, 3)
    pl = SymRegLikelihoods.prepare(l, model)
    case = "$T / $n obs"

    SUITE["xuniform diagonal"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($pl, p) end setup=(p=copy($p))
    SUITE["xuniform diagonal"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $pl, p) end setup=(p=copy($p))
    # includes symbolic differentiation on every evaluation
    SUITE["xuniform diagonal"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Diagonal MNR (likelihood parameters sigma², mu_gauss, w²)
# ---------------------------------------------------------------------------
for T in (Float64,), n in (1024, 4096)
    rng = MersenneTwister(2)
    xobs = rand(rng, T, n) .+ T(1)
    xerr = T(0.2) .* rand(rng, T, n)
    yobs = (2 .* xobs .- T(0.5)) .+ T(0.1) .* randn(rng, T, n)
    yerr = T(0.15) .* rand(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    l = MNRDiagonalLikelihood(xobs, xerr, yobs, yerr)
    p = [T(2.0), T(-0.5), var_param(T(0.3)^2), T(1.5), var_param(T(1.2)^2)]
    g = zeros(T, 5)
    case = "$T / $n obs"

    SUITE["mnr diagonal"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["mnr diagonal"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    SUITE["mnr diagonal"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Cosmic chronometer (per-observation x, age, sig vectors)
# ---------------------------------------------------------------------------
for T in (Float64,), n in (1024, 4096)
    rng = MersenneTwister(3)
    x = rand(rng, T, n) .+ T(2)
    age = sqrt.(T(1) .+ T(0.2) .* x)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
    l = CosmicChronometerLikelihood(x, age, T(0.1) .* ones(T, n))
    p = [T(1.0), T(0.2)]
    g = zeros(T, 2)
    case = "$T / $n obs"

    SUITE["cosmic"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["cosmic"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    SUITE["cosmic"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Dense multivariate MNR likelihood, O((n*d)^3) per evaluation
# ---------------------------------------------------------------------------
for T in (Float64,), n in (64, 256)
    rng = MersenneTwister(5)
    X = [randn(rng, T, n) .* 2  T(0.7) .* randn(rng, T, n)]
    y = (X[:, 1] .^ 2 .- 1) .+ T(0.4) .* randn(rng, T, n)
    Σxx = Matrix(T(0.5)I, 2n, 2n) + T(0.02) .* randn(rng, T, 2n, 2n); Σxx = Σxx * Σxx'
    Σyy = Matrix(T(0.7)I, n, n) + T(0.02) .* randn(rng, T, n, n); Σyy = Σyy * Σyy'
    Σxy = T(0.02) .* randn(rng, T, 2n, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2] + p[3])), 3)
    l = MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy)
    p = [T(1.0), T(0.0), T(-1.0), var_param(T(0.4)^2), T(0.3), T(-0.2), var_param(T(1.1)^2), var_param(T(0.8)^2)]
    g = zeros(T, 8)
    case = "$T / $n obs"

    SUITE["mnr dense"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["mnr dense"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    SUITE["mnr dense"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Structured multivariate MNR likelihood, O(n*d) per evaluation
# ---------------------------------------------------------------------------
for T in (Float64,), n in (64, 256)
    rng = MersenneTwister(31)
    d = 2
    nd = 2 * n
    X = [randn(rng, T, n) .* 2  randn(rng, T, n)]
    y = (X[:, 1] .^ 2 .- 1) .+ T(0.4) .* randn(rng, T, n)
    Σxx = Diagonal(rand(rng, T, nd) .+ T(0.5))
    Σxy = zeros(T, nd, n)
    # variable-major order (vec(X) convention)
    for i in 1:n, k in 1:d
        Σxy[(k-1)*n+i, i] = T(0.2) * (rand(rng, T) - T(0.5))
    end
    Σyy = SymTridiagonal(rand(rng, T, n) .* 3 .+ 1.0, rand(rng, T, n - 1) .* T(0.4))
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
    l = MNRBandedLikelihood(X, y, Σxx, Σxy, Σyy)
    p = [T(1.0), T(-1.0), var_param(T(0.4)^2), T(0.3), T(-0.2), var_param(T(1.1)^2), var_param(T(0.8)^2)]
    g = zeros(T, 7)
    case = "$T / $n obs"

    SUITE["mnr banded"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["mnr banded"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    SUITE["mnr banded"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Profile ("prof") likelihood with dense covariances
# ---------------------------------------------------------------------------
# Blocks of the joint covariance L L' of x and y, positive definite by construction.
function xprofile_covariances(rng, ::Type{T}, n) where {T}
    L = [T(0.3)I zeros(T, n, n); zeros(T, n, n) T(0.6)I] .+ T(0.01) .* randn(rng, T, 2n, 2n)
    Σ = L * L'
    Σ[1:n, 1:n], Σ[1:n, n+1:end], Σ[n+1:end, n+1:end]
end

for T in (Float64,), n in (64, 256)
    rng = MersenneTwister(13)
    xcol = reshape(rand(rng, T, n) .+ 1, :, 1)
    y = (xcol[:, 1] .^ 2 .- 1) .+ T(0.4) .* randn(rng, T, n)
    Σxx, Σxy, Σyy = xprofile_covariances(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)
    l = XProfileDenseLikelihood(xcol, y, Σxx, Σxy, Σyy)
    p = [T(1.0), T(-1.0)]
    g = zeros(T, 2)
    case = "$T / $n obs"

    SUITE["xprofile dense"]["nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["xprofile dense"]["nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    SUITE["xprofile dense"]["nll + grad fd"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad_fd!($g, $l, $model, p) end setup=(p=copy($p))
end

# ---------------------------------------------------------------------------
# Intrinsic scatter, switched opposite to each likelihood's default
# ---------------------------------------------------------------------------
for T in (Float64,), n in (1024,)
    rng = MersenneTwister(41)
    X = reshape(rand(rng, T, n) .* T(4), :, 1) .- T(2)
    y = T(2) .* vec(X) .- T(1) .+ T(0.1) .* randn(rng, T, n)
    sy = T(0.05) .+ T(0.1) .* rand(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    p = [T(1.0), T(0.0), var_param(T(0.1)^2)]
    g = zeros(T, 3)
    case = "$T / $n obs"
    for (name, l) in (("gaussian", gaussian_scatter(X, y, sy)),
                      ("laplace", laplace_scatter(X, y, sy)))
        SUITE["intrinsic scatter"]["$name nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
        SUITE["intrinsic scatter"]["$name nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
    end

    x = rand(rng, T, n) .+ T(2)
    age = sqrt.(T(1) .+ T(0.2) .* x)
    mcc = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
    lcc = CosmicChronometerLikelihood(x, age, T(0.1) .* ones(T, n); intrinsic_scatter = true)
    pcc = [T(1.0), T(0.2), var_param(T(0.05)^2)]
    SUITE["intrinsic scatter"]["cosmic nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($lcc, $mcc, p) end setup=(p=copy($pcc))
    SUITE["intrinsic scatter"]["cosmic nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $lcc, $mcc, p) end setup=(p=copy($pcc))

    gbar = rand(rng, T, n) .+ T(0.1)
    mrar = SymRegInterpreter.to_logspace_xy_model(SymRegLikelihoods.Model(T,
        Expr(:->, :(x, p), :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))), 3))
    lrar = XUniformDiagonalLikelihood(log10.(gbar), T(0.05)*rand(rng, T, n),
                                      log10.(rand(rng, T, n) .+ T(0.1)), T(0.05)*rand(rng, T, n);
                                      intrinsic_scatter = true)
    plrar = SymRegLikelihoods.prepare(lrar, mrar)
    prar = [T(0.8), T(-0.02), T(0.38), var_param(T(0.05)^2)]
    grar = zeros(T, 4)
    SUITE["intrinsic scatter"]["xuniform diagonal nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($plrar, p) end setup=(p=copy($prar))
    SUITE["intrinsic scatter"]["xuniform diagonal nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($grar, $plrar, p) end setup=(p=copy($prar))

    xobs = rand(rng, T, n) .+ T(1)
    lmnr = MNRDiagonalLikelihood(xobs, T(0.2) .* rand(rng, T, n),
                                 (2 .* xobs .- T(0.5)) .+ T(0.1) .* randn(rng, T, n),
                                 T(0.15) .* rand(rng, T, n); intrinsic_scatter = false)
    pmnr = [T(2.0), T(-0.5), T(1.5), var_param(T(1.2)^2)]
    gmnr = zeros(T, 4)
    SUITE["intrinsic scatter"]["mnr diagonal off nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($lmnr, $model, p) end setup=(p=copy($pmnr))
    SUITE["intrinsic scatter"]["mnr diagonal off nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($gmnr, $lmnr, $model, p) end setup=(p=copy($pmnr))
end

for T in (Float64,), n in (64,)
    rng = MersenneTwister(43)
    xcol = reshape(rand(rng, T, n) .+ 1, :, 1)
    y = (xcol[:, 1] .^ 2 .- 1) .+ T(0.4) .* randn(rng, T, n)
    Σxx, Σxy, Σyy = xprofile_covariances(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)
    l = XProfileDenseLikelihood(xcol, y, Σxx, Σxy, Σyy; intrinsic_scatter = true)
    p = [T(1.0), T(-1.0), var_param(T(0.3)^2)]
    g = zeros(T, 3)
    case = "$T / $n obs"
    SUITE["intrinsic scatter"]["xprofile dense nll"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll($l, $model, p) end setup=(p=copy($p))
    SUITE["intrinsic scatter"]["xprofile dense nll + grad"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.evaluate_nll_grad!($g, $l, $model, p) end setup=(p=copy($p))
end

SUITE

# ---------------------------------------------------------------------------
# Observed Fisher information (the description length's information matrix)
# ---------------------------------------------------------------------------
for T in (Float64,), n in (1024,)
    rng = MersenneTwister(47)
    X = reshape(rand(rng, T, n) .* T(4), :, 1) .- T(2)
    y = T(2) .* vec(X) .- T(1) .+ T(0.1) .* randn(rng, T, n)
    model = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*exp(p[2]*x[1]) + p[3])), 3)
    p0 = [T(1.0), T(0.3), T(-1.0)]
    case = "$T / $n obs"
    for (name, l, p) in (("gaussian known", GaussianLikelihood(X, y, T(0.2)), p0),
                         ("gaussian profiled", gaussian_profiled(X, y), profiled(p0, T(0.01))),
                         ("laplace profiled", laplace_profiled(X, y), profiled(p0, T(0.01))),
                         ("gaussian scatter", gaussian_scatter(X, y, T(0.05)), [p0; var_param(T(0.1)^2)]))
        SUITE["fim"][name]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.information_matrix($l, $model, p) end setup=(p=copy($p))
    end

    xobs = rand(rng, T, n) .+ T(1)
    lmnr = MNRDiagonalLikelihood(xobs, T(0.2) .* rand(rng, T, n),
                                 (2 .* xobs .- T(0.5)) .+ T(0.1) .* randn(rng, T, n),
                                 T(0.15) .* rand(rng, T, n))
    mmnr = SymRegLikelihoods.Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    pmnr = [T(2.0), T(-0.5), var_param(T(0.3)^2), T(1.5), var_param(T(1.2)^2)]
    SUITE["fim"]["mnr diagonal"]["$case"] = @benchmarkable with_fit_buffer_cache() do; SymRegLikelihoods.information_matrix($lmnr, $mmnr, p) end setup=(p=copy($pmnr))
end

SUITE
using SymRegLikelihoods
using SymRegInterpreter
using SymRegInterpreter: Model, code, differentiate, to_logspace_xy_model,
    with_fit_buffer_cache, EltypeMismatchError
using SymRegLikelihoods: AbstractProfiledLikelihood, AbstractLeastSquaresLikelihood,
    PreparedGeneric, prepare, positive_params, n_opt_params, optimizer_parameters,
    natural_parameters, evaluate_nll_grad_fd!, evaluate_profiled_nll,
    evaluate_profiled_nll_grad!, profile!, has_intrinsic_scatter, intrinsic_scatter,
    residual!, residual_jacobian!, residual_and_jacobian!, least_squares_complete!,
    observation_scatter, pairwise_confidence_regions_laplace,
    create_diffcache, DiagonalDiffCache, BidiagonalDiffCache, SymTridiagonalDiffCache,
    TridiagonalDiffCache
using Test
using ForwardDiff
using Random
using LinearAlgebra
using PreallocationTools
using DelimitedFiles
using Distributions

X = reshape(Float64[1, 2, 3, 4], :, 1)
y = 1.5 .* vec(X) .- 0.7
model = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)

const LaplaceLike = Union{LaplaceLikelihood, LaplaceProfiledLikelihood, LaplaceScatterLikelihood}

@testset "Gaussian fixed sigma" begin
    l = GaussianLikelihood(X, y, 0.4)
    @test n_observations(l) == 4
    @test n_likelihood_params(l) == 0
    @test n_param(l, model) == 2
    p = [1.5, -0.7]
    nll = evaluate_nll(l, model, p)
    ypred = X[:, 1] * p[1] .+ p[2]
    wvals = 1/0.4 .* ones(4)
    # NLL = 1/2 sum(r.^2) + n/2 log(2pi) - sum(log(w)),  r = (ypred-y).*w
    expected = 1/2 * sum(abs2, (ypred .- y) .* wvals) + 4/2 * log(2*pi) - sum(log, wvals)
    @test nll ≈ expected atol=1e-10
    @test nll ≈ evaluate_nll(l, model, p)

    grad = zeros(2)
    nll2 = evaluate_nll_grad!(grad, l, model, p)
    fd = ForwardDiff.gradient(pp -> evaluate_nll(l, model, pp), p)
    @test grad ≈ fd atol=1e-8
    @test nll2 ≈ nll
end

@testset "Gaussian profiled sigma" begin
    # noisy target so the profiled residual variance is positive and finite
    rng = MersenneTwister(7)
    y_n = y .+ 0.05f0 .* randn(rng, 4)
    l = GaussianProfiledLikelihood(X, y_n)
    @test n_likelihood_params(l) == 1   # sigma² is a parameter ...
    @test n_param(l, model) == 3
    @test n_opt_params(l, model) == 2   # ... profiled out of the fit
    p = [1.5, -0.7]
    ypred = X[:, 1] * p[1] .+ p[2]
    rss = sum(abs2, ypred .- y_n)

    # the full NLL at any sigma², which is the profile NLL at sigma² = RSS/n
    @test evaluate_nll(l, model, [p; 0.02]) ≈ rss / (2 * 0.02) + 4/2 * log(2*pi*0.02) rtol=1e-10
    θ = natural_parameters(l, model, p)
    @test θ ≈ [p; rss / 4]
    @test evaluate_nll(l, model, θ) ≈ 4/2 * (1 + log(2*pi) + log(rss / 4)) rtol=1e-8
    @test evaluate_profiled_nll(l, model, p) ≈ evaluate_nll(l, model, θ)

    grad = zeros(3)
    nll2 = evaluate_nll_grad!(grad, l, model, [p; 0.02])
    fd = ForwardDiff.gradient(pp -> evaluate_nll(l, model, pp), [p; 0.02])
    @test grad ≈ fd rtol=1e-10
    @test nll2 ≈ evaluate_nll(l, model, [p; 0.02])

    # a variance that is not positive is the worst loss
    @test evaluate_nll(l, model, [p; 0.0]) == floatmax(Float64)
    @test evaluate_nll(l, model, [p; -1.0]) == floatmax(Float64)
end

@testset "Laplace fixed scale" begin
    # noisy target: the Laplace NLL is not differentiable at zero residuals
    rng = MersenneTwister(1)
    y_n = y .+ 0.05 .* randn(rng, 4)
    l = LaplaceLikelihood(X, y_n, 0.5)
    @test n_likelihood_params(l) == 0
    @test n_param(l, model) == 2
    p = [1.5, -0.7]
    nll = evaluate_nll(l, model, p)
    ypred = X[:, 1] * p[1] .+ p[2]
    expected = sum(abs, ypred .- y_n) / 0.5 + 4 * log(2 * 0.5)
    @test nll ≈ expected atol=1e-8
    grad = zeros(2)
    evaluate_nll_grad!(grad, l, model, p)
    @test grad ≈ ForwardDiff.gradient(pp -> evaluate_nll(l, model, pp), p) atol=1e-6
end

@testset "Laplace profiled scale" begin
    # noisy target so the profiled scale (mean absolute residual) is positive
    rng = MersenneTwister(3)
    y_n = y .+ 0.05 .* randn(rng, 4)
    l = LaplaceProfiledLikelihood(X, y_n)
    @test n_likelihood_params(l) == 1   # b² is a parameter ...
    @test n_param(l, model) == 3
    @test n_opt_params(l, model) == 2   # ... profiled out of the fit
    p = [1.5, -0.7]
    ypred = X[:, 1] * p[1] .+ p[2]
    sae = sum(abs, ypred .- y_n)

    # the full NLL at any b², which is the profile NLL at b = mean |e|
    @test evaluate_nll(l, model, [p; 0.09]) ≈ sae / 0.3 + 4 * log(2 * 0.3) rtol=1e-10
    θ = natural_parameters(l, model, p)
    b = sae / 4
    @test θ ≈ [p; b^2]
    @test evaluate_nll(l, model, θ) ≈ 4 * (1 + log(2 * b)) rtol=1e-8
    @test evaluate_profiled_nll(l, model, p) ≈ evaluate_nll(l, model, θ)

    grad = zeros(3)
    nll2 = evaluate_nll_grad!(grad, l, model, [p; 0.09])
    fd = ForwardDiff.gradient(pp -> evaluate_nll(l, model, pp), [p; 0.09])
    @test grad ≈ fd rtol=1e-10
    @test nll2 ≈ evaluate_nll(l, model, [p; 0.09])
end

@testset "Dimension checks" begin
    l = GaussianLikelihood(X, y, 0.4)
    @test_throws DimensionMismatch evaluate_nll(l, model, [1.0])  # wrong p length
    @test_throws DimensionMismatch evaluate_nll(l, model, [1.0, 2.0, 3.0])
    @test_throws DimensionMismatch evaluate_nll(LaplaceProfiledLikelihood(X, y), model, [1.0, 2.0])
end

# At the profiled MLE the full NLL is stationary in φ, so the profile NLL and
# its gradient are the full NLL and its β-block.
@testset "profiled likelihoods" begin
    rng = MersenneTwister(5)
    y_n = y .+ 0.05 .* randn(rng, 4)
    p = [1.5, -0.7]
    for l in (GaussianProfiledLikelihood(X, y_n), LaplaceProfiledLikelihood(X, y_n))
        θ = natural_parameters(l, model, p)
        @test optimizer_parameters(l, model, θ) == p
        g = zeros(3)
        nll = evaluate_nll_grad!(g, l, model, θ)
        @test abs(g[3]) < 1e-10 * maximum(abs, g)

        @test evaluate_profiled_nll(l, model, p) ≈ nll
        gp = zeros(2)
        @test evaluate_profiled_nll_grad!(gp, l, model, p) ≈ nll
        @test gp ≈ g[1:2]
        @test gp ≈ ForwardDiff.gradient(q -> evaluate_profiled_nll(l, model, q), p)

        # the generic (slow) path
        gs = zeros(2)
        @test invoke(evaluate_profiled_nll_grad!,
                     Tuple{AbstractVector, AbstractProfiledLikelihood, Model, AbstractVector},
                     gs, l, model, p) ≈ nll
        @test gs ≈ gp
        @test invoke(evaluate_profiled_nll, Tuple{AbstractProfiledLikelihood, Model, AbstractVector},
                     l, model, p) ≈ nll
    end
end

@testset "information_matrix includes the profiled parameter" begin
    rng = MersenneTwister(7)
    y_n = y .+ 0.05 .* randn(rng, 4)
    p = [1.5, -0.7]
    n = length(y_n)
    ypred = X[:, 1] * p[1] .+ p[2]

    # fixed noise parameter -> only the model parameters are counted
    @test size(information_matrix(GaussianLikelihood(X, y_n, 0.4), model, p)) == (2, 2)
    @test size(information_matrix(LaplaceLikelihood(X, y_n, 0.5), model, p)) == (2, 2)

    # profiled Gaussian sigma2: d2/ds2^2 [n/2 log(2 pi s2) + rss/(2 s2)]
    #                         = rss/s2^3 - n/(2 s2^2)
    lg = GaussianProfiledLikelihood(X, y_n)
    θg = natural_parameters(lg, model, p)
    FIg = information_matrix(lg, model, θg)
    rss = sum(abs2, ypred .- y_n)
    s2 = rss / n
    @test θg[3] ≈ s2
    @test size(FIg) == (3, 3)
    @test FIg ≈ FIg'
    @test FIg[3, 3] ≈ rss / s2^3 - n / (2 * s2^2) rtol=1e-8
    _, J = interpret_jac(model, X, p)
    @test FIg[1:2, 1:2] ≈ (J' * J) ./ s2 rtol=1e-6

    # profiled Laplace scale, counted as v = b²: d2/dv^2 [sae/sqrt(v) + n log(2 sqrt(v))]
    # = 3 sae/(4 v^(5/2)) - n/(2 v^2), which at the MLE sae = n*b is n / (4 b^4).
    ll = LaplaceProfiledLikelihood(X, y_n)
    θl = natural_parameters(ll, model, p)
    FIl = information_matrix(ll, model, θl)
    b = sum(abs, ypred .- y_n) / n
    @test size(FIl) == (3, 3)
    @test FIl ≈ FIl'
    @test θl[3] ≈ b^2
    @test FIl[3, 3] ≈ n / (4 * b^4) rtol=1e-8
end

@testset "information_matrix symmetrizes in place" begin
    # exactly symmetric, not only to a tolerance
    p = [1.5, -0.7]
    y_n = y .+ 0.05 .* randn(MersenneTwister(7), 4)
    for (l, θ) in ((GaussianLikelihood(X, y_n, 0.4), p), (GaussianProfiledLikelihood(X, y_n), [p; 0.01]),
                   (LaplaceLikelihood(X, y_n, 0.5), p), (LaplaceProfiledLikelihood(X, y_n), [p; 0.01]))
        FI = information_matrix(l, model, θ)
        @test FI == FI'
    end

    # and of the likelihood's element type
    X32 = Float32.(X)
    m32 = Model(Float32, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    l32 = GaussianLikelihood(X32, Float32.(y_n), 0.4f0)
    FI32 = information_matrix(l32, m32, Float32[1.5, -0.7])
    @test eltype(FI32) == Float32
    @test FI32 == FI32'
end

@testset "information_matrix is the Hessian in the natural parameters" begin
    # away from any optimum, so that every gradient entry is nonzero
    rng = MersenneTwister(41)
    n, d = 24, 2
    X2 = [exp.(randn(rng, n)) .* 2  0.7 .* exp.(randn(rng, n))]
    xerr = sqrt.(0.02 .+ 0.1 .* rand(rng, n, d))
    yerr = 0.05 .+ 0.2 .* rand(rng, n)
    yobs = 1.3 .* X2[:, 1].^2 .+ 0.5 .* sin.(X2[:, 2]) .- 0.8 .+ 0.3 .* randn(rng, n)
    m2 = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*sin(x[2]*p[3]) + p[4])), 4)
    pm = [1.3, 0.5, 1.0, -0.8]
    lik = [0.4^2, 0.3, -0.5, 1.1^2, 0.8^2]    # σ², μ, w²

    x1 = rand(rng, n) .* 3 .+ 0.5
    m1 = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*sin(x[1]*p[3]) + p[4])), 4)
    y1 = 0.8 .* x1.^2 .+ 0.3 .* sin.(1.2 .* x1) .+ 0.1 .+ 0.2 .* randn(rng, n)
    xc = rand(rng, n) .+ 2
    mc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1] + p[3]*x[1]^2)), 3)
    age = sqrt.(1.0 .+ 0.2 .* xc .+ 0.05 .* xc.^2) .+ 0.02 .* randn(rng, n)
    np_ = 10
    Xp = reshape(rand(rng, np_) .+ 1, :, 1)
    yp = Xp[:, 1].^2 .- 1 .+ 0.4 .* randn(rng, np_)
    A = Matrix(0.3I, np_, np_) + 0.01randn(rng, np_, np_)
    B = Matrix(0.6I, np_, np_) + 0.01randn(rng, np_, np_)
    lprof = XProfileDenseLikelihood(Xp, yp, A * A', 0.01randn(rng, np_, np_), B * B')
    mp = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*exp(-p[3]*x[1]) + p[4])), 4)

    cases = [
        ("Gaussian, yerr", GaussianLikelihood(X2, yobs, yerr), m2, pm),
        ("Gaussian, profiled", GaussianProfiledLikelihood(X2, yobs), m2, [pm; 0.3]),
        ("Gaussian, yerr + σint", GaussianScatterLikelihood(X2, yobs, yerr), m2, [pm; 0.3^2]),
        ("Laplace, fixed b", LaplaceLikelihood(X2, yobs, 0.3), m2, pm),
        ("Laplace, profiled", LaplaceProfiledLikelihood(X2, yobs), m2, [pm; 0.3]),
        ("Laplace, yerr + σint", LaplaceScatterLikelihood(X2, yobs, yerr), m2, [pm; 0.2^2]),
        ("XUniformDiagonal + σint",
         XUniformDiagonalLikelihood(x1, fill(0.1, n), y1, fill(0.2, n); intrinsic_scatter = true),
         m1, [0.8, 0.3, 1.2, 0.1, 0.1^2]),
        ("Cosmic chronometer", CosmicChronometerLikelihood(xc, age, fill(0.1, n)), mc, [1.0, 0.2, 0.05]),
        ("MNRDiagonal", MNRDiagonalLikelihood(X2, xerr, yobs, yerr), m2, [pm; lik]),
        ("MNRBanded", MNRBandedLikelihood(X2, yobs, Diagonal(vec(xerr .^ 2)), zeros(n * d, n),
                                          SymTridiagonal(yerr .^ 2, zeros(n - 1))), m2, [pm; lik]),
        ("MNRDense", MNRDenseLikelihood(X2, yobs, Matrix(Diagonal(vec(xerr .^ 2))),
                                        zeros(n * d, n), Matrix(Diagonal(yerr .^ 2))), m2, [pm; lik]),
        ("XProfileDense", lprof, mp, [1.0, 0.1, 0.5, -1.0]),
    ]
    with_fit_buffer_cache() do
        for (tag, l, m, p0) in cases
            p = p0 .+ 0.07 .* (1:length(p0)) ./ length(p0)
            H = information_matrix(l, m, p)
            @test size(H) == (n_param(l, m), n_param(l, m))
            # the Laplace likelihoods use the expected Fisher information instead
            l isa LaplaceLike && continue
            @test H ≈ ForwardDiff.hessian(q -> evaluate_nll(l, m, q), p) rtol = 1e-10
        end
    end

    # the gradient rejects a different float type; `information_matrix` converts
    l32 = GaussianLikelihood(Float32.(X2), Float32.(yobs), Float32.(yerr))
    m32 = Model(Float32, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*sin(x[2]*p[3]) + p[4])), 4)
    @test_throws EltypeMismatchError evaluate_nll_grad!(zeros(4), l32, m32, pm)
    # a precision mismatch is a bug, which the optimizers must not swallow as a
    # numerically hard problem
    @test SymRegLikelihoods._is_bug(EltypeMismatchError(Float32, Float64))
    @test SymRegLikelihoods._is_bug(CapturedException(EltypeMismatchError(Float32, Float64), []))
    m64 = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*sin(x[2]*p[3]) + p[4])), 4)
    @test_throws MethodError optimize(LevenbergMarquardtOptimizer(), l32, m64, Float32.(pm))
    @test_throws CapturedException optimize(NLoptOptimizer(), l32, m64, Float32.(pm))
    @test_throws MethodError evaluate_nll_grad!(zeros(Float32, 4), l32, m32, pm)
    @test size(information_matrix(l32, m32, pm)) == (4, 4)
end

@testset "Laplace likelihoods use the expected Fisher information" begin
    # I_ββ = Σ J_iᵀ J_i / b_i², I_vv = Σ 1/(4 b_i⁴) for v = b², I_βv = 0
    rng = MersenneTwister(7)
    n = 30
    Xl = reshape(collect(range(0.0, 2.0; length = n)), :, 1)
    ml = Model(Float64, Expr(:->, :(x, p), :(p[1] * exp(p[2] * x[1]) + p[3])), 3)
    β = [1.2, 0.7, -0.4]
    f, J = interpret_jac(ml, Xl, β)
    sy = 0.1 .+ 0.2 .* rand(rng, n)
    cases = [
        ("fixed", y -> LaplaceLikelihood(Xl, y, sy), β, sy .^ 2),
        ("profiled", y -> LaplaceProfiledLikelihood(Xl, y), [β; 0.3^2], fill(0.3^2, n)),
        ("scatter", y -> LaplaceScatterLikelihood(Xl, y, sy), [β; 0.2^2], sy .^ 2 .+ 0.2^2),
    ]
    for (tag, mk, θ, b2) in cases
        I = information_matrix(mk(f), ml, θ)
        @test I[1:3, 1:3] ≈ J' * Diagonal(1 ./ b2) * J
        if length(θ) == 4
            @test I[4, 4] ≈ sum(1 ./ (4 .* b2 .^ 2))
            @test all(iszero, I[1:3, 4]) && all(iszero, I[4, 1:3])
        end
        # it does not depend on the observations
        @test information_matrix(mk(f .+ 1), ml, θ) == I

        # the average outer product of the score over data drawn from the model
        S = zeros(length(θ), length(θ))
        g = zeros(length(θ))
        R = 5000
        for _ in 1:R
            ysim = f .+ [rand(rng, Laplace(0.0, sqrt(b2[i]))) for i in 1:n]
            evaluate_nll_grad!(g, mk(ysim), ml, θ)
            S .+= g * g'
        end
        @test maximum(abs.(S ./ R .- I) ./ sqrt.(diag(I) * diag(I)')) < 0.05
    end

    # an invalid scale has no information matrix
    @test all(isnan, information_matrix(LaplaceProfiledLikelihood(Xl, f), ml, [β; 0.0]))
    @test all(isnan, information_matrix(LaplaceScatterLikelihood(Xl, f, sy), ml, [β; -0.1]))
end

@testset "the description length does not depend on the units of y" begin
    # For a linear model, scaling y by c maps p -> c p and σ² -> c² σ², which
    # leaves each term log|θ| + ½ log I_θθ unchanged.
    rng = MersenneTwister(43)
    n = 40
    Xu = reshape(rand(rng, n) .* 3, :, 1)
    mu_ = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    sy = 0.1 .+ 0.2 .* rand(rng, n)
    yu = 1.5 .* Xu[:, 1] .- 0.7 .+ sqrt.(sy .^ 2 .+ 0.09) .* randn(rng, n)
    pu = [1.4, -0.6]
    terms(l, θ) = log.(abs.(θ)) .+ log.(diag(information_matrix(l, mu_, θ))) ./ 2
    for (tag, mk, pt) in (
            ("Gaussian, σ_y + σ_int", c -> GaussianScatterLikelihood(Xu, c .* yu, c .* sy),
                                      (l, c) -> [c .* pu; (0.3c)^2]),
            ("Gaussian, profiled", c -> GaussianProfiledLikelihood(Xu, c .* yu),
                                   (l, c) -> natural_parameters(l, mu_, c .* pu)),
            ("Laplace, σ_y + b_int", c -> LaplaceScatterLikelihood(Xu, c .* yu, c .* sy),
                                     (l, c) -> [c .* pu; (0.3c)^2]),
            ("Laplace, profiled", c -> LaplaceProfiledLikelihood(Xu, c .* yu),
                                  (l, c) -> natural_parameters(l, mu_, c .* pu)))
        l1 = mk(1.0)
        ref = terms(l1, pt(l1, 1.0))
        for c in (1e-2, 1e2)
            lc = mk(c)
            @test terms(lc, pt(lc, c)) ≈ ref rtol = 1e-8
        end
    end
end

@testset "PreparedGeneric fallback" begin
    # noisy target: the Laplace NLL is not differentiable at exact zero residuals
    rng = MersenneTwister(9)
    y_n = y .+ 0.05 .* randn(rng, 4)
    l = LaplaceLikelihood(X, y_n, 0.5)
    pl = prepare(l, model)
    @test pl isa PreparedGeneric
    p = [1.5, -0.7]
    @test evaluate_nll(pl, p) ≈ evaluate_nll(l, model, p)
    grad = zeros(2); grad_fd = zeros(2)
    evaluate_nll_grad!(grad, pl, p)
    SymRegLikelihoods.evaluate_nll_grad_fd!(grad_fd, pl, p)
    @test grad ≈ grad_fd atol=1e-6
end

println("SymRegLikelihoods tests passed")

@testset "Structured caches" begin
    n = 6; k = 4
    d  = Diagonal(rand(n))
    bd = Bidiagonal(rand(n), rand(n-1), :L)
    st = SymTridiagonal(rand(n), rand(n-1))
    tr = Tridiagonal(rand(n-1), rand(n), rand(n-1))

    cd = create_diffcache(d, k); cb = create_diffcache(bd, k)
    cs = create_diffcache(st, k); ct = create_diffcache(tr, k)

    @test cd isa DiagonalDiffCache
    @test cb isa BidiagonalDiffCache
    @test cs isa SymTridiagonalDiffCache
    @test ct isa TridiagonalDiffCache

    # Dual access keeps the structured shape and element type
    TD = ForwardDiff.Dual{Nothing,Float64,4}
    gd = PreallocationTools.get_tmp(cd, TD)
    gb = PreallocationTools.get_tmp(cb, TD)
    gs = PreallocationTools.get_tmp(cs, TD)
    gt = PreallocationTools.get_tmp(ct, TD)
    @test gd isa Diagonal{TD}
    @test gb isa Bidiagonal{TD}
    @test gs isa SymTridiagonal{TD}
    @test gt isa Tridiagonal{TD}

    # primal access keeps original element type
    gp = PreallocationTools.get_tmp(cd, Float64)
    @test gp isa Diagonal{Float64}

    # every band is sized for a chunk larger than the length-derived default
    k2 = 8
    ct2 = create_diffcache(Tridiagonal(rand(n-1), rand(n), rand(n-1)), k2)
    @test length(ct2.du.dual_du) == length(ct2.dl.dual_du)
    TD2 = ForwardDiff.Dual{Nothing,Float64,k2}
    gt2 = @test_logs PreallocationTools.get_tmp(ct2, TD2)
    @test gt2 isa Tridiagonal{TD2}
end

@testset "XUniformDiagonal likelihood (input-adjoint)" begin
    # Reproduce the ESR RAR fixture at the fitted parameters.
    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    lgbar = log10.(gbar); lgobs = log10.(gobs)
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))

    l = XUniformDiagonalLikelihood(lgbar, elgbar, lgobs, elgobs)
    body = :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))
    model = Model(Float64, Expr(:->, :(x, p), body), 3)
    model = to_logspace_xy_model(model)
    p = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]
    nll = evaluate_nll(l, model, p)
    @test isfinite(nll)

    grad = zeros(3)
    nll2 = evaluate_nll_grad!(grad, l, model, p)
    @test nll2 ≈ nll
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, model, q), p) rtol=1e-6
end

@testset "prepared and unprepared value paths" begin
    # Away from the interpreter's guarded points the two derivative routes
    # agree, so preparing must not change the number.
    xs = [0.5, 1.0, 2.0, 3.0]
    ys = [1.1, 1.9, 3.2, 4.1]
    ms = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    ls = XUniformDiagonalLikelihood(xs, fill(0.1, 4), ys, fill(0.2, 4))
    ps = [1.0, 0.1]
    @test evaluate_nll(prepare(ls, ms), ps) ≈ evaluate_nll(ls, ms, ps) rtol=1e-12

    # At a guarded point (`sqrtabs` at zero) the routes differ, but the
    # prepared value and gradient must still agree.
    xd = [0.0, 1.0, 2.0, 3.0]
    md = Model(Float64, Expr(:->, :(x, p), :(p[1] * sqrtabs(x[1]) + p[2])), 2)
    ld = XUniformDiagonalLikelihood(xd, fill(0.1, 4), ys, fill(0.2, 4))
    pd = prepare(ld, md)

    gd = zeros(2)
    nll = evaluate_nll(pd, ps)
    @test nll == evaluate_nll_grad!(gd, pd, ps)
    @test nll == floatmax(Float64)
    @test all(iszero, gd)
end

@testset "XUniformDiagonal likelihood (prepared)" begin
    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    lgbar = log10.(gbar); lgobs = log10.(gobs)
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    l = XUniformDiagonalLikelihood(lgbar, elgbar, lgobs, elgobs)
    body = :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))
    model = Model(Float64, Expr(:->, :(x, p), body), 3)
    model = to_logspace_xy_model(model)
    p = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]

    pl = prepare(l, model)
    # the RAR data has a single input column, so there is no `varidx`
    @test_throws MethodError prepare(l, model; varidx = 2)
    @test 1 <= pl.fidx <= length(pl.dcode)
    @test 1 <= pl.dfidx <= length(pl.dcode)
    @test size(pl.x) == (length(l.x), 1)      # model input column
    # prepared NLL and gradient agree with the model-agnostic path and FD
    nll_p = evaluate_nll(pl, p)
    @test nll_p ≈ evaluate_nll(l, model, p)
    grad = zeros(3)
    nll2 = evaluate_nll_grad!(grad, pl, p)
    @test nll2 ≈ nll_p
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(pl, q), p) rtol=1e-6
    # repeated calls with fresh parameters stay consistent
    p2 = 0.9 .* p .+ 0.1
    grad2 = zeros(3)
    nll3 = evaluate_nll_grad!(grad2, pl, p2)
    @test nll3 ≈ evaluate_nll(pl, p2)
    @test grad2 ≈ ForwardDiff.gradient(q -> evaluate_nll(pl, q), p2) rtol=1e-6
    # optimizing through the prepared object evaluates the same objective
    r = optimize(NLoptOptimizer(), l, model, [0.84, -0.022, 0.38]; maxeval = 500)
    @test r isa OptResult
    @test r.f ≈ evaluate_nll(pl, r.x) rtol=1e-10
end

# check that batched jacobians work (data larger than batchsize)
@testset "RAR fixture + engine regression" begin
    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    @test length(gbar) > 128   # forces multiple batches

    l = XUniformDiagonalLikelihood(log10.(gbar), elgbar, log10.(gobs), elgobs)
    body = :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))
    model = Model(Float64, Expr(:->, :(x, p), body), 3)
    model = to_logspace_xy_model(model)
    p = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]
    nll = evaluate_nll(l, model, p)
    @test nll ≈ -1279.1 atol=1e-1
    # at the ESR parameters the gradient is small
    grad = zeros(3)
    evaluate_nll_grad!(grad, l, model, p)
    @test norm(grad, Inf) < 1.0
end

@testset "Optimizer" begin
    # y = 2 x - 1 with known sigma; optimize model params from a poor start
    Xf = reshape(Float64[1,2,3,4,5], :, 1)
    yf = 2.0 .* vec(Xf) .- 1.0
    mf = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    lf = GaussianLikelihood(Xf, yf, 0.5)
    res = optimize(lf, mf, [0.0, 0.0])
    @test isfinite(res.f)
    @test res.x[1] ≈ 2.0 atol=1e-3
    @test res.x[2] ≈ -1.0 atol=1e-3
    # objective improves over the start
    @test res.f <= evaluate_nll(lf, mf, [0.0, 0.0]) + 1e-6
end

@testset "Levenberg-Marquardt optimizer" begin
    Xf = reshape(Float64[1,2,3,4,5], :, 1)
    yf = 2.0 .* vec(Xf) .- 1.0
    mf = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    lf = GaussianLikelihood(Xf, yf, 0.5)   # fixed sigma (LSQ weights = 1/sigma)
    lp = GaussianProfiledLikelihood(Xf, yf) # profiled sigma (unit weights)

    # residuals and Jacobian
    F = zeros(5); J = zeros(5, 2)
    residual!(F, lf, mf, [1.0, 0.0])
    residual_jacobian!(J, lf, mf, [1.0, 0.0])
    F2 = zeros(5); J2 = zeros(5, 2)
    residual_and_jacobian!(F2, J2, lf, mf, [1.0, 0.0])
    @test F ≈ lf.w .* (Xf[:, 1] .- yf)          # p=[1,0] => pred = x
    @test F2 ≈ F
    @test J2 ≈ J
    @test J[:, 1] ≈ lf.w[1] .* vec(Xf[:, 1])    # d res/d p1 = w*x

    # the profiled sigma uses unit weights
    Fp = zeros(5); Jp = zeros(5, 2)
    residual!(Fp, lp, mf, [1.0, 0.0])
    residual_jacobian!(Jp, lp, mf, [1.0, 0.0])
    Fp2 = zeros(5); Jp2 = zeros(5, 2)
    residual_and_jacobian!(Fp2, Jp2, lp, mf, [1.0, 0.0])
    @test Fp ≈ Xf[:, 1] .- yf
    @test Jp ≈ [Xf[:, 1] ones(5)]
    @test Fp2 ≈ Fp
    @test Jp2 ≈ Jp
    @test J[:, 2] ≈ lf.w[1] .* ones(5)          # d res/d p2 = w

    # least_squares_complete! turns the residuals at β into the NLL at θ,
    # filling in the profiled sigma² = RSS/n
    θf = [1.0, 0.0]
    @test least_squares_complete!(copy(θf), lf, F) ≈ evaluate_nll(lf, mf, θf) atol=1e-9
    θp = [1.0, 0.0, NaN]
    residual!(F, lp, mf, θp[1:2])
    nllp = least_squares_complete!(θp, lp, F)
    @test θp[3] ≈ sum(abs2, F) / 5
    @test nllp ≈ evaluate_nll(lp, mf, θp) atol=1e-9
    @test nllp ≈ 5/2 * (1 + log(2*pi) + log(θp[3])) atol=1e-9

    # LM converges both sigma modes from a poor start
    opt = LevenbergMarquardtOptimizer()
    r1 = optimize(opt, lf, mf, [0.0, 0.0])
    @test r1.returnvalue == :SUCCESS
    @test r1.x[1] ≈ 2.0 atol=1e-3
    @test r1.x[2] ≈ -1.0 atol=1e-3
    @test r1.f ≈ evaluate_nll(lf, mf, r1.x) atol=1e-6   # loss == NLL at fitted params

    # equally accurate at every scale of the data
    let T = Float32
        for scale in (T(1e-3), T(1), T(1e3))
            Xs = reshape(T[1, 2, 3, 4, 5], :, 1)
            ys = scale .* (T(2) .* vec(Xs) .- T(1))
            ms = Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
            ls = GaussianLikelihood(Xs, ys, T(1))   # known sigma => unit weights
            rs = optimize(LevenbergMarquardtOptimizer(), ls, ms, T[0, 0])
            @test rs.x[1] ≈ 2 * scale rtol=1e-4
            @test rs.x[2] ≈ -scale rtol=1e-4
        end
    end

    # the start's sigma² is not read: it is profiled from the residuals.  The
    # noise is orthogonal to the columns [x 1], so the fit is still (2, -1),
    # but RSS > 0: an exact fit has an unbounded profiled NLL.
    yo = yf .+ 0.01 .* [1, -2, 0, 2, -1]
    lpo = GaussianProfiledLikelihood(Xf, yo)
    r2 = optimize(opt, lpo, mf, [0.0, 0.0, 123.0])
    @test r2.returnvalue == :SUCCESS
    @test r2.x[1] ≈ 2.0 atol=1e-3
    @test r2.x[2] ≈ -1.0 atol=1e-3
    @test length(r2.x) == 3
    @test r2.x[3] ≈ sum(abs2, interpret_vec(mf, Xf, r2.x) .- yo) / 5
    @test r2.f ≈ evaluate_nll(lpo, mf, r2.x) atol=1e-6

    # the same fit through NLopt takes the generic profiled path
    yfn = yf .+ 0.1 .* sin.(1:5)
    lpn = GaussianProfiledLikelihood(Xf, yfn)
    rlm = optimize(opt, lpn, mf, [0.0, 0.0, 1.0])
    rnl = optimize(NLoptOptimizer(), lpn, mf, [0.0, 0.0, 1.0])
    @test rnl.x ≈ rlm.x rtol=1e-5
    @test rnl.f ≈ rlm.f rtol=1e-8

    # Gaussian is an AbstractLeastSquaresLikelihood => default dispatch is LM
    @test optimize(lf, mf, [0.0, 0.0]) isa OptResult

    # wrong start length rejected
    @test_throws DimensionMismatch optimize(opt, lf, mf, [0.0])

    # constant model with a single parameter also converges
    mc = Model(Float64, Expr(:->, :(x, p), :(p[1])), 1)
    r3 = optimize(opt, GaussianLikelihood(Xf, yf, 0.5), mc, [0.0])
    @test r3.x[1] ≈ sum(yf) / length(yf) atol=1e-3   # best constant prediction is mean(y)
end

@testset "profiled scale is bounded below at an exact fit" begin
    for T in (Float64, Float32)
        Xe = reshape(T[1, 2, 3, 4, 5], :, 1)
        ye = T(2) .* vec(Xe) .- T(1)
        me = Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
        smin = eps(T) * sqrt(sum(abs2, ye) / 5)

        lg = GaussianProfiledLikelihood(Xe, ye)
        @test lg.s2min ≈ smin^2
        θ = T[2, -1, NaN]
        nll = least_squares_complete!(θ, lg, zeros(T, 5))
        @test θ[3] == lg.s2min
        @test nll ≈ 5 * log(2 * T(pi) * lg.s2min) / 2
        @test nll ≈ evaluate_nll(lg, me, θ)
        r = optimize(LevenbergMarquardtOptimizer(), lg, me, T[0, 0, 1])
        @test r.returnvalue == :SUCCESS
        @test isfinite(r.f) && r.f < 0
        @test r.x[1:2] ≈ [2, -1] rtol=sqrt(eps(T))
        @test r.x[3] >= lg.s2min

        ll = LaplaceProfiledLikelihood(Xe, ye)
        @test ll.bmin ≈ smin
        β = T[2, -1]
        @test profile!(T[β; NaN], ll, me)[3] == ll.bmin^2
        @test evaluate_profiled_nll(ll, me, β) ≈ 5 * log(2 * ll.bmin)
        g = zeros(T, 2)
        @test evaluate_profiled_nll_grad!(g, ll, me, β) ≈ evaluate_profiled_nll(ll, me, β)
        @test all(isfinite, g)
    end

    # an all-zero response still gets a positive bound
    @test GaussianProfiledLikelihood(ones(3, 1), zeros(3)).s2min == floatmin(Float64)
    @test LaplaceProfiledLikelihood(ones(3, 1), zeros(3)).bmin == sqrt(floatmin(Float64))
end

# A likelihood that throws whatever it is handed, used to check how the
# multi-start loop classifies exceptions.
struct ThrowingLikelihood{E} <: SymRegLikelihoods.AbstractLikelihood{Float64}
    ex::E
end
SymRegLikelihoods.n_observations(::ThrowingLikelihood) = 4
SymRegLikelihoods.n_likelihood_params(::ThrowingLikelihood) = 0
SymRegLikelihoods.evaluate_nll(l::ThrowingLikelihood, model::Model, p::AbstractVector) = throw(l.ex)
SymRegLikelihoods.evaluate_nll_grad!(g::AbstractVector, l::ThrowingLikelihood, model::Model,
                                     p::AbstractVector) = throw(l.ex)

@testset "multi-start optimize does not swallow bugs or interrupts" begin
    mt = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)

    # a numerically hard start is an ordinary outcome: every start fails and
    # the loop reports that it has nothing
    @test_throws ErrorException optimize(ThrowingLikelihood(DomainError(1.0)), mt;
                                         num_starts = 2, rng = MersenneTwister(1))

    # bugs and interrupts are rethrown
    caught(ex) = try
        optimize(ThrowingLikelihood(ex), mt; num_starts = 2, rng = MersenneTwister(1))
        nothing
    catch e
        e
    end

    # wrapped by NLopt
    for ex in (MethodError(sin, (1,)), UndefVarError(:nope))
        e = caught(ex)
        @test e isa CapturedException
        @test e.ex isa typeof(ex)
    end

    # NLopt returns `:FORCED_STOP` for an interrupt instead of rethrowing it
    @test caught(InterruptException()) isa InterruptException
end

@testset "two-argument optimize for a least-squares likelihood" begin
    Xt = reshape(Float64[1, 2, 3, 4, 5], :, 1)
    yt = 2.0 .* vec(Xt) .- 1.0
    mt = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    lt = GaussianLikelihood(Xt, yt, 0.5)

    r = optimize(lt, mt; num_starts = 3, rng = MersenneTwister(4))
    @test r.x ≈ [2.0, -1.0] atol = 1e-6
    @test r.f ≈ evaluate_nll(lt, mt, r.x) atol = 1e-8

    # the same fit as any other likelihood gets from the two-argument form
    rl = optimize(LaplaceLikelihood(Xt, yt, 0.3), mt; num_starts = 3,
                  rng = MersenneTwister(4))
    @test rl.x ≈ [2.0, -1.0] atol = 1e-4

    # with nothing to fit the single-evaluation shortcut still applies
    m0 = Model(Float64, Expr(:->, :(x, p), :(2 * x[1])), 0)
    @test n_param(lt, m0) == 0
    r0 = optimize(lt, m0)
    @test isempty(r0.x)
    @test r0.f ≈ evaluate_nll(lt, m0, Float64[]) atol = 1e-8
end

# Optimizers take and return the natural parameters but search in their own
# coordinates: log variances for a joint likelihood, the model parameters alone
# for a profiled one.
@testset "optimizer coordinates" begin
    rng = MersenneTwister(17)
    Xo = reshape(collect(range(0.5, 3.0; length = 20)), :, 1)
    yo = 1.5 .* vec(Xo) .- 0.7 .+ 0.2 .* randn(rng, 20)
    mo = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)

    # joint: positive likelihood parameters as their logarithm
    lj = LaplaceScatterLikelihood(Xo, yo, 0.1)
    θ = [1.4, -0.6, 0.04]
    @test n_opt_params(lj, mo) == 3
    u = optimizer_parameters(lj, mo, θ)
    @test u ≈ [1.4, -0.6, log(0.04)]
    @test natural_parameters(lj, mo, u) ≈ θ
    @test_throws DimensionMismatch optimizer_parameters(lj, mo, [1.4, -0.6])
    @test_throws DimensionMismatch natural_parameters(lj, mo, [1.4, -0.6])

    # the gradient in optimizer coordinates is the chain rule of the natural one
    pl = prepare(lj, mo)
    gu = zeros(3)
    SymRegLikelihoods._opt_nll_grad!(gu, lj, pl, u)
    @test gu ≈ ForwardDiff.gradient(q -> SymRegLikelihoods._opt_nll(lj, pl, q), u) rtol = 1e-8

    # profiled: the model parameters only, the rest from `profile!`
    lp = LaplaceProfiledLikelihood(Xo, yo)
    @test n_opt_params(lp, mo) == 2
    @test optimizer_parameters(lp, mo, [1.4, -0.6, 123.0]) == [1.4, -0.6]
    θp = natural_parameters(lp, mo, [1.4, -0.6])
    @test θp[3] ≈ (sum(abs, interpret_vec(mo, Xo, [1.4, -0.6]) .- yo) / 20)^2

    # a fit reports natural parameters, with the profiled one at its MLE
    r = optimize(lp, mo; num_starts = 3, rng = MersenneTwister(1))
    @test length(r.x) == 3
    @test r.x ≈ natural_parameters(lp, mo, r.x[1:2])
    @test r.f ≈ evaluate_nll(lp, mo, r.x)

    # random starts never produce an invalid variance
    rj = optimize(lj, mo; num_starts = 3, rng = MersenneTwister(1))
    @test rj.x[3] > 0
    @test rj.f ≈ evaluate_nll(lj, mo, rj.x)
end

# Evaluating a fit on new data uses the fitted noise parameters from θ.
@testset "evaluate_nll does not re-profile" begin
    rng = MersenneTwister(23)
    Xa = reshape(collect(range(0.5, 3.0; length = 30)), :, 1)
    ya = 1.5 .* vec(Xa) .- 0.7 .+ 0.2 .* randn(rng, 30)
    Xb = reshape(collect(range(0.6, 3.1; length = 15)), :, 1)
    yb = 1.5 .* vec(Xb) .- 0.7 .+ 0.5 .* randn(rng, 15)
    mo = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    for (ltrain, ltest) in ((GaussianProfiledLikelihood(Xa, ya), GaussianProfiledLikelihood(Xb, yb)),
                            (LaplaceProfiledLikelihood(Xa, ya), LaplaceProfiledLikelihood(Xb, yb)))
        θ = optimize(ltrain, mo, [1.0, 0.0, 1.0]).x
        # the test-set NLL at the fitted variance is worse than at the
        # variance the test residuals would imply
        θtest = natural_parameters(ltest, mo, θ[1:2])
        @test θtest[3] > 2 * θ[3]
        @test evaluate_nll(ltest, mo, θ) > evaluate_nll(ltest, mo, θtest)
    end
    lg = GaussianProfiledLikelihood(Xb, yb)
    θ = optimize(GaussianProfiledLikelihood(Xa, ya), mo, [1.0, 0.0, 1.0]).x
    r = interpret_vec(mo, Xb, θ) .- yb
    @test evaluate_nll(lg, mo, θ) ≈ sum(abs2, r) / (2 * θ[3]) + 15 / 2 * log(2pi * θ[3])
end

@testset "Diagonal MNR likelihood" begin
    # y = f(x) + noise with x errors; model f = p1*x + p2
    n = 40
    rng = MersenneTwister(3)
    xobs = rand(rng, n) .+ 1
    xerr = 0.2 .* rand(rng, n)
    sigma = 0.3; mugauss = 1.5; wgauss = 1.2
    f_true = 2.0 .* xobs .- 0.5
    yobs = f_true .+ sigma .* randn(rng, n)
    yerr = 0.15 .* rand(rng, n)
    xerr2 = xerr.^2; yerr2 = yerr.^2

    l = MNRDiagonalLikelihood(xobs, xerr, yobs, yerr)
    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
    @test n_param(l, mm) == 5   # 2 model + sigma², mu_gauss, w²
    @test n_likelihood_params(l) == 3
    @test collect(positive_params(l)) == [1, 3]

    p = [2.0, -0.5, sigma^2, mugauss, wgauss^2]
    nll = evaluate_nll(l, mm, p)
    @test isfinite(nll)

    grad = zeros(5)
    nll2 = evaluate_nll_grad!(grad, l, mm, p)
    @test nll2 ≈ nll
    # gradient must match ForwardDiff and central differences on every parameter
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, mm, q), p) rtol=1e-6
    step = 1e-6
    for j in eachindex(p)
        p1 = copy(p); p1[j] += step
        p2 = copy(p); p2[j] -= step
        fd = (evaluate_nll(l, mm, p1) - evaluate_nll(l, mm, p2)) / (2*step)
        @test grad[j] ≈ fd rtol=1e-4
    end

    # overflowing or negative variances are rejected with floatmax
    for (k, v) in ((3, 1e20), (5, 1e20), (3, -0.1), (5, -0.1))
        bad = copy(p); bad[k] = v
        @test evaluate_nll(l, mm, bad) == floatmax(Float64)
    end

    # wrong parameter length rejected
    @test_throws DimensionMismatch evaluate_nll(l, mm, [1.0, 2.0, 3.0])
end

@testset "Diagonal MNR fixture (raw-derivative convention)" begin
    # ESR MNR fixture; the logspace transformation is applied to the model, and
    # >128 rows span several batches.
    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    @test length(gbar) > 128
    Z = log10.(gbar)
    l = MNRDiagonalLikelihood(Z, elgbar, log10.(gobs), elgobs)
    body = :(p[1] * abs(x[1])^(-1 / (p[2] + abs(p[3] + x[1])^p[4])))
    model = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p), body), 4))
    # p = [model params; sigma², mu_gauss, w²] at the ESR fixture values
    p = [-1.7302245173940842, -2.40586258570395, -0.017892186956077034,
         0.051284150873232376, 0.07915900803213895^2, -0.5024228473024372,
         0.7397713386530679^2]
    nll = evaluate_nll(l, model, p)
    @test isfinite(nll)
    @test nll ≈ 1464.219863790258 atol=1e-4
    # gradient consistent with ForwardDiff across the full parameter vector
    grad = zeros(7)
    evaluate_nll_grad!(grad, l, model, p)
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, model, q), p) rtol=1e-6
end

@testset "Multivariate diagonal MNR likelihood" begin
    # d = 2 independent x variables
    n, d = 30, 2
    rng = MersenneTwister(41)
    X = [exp.(randn(rng, n)) .* 2  0.8 .* exp.(randn(rng, n))]   # raw-scale helper
    Z = log10.(X)                                                 # observed x; model input
    Xerr = 0.15 .* rand(rng, n, d)
    sigma = 0.3; mugauss = [0.4, -0.2]; wgauss = [1.1, 0.7]
    yobs = (2.0 .* Z[:, 1] .- 1.0 .* Z[:, 2] .+ 0.5) .+ sigma .* randn(rng, n)
    yerr = 0.12 .* rand(rng, n)
    l = MNRDiagonalLikelihood(Z, Xerr, yobs, yerr)
    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2]*x[2] + p[3])), 3)
    @test n_observations(l) == n
    @test n_likelihood_params(l) == 1 + 2d
    @test n_param(l, mm) == 3 + 1 + 2d

    p = [2.0, -1.0, 0.5, sigma^2, mugauss..., (wgauss .^ 2)...]
    nll = evaluate_nll(l, mm, p)
    @test isfinite(nll)
    grad = zeros(8)
    nll2 = evaluate_nll_grad!(grad, l, mm, p)
    @test nll2 ≈ nll
    # analytic gradient must agree with ForwardDiff and central differences
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, mm, q), p) rtol=1e-6
    step = 1e-6
    for j in eachindex(p)
        p1 = copy(p); p1[j] += step
        p2 = copy(p); p2[j] -= step
        fd = (evaluate_nll(l, mm, p1) - evaluate_nll(l, mm, p2)) / (2*step)
        @test grad[j] ≈ fd rtol=1e-4
    end

    # prepared path differentiates over all d inputs in one code vector
    pl = prepare(l, mm)
    @test evaluate_nll(pl, p) ≈ nll rtol=1e-12
    g2 = zeros(8)
    @test evaluate_nll_grad!(g2, pl, p) ≈ nll
    @test g2 ≈ grad rtol=1e-12
    @test g2 ≈ ForwardDiff.gradient(q -> evaluate_nll(pl, q), p) rtol=1e-6

    # d >= 4 inputs seed the reverse sweep through the general (non-Val) path
    let n4 = 25, d4 = 4, rng4 = MersenneTwister(43)
        Z4 = randn(rng4, n4, d4)
        Xerr4 = 0.1 .+ 0.1 .* rand(rng4, n4, d4)
        m4 = Model(Float64, Expr(:->, :(x, p),
            :(p[1]*x[1] + p[2]*x[2]^2 + p[3]*sin(x[3]) + x[4]*x[1] + p[4])), 4)
        y4 = Z4[:, 1] .- 0.5 .* Z4[:, 2] .^ 2 .+ 0.8 .* sin.(Z4[:, 3]) .+ Z4[:, 4] .* Z4[:, 1] .+ 0.2 .+
             0.2 .* randn(rng4, n4)
        l4 = MNRDiagonalLikelihood(Z4, Xerr4, y4, 0.1 .+ 0.05 .* rand(rng4, n4))
        p4 = [1.0, -0.5, 0.8, 0.2, 0.2^2, zeros(d4)..., ones(d4)...]
        g4 = zeros(length(p4))
        @test evaluate_nll_grad!(g4, l4, m4, p4) ≈ evaluate_nll(l4, m4, p4)
        @test g4 ≈ ForwardDiff.gradient(q -> evaluate_nll(l4, m4, q), p4) rtol=1e-6
    end

    # data shape mismatches rejected
    @test_throws DimensionMismatch MNRDiagonalLikelihood(Z, Xerr, yobs, yerr[1:end-1])
    @test_throws DimensionMismatch MNRDiagonalLikelihood(Z, Xerr[1:end-1, :], yobs, yerr)
end

@testset "Multivariate diagonal MNR == banded MNR (independent x)" begin
    # With diagonal Σxx and zero Σxy, MNRDiagonalLikelihood, MNRBandedLikelihood and
    # MNRDenseLikelihood describe the same system.
    rng = MersenneTwister(5)
    n, d = 12, 2
    X = [exp.(randn(rng, n)) .* 2  0.7 .* exp.(randn(rng, n))]
    xerr2 = 0.02 .+ 0.1 .* rand(rng, n, d)
    sigma = 0.4; mugauss = [0.3, -0.5]; wgauss = [1.1, 0.8]
    yobs = (1.3 .* X[:, 1].^2 .+ 0.5 .* X[:, 2] .- 0.8) .+ sigma .* randn(rng, n)
    yerr2 = (0.05 .+ 0.2 .* rand(rng, n)).^2
    p = [1.3, 0.5, -0.8, sigma^2, mugauss..., (wgauss .^ 2)...]

    ls = MNRDiagonalLikelihood(X, sqrt.(xerr2), yobs, sqrt.(yerr2))
    ms = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2] + p[3])), 3)
    # Σxx diagonal in variable-major order (vec(X) convention)
    Σxx = Diagonal(vec(xerr2))
    lm = MNRBandedLikelihood(X, yobs, Σxx, zeros(n*d, n),
                             SymTridiagonal(vec(yerr2), zeros(n - 1)))
    ml = ms

    lmd = MNRDenseLikelihood(X, yobs, Matrix(Σxx), zeros(n * d, n),
                             Matrix(SymTridiagonal(vec(yerr2), zeros(n - 1))))

    ns = evaluate_nll(ls, ms, p)
    nm = evaluate_nll(lm, ml, p)
    nd_ = evaluate_nll(lmd, ml, p)
    @test ns ≈ nm atol=1e-6
    @test ns ≈ nd_ atol=1e-6
    # the constant cancels in the gradient, so the full gradient vectors agree
    gs = zeros(8); gm = zeros(8); gd = zeros(8)
    evaluate_nll_grad!(gs, ls, ms, p)
    evaluate_nll_grad!(gm, lm, ml, p)
    evaluate_nll_grad!(gd, lmd, ml, p)
    @test gs ≈ gm atol=1e-6
    @test gs ≈ gd atol=1e-6
end

@testset "Cosmic chronometer likelihood" begin
    rng = MersenneTwister(11)
    x = rand(rng, 30) .+ 2
    # model = p1 + p2*x  and H = sqrt(model); age must equal truth sqrt(model)
    age = sqrt.(1.0 .+ 0.2 .* x)
    sig = 0.1 .* ones(30)
    l = CosmicChronometerLikelihood(x, age, sig)
    mc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
    @test n_param(l, mc) == 2
    p = [1.0, 0.2]
    nll = evaluate_nll(l, mc, p)
    @test isfinite(nll)
    # exact fit => all residuals ~ 0
    @test nll ≈ 0.0 atol=1e-4
    grad = zeros(2)
    evaluate_nll_grad!(grad, l, mc, p)
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, mc, q), p) atol=1e-6
end

@testset "Dense MNR likelihood" begin
    rng = MersenneTwister(5)
    n = 12
    X = [randn(rng, n) .* 2 0.7 .* randn(rng, n)]
    f_true = X[:, 1] .^ 2 .- 1
    sigma = 0.4; mugauss = [0.3, -0.4]; wgauss = [1.1, 0.8]
    y = f_true .+ sigma .* randn(rng, n)
    Σxx = Matrix(0.5I, 2n, 2n) + 0.02*randn(rng, 2n, 2n); Σxx = Σxx*Σxx'
    Σyy = Matrix(0.7*I, n, n) + 0.02*randn(rng, n, n); Σyy = Σyy*Σyy'
    Σxy = 0.02*randn(rng, 2n, n)

    l = MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy)
    # model f = p1*x[1]^2 + p2*x[2] + p3
    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2] + p[3])), 3)
    @test n_param(l, mm) == 8
    p = [1.0, 0.0, -1.0, sigma^2, mugauss..., (wgauss .^ 2)...]
    nll = evaluate_nll(l, mm, p)
    @test isfinite(nll)

    grad = zeros(8)
    nll2 = evaluate_nll_grad!(grad, l, mm, p)
    @test nll2 ≈ nll
    # analytic gradient must agree with ForwardDiff and central differences
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, mm, q), p) rtol=1e-6
    step = 1e-6
    for j in eachindex(p)
        p1 = copy(p); p1[j] += step
        p2 = copy(p); p2[j] -= step
        fd = (evaluate_nll(l, mm, p1) - evaluate_nll(l, mm, p2)) / (2*step)
        @test grad[j] ≈ fd rtol=1e-4
    end
end

@testset "Banded MNR likelihood == dense" begin
    rng = MersenneTwister(9)
    n = 10
    x = randn(rng, n) .* 2
    y = (x .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    Σxx = Diagonal(rand(rng, n) .+ 0.5)
    Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, n - 1) .* 0.4)
    Σxy = Bidiagonal(rand(rng, n) .* 0.3 .- 0.15, rand(rng, n - 1) .* 0.3 .- 0.15, :L)
    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)

    ls = MNRBandedLikelihood(x, y, Σxx, Σxy, Σyy)
    ld = MNRDenseLikelihood(reshape(x, :, 1), y, Matrix(Σxx), Matrix(Σxy), Matrix(Σyy))
    p = [1.0, -1.0, 0.4^2, 0.3, 1.1^2]
    ns = evaluate_nll(ls, mm, p)
    nd = evaluate_nll(ld, mm, p)
    @test ns ≈ nd atol=1e-8          # structured == dense (ESR's own check)
    @test isfinite(ns)

    gs = zeros(5); gd = zeros(5)
    evaluate_nll_grad!(gs, ls, mm, p)
    evaluate_nll_grad!(gd, ld, mm, p)
    @test gs ≈ gd atol=1e-8
    @test gs ≈ ForwardDiff.gradient(q -> evaluate_nll(ls, mm, q), p) atol=1e-6
end


@testset "Multivariate banded MNR == dense & FD" begin
    rng = MersenneTwister(31)
    n, d = 10, 2
    nd = n * d
    X = [randn(rng, n) .* 2  randn(rng, n)]
    y = (X[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    # block-diagonal Σxx (per-variable n x n blocks) and block-column Σxy,
    # variable-major order (vec(X) convention)
    Σxx = Diagonal(rand(rng, nd) .+ 0.5)
    Σxy = zeros(nd, n)
    for i in 1:n, k in 1:d
        Σxy[(k-1)*n+i, i] = 0.2 * (rand(rng) - 0.5)
    end
    Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, n - 1) .* 0.4)

    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
    ls = MNRBandedLikelihood(X, y, Σxx, Σxy, Σyy)
    ld = MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy)
    p = [1.0, -1.0, 0.4^2, 0.3, -0.2, 1.1^2, 0.8^2]
    @test n_likelihood_params(ls) == 1 + 2d
    @test n_param(ls, mm) == 2 + 1 + 2d
    @test evaluate_nll(ls, mm, p) ≈ evaluate_nll(ld, mm, p) atol=1e-8
    gs = zeros(7); gd = zeros(7); gf = zeros(7)
    evaluate_nll_grad!(gs, ls, mm, p)
    evaluate_nll_grad!(gd, ld, mm, p)
    gf .= ForwardDiff.gradient(q -> evaluate_nll(ls, mm, q), p)
    @test gs ≈ gd atol=1e-8
    @test gs ≈ gf atol=1e-6
end

@testset "symmetric tridiagonal kernels" begin
    # the kernels of the banded MNR fast path against dense linear algebra
    rng = MersenneTwister(77)
    for n in (1, 2, 5, 9)
        Sd = rand(rng, n) .* 2 .+ 3.0              # diagonally dominant, so the
        Se = rand(rng, max(n - 1, 0)) .* 0.5       # unpivoted LDL' is stable
        S = Matrix(SymTridiagonal(Sd, Se))
        ld = zeros(n); ll = zeros(max(n - 1, 0))
        @test SymRegLikelihoods._sym_tridiag_ldlt!(ld, ll, Sd, Se)
        L = Matrix(1.0I, n, n)
        for i in 1:n-1
            L[i+1, i] = ll[i]
        end
        @test L * Diagonal(ld) * L' ≈ S

        b = randn(rng, n); v = zeros(n)
        SymRegLikelihoods._sym_tridiag_solve!(v, ld, ll, b)
        @test v ≈ S \ b

        # the tridiagonal part of S⁻¹
        Zd = zeros(n); Z1 = zeros(max(n - 1, 0))
        SymRegLikelihoods._tridiag_selected_inverse!(Zd, Z1, ld, ll)
        Z = inv(S)
        @test Zd ≈ diag(Z)
        @test Z1 ≈ [Z[i, i+1] for i in 1:n-1]
    end

    # a singular or non-finite system is reported, not thrown
    @test !SymRegLikelihoods._sym_tridiag_ldlt!(zeros(2), zeros(1), [0.0, 1.0], [1.0])
    @test !SymRegLikelihoods._sym_tridiag_ldlt!(zeros(2), zeros(1), [1.0, 1.0], [1.0])
    @test !SymRegLikelihoods._sym_tridiag_ldlt!(zeros(2), zeros(1), [NaN, 1.0], [1.0])
end

@testset "banded MNR fast path == dense for every allowed Σxy" begin
    # Σxy blocks bidiagonal below, above, diagonal, and on different sides per variable
    rng = MersenneTwister(2024)
    for (tag, n, d, sxy) in (
            ("d=1 lower", 10, 1, :lower),
            ("d=1 upper", 10, 1, :upper),
            ("d=1 diagonal", 10, 1, :diag),
            ("d=2 mixed sides", 9, 2, :mixed),
            ("d=3 lower", 8, 3, :lower),
            ("n=1", 1, 2, :diag))
        nd = n * d
        X = reduce(hcat, [randn(rng, n) .* 2 for _ in 1:d])
        y = (X[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
        Σxx = Diagonal(rand(rng, nd) .+ 0.5)
        Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, max(n - 1, 0)) .* 0.4)
        Σxy = zeros(nd, n)
        for k in 1:d, i in 1:n
            Σxy[(k - 1) * n + i, i] = 0.2 * (rand(rng) - 0.5)
            side = sxy === :lower ? -1 :
                   sxy === :upper ? 1 :
                   sxy === :mixed ? (isodd(k) ? -1 : 1) : 0
            j = i + side
            (side != 0 && 1 <= j <= n) &&
                (Σxy[(k - 1) * n + i, j] = 0.1 * (rand(rng) - 0.5))
        end

        # both parameters are used whatever d is, or `Model` rejects the code
        expr = foldl((a, k) -> :($a + $(Expr(:call, :*, :(p[2]), :(x[$k])))),
                     2:d; init = :(p[1] * x[1]^2 + p[2]))
        m = Model(Float64, Expr(:->, :(x, p), expr), 2)
        p = [1.0; -1.0; 0.4^2; fill(0.3, d); fill(1.1^2, d)]

        ls = MNRBandedLikelihood(X, y, Σxx, Σxy, Σyy)
        ldn = MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy)
        @test evaluate_nll(ls, m, p) ≈ evaluate_nll(ldn, m, p) rtol=1e-10

        np = length(p)
        gs = zeros(np); gd = zeros(np)
        evaluate_nll_grad!(gs, ls, m, p)
        evaluate_nll_grad!(gd, ldn, m, p)
        @test gs ≈ gd rtol=1e-8
        @test gs ≈ ForwardDiff.gradient(q -> evaluate_nll(ls, m, q), p) rtol=1e-6
    end
end

@testset "banded MNR == dense on small systems" begin
    # MNRBandedLikelihood is a fast path for structured covariances, so on every
    # structure it accepts it must reproduce MNRDenseLikelihood: the NLL (also
    # through `prepare`), the analytic gradient, its derivative in
    # `information_matrix`, and the rejection of invalid variances.  Each
    # variable gets its own mu, w² and Σxy side, and the input partials differ
    # per variable and depend nonlinearly on the parameters.
    bodies = Dict(
        1 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[1])), 3),
        2 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[2]) + p[4] * x[1] * x[2]), 4),
        3 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[2]) + p[4] * x[1] * x[3]), 4))
    # side of the off-diagonal in each n x n block of Σxy: -1 below, +1 above, 0 none
    sides_per_d = Dict(1 => ([-1], [0], [1]), 2 => ([-1, 1], [1, 0]), 3 => ([-1, 0, 1], [1, 1, -1]))
    rng = MersenneTwister(4711)
    for d in 1:3, sides in sides_per_d[d], n in (1, 2, 3, 5), scatter in (true, false)
        tag = "n=$n d=$d sides=$sides scatter=$scatter"
        nd = n * d
        X = randn(rng, n, d)
        body, nmod = bodies[d]
        m = Model(Float64, Expr(:->, :(x, p), body), nmod)
        β = 0.5 .+ rand(rng, nmod)
        y = interpret_vec(m, X, β) .+ 0.3 .* randn(rng, n)

        Σxx = Diagonal(0.5 .+ rand(rng, nd))
        Σyy = SymTridiagonal(1.0 .+ rand(rng, n), 0.3 .* (rand(rng, max(n - 1, 0)) .- 0.5))
        Σxy = zeros(nd, n)
        for k in 1:d, i in 1:n
            Σxy[(k - 1) * n + i, i] = 0.3 * (rand(rng) - 0.5)
            j = i + sides[k]
            (sides[k] != 0 && 1 <= j <= n) && (Σxy[(k - 1) * n + i, j] = 0.3 * (rand(rng) - 0.5))
        end

        lb = MNRBandedLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = scatter)
        ld = MNRDenseLikelihood(X, y, Matrix(Σxx), Σxy, Matrix(Σyy); intrinsic_scatter = scatter)
        p = [β; (scatter ? [0.1 + 0.2rand(rng)] : Float64[]);
             randn(rng, d); 0.5 .+ rand(rng, d)]     # [β; σ²; mu_1..mu_d; w_1²..w_d²]
        np = length(p)
        @test np == n_param(lb, m) == n_param(ld, m)

        nll_d = evaluate_nll(ld, m, p)
        @test isfinite(nll_d)
        @test evaluate_nll(lb, m, p) ≈ nll_d rtol = 1e-10
        @test evaluate_nll(prepare(lb, m), p) ≈ nll_d rtol = 1e-10

        gb = zeros(np); gd = zeros(np)
        @test evaluate_nll_grad!(gb, lb, m, p) ≈ nll_d rtol = 1e-10
        evaluate_nll_grad!(gd, ld, m, p)
        @test gb ≈ gd rtol = 1e-8

        @test information_matrix(lb, m, p) ≈ information_matrix(ld, m, p) rtol = 1e-7

        # an invalid variance gives the worst loss and a zero gradient in both
        pbad = copy(p); pbad[end] = -1.0
        @test evaluate_nll(lb, m, pbad) == evaluate_nll(ld, m, pbad) == floatmax(Float64)
        @test evaluate_nll_grad!(gb, lb, m, pbad) == floatmax(Float64)
        @test all(iszero, gb)
    end
end

@testset "MNR likelihoods stay accurate for a wide latent prior (large w²)" begin
    # Forward-difference derivatives of a noisy trajectory: x and ydot share
    # their noise sources, so Σxy is lower bidiagonal and the joint covariance
    # is nearly singular.  For w² g² ≫ Σyy the banded and the dense likelihood
    # used to lose all digits there (NLL ≈ -1e10 or -1e31 instead of +1e4),
    # which optimizers exploit; the diagonal one lost digits in its derivatives.
    rng = MersenneTwister(17)
    n = 30
    t = range(0, 3, length = n + 1)
    lvl = 0.05
    xobs = 10 .* exp.(-0.4 .* t) .* (1 .+ lvl .* randn(rng, n + 1))
    dt = diff(t)
    s = lvl^2 .* xobs .^ 2
    x = xobs[1:end-1]
    y = diff(xobs) ./ dt
    Σxx = Diagonal(s[1:end-1] .+ 1e-6)
    Σyy = SymTridiagonal((s[1:end-1] .+ s[2:end]) ./ dt .^ 2 .+ 1e-6,
                         -s[2:end-1] ./ (dt[1:end-1] .* dt[2:end]))
    Σxy = Bidiagonal(-s[1:end-1] ./ dt, s[2:end-1] ./ dt[1:end-1], :L)
    l = MNRBandedLikelihood(x, y, Σxx, Σxy, Σyy)
    ld = MNRDenseLikelihood(reshape(x, :, 1), y, Matrix(Σxx), Matrix(Σxy), Matrix(Σyy))
    lg = MNRDiagonalLikelihood(x, sqrt.(Σxx.diag), y, sqrt.(Σyy.dv))
    m = Model(Float64, :((x, p) -> p[1] * x[1]), 1)

    # exact NLL of the dense system M = [Σxx+W  Σxy+WJ; (Σxy+WJ)'  Σyy+JWJ+σ²I]
    function exact_nll(a, sig2, mu, w2)
        R = BigFloat
        W = R(w2) * I(n); J = R(a) * I(n)
        Sxy = R.(Matrix(Σxy))
        M = [R.(Matrix(Σxx)) + W   Sxy + W * J;
             (Sxy + W * J)'        R.(Matrix(Σyy)) + J * W * J + R(sig2) * I(n)]
        z = [R(mu) .- R.(x); R(a) .* R(mu) .- R.(y)]   # f + J(mu - x) - y with f = a x
        F = cholesky(Symmetric(M))
        Float64((logdet(F) + 2n * log(2 * R(pi)) + dot(z, F \ z)) / 2)
    end

    setprecision(BigFloat, 256) do
        for (a, sig2, mu, w2) in ((-0.4, 0.01, 6.0, 5.0),
                                  (-3.0, 1e-4, 6.0, 1e10),
                                  (-2.8e4, 1e-18, 19.0, 4e13),
                                  (-6e8, 1e-6, 5.0, 1.5e13))
            p = [a, sig2, mu, w2]
            nll = exact_nll(a, sig2, mu, w2)
            @test evaluate_nll(l, m, p) ≈ nll rtol = 1e-9
            @test evaluate_nll(prepare(l, m), p) ≈ nll rtol = 1e-9
            g = zeros(4); gfd = zeros(4)
            @test evaluate_nll_grad!(g, l, m, p) ≈ nll rtol = 1e-9
            evaluate_nll_grad_fd!(gfd, l, m, p)
            @test g ≈ gfd rtol = 1e-8

            gd = zeros(4)
            @test evaluate_nll(ld, m, p) ≈ nll rtol = 1e-9
            @test evaluate_nll_grad!(gd, ld, m, p) ≈ nll rtol = 1e-9
            @test gd ≈ g rtol = 1e-8

            gg = zeros(4); ggfd = zeros(4)
            evaluate_nll_grad!(gg, lg, m, p)
            evaluate_nll_grad_fd!(ggfd, lg, m, p)
            @test ggfd ≈ gg rtol = 1e-8
        end
    end
end

@testset "XProfileDenseLikelihood with a singular covariance" begin
    # Forward differences without jitter: [Σxx Σxy; Σxy' Σyy] has rank n + 1 of
    # 2n, so all but one eigenvalue of its Schur complement C are zero, and
    # roundoff used to make some of them negative, with spurious poles of the
    # NLL at σ² = -λ > 0.
    rng = MersenneTwister(23)
    n = 30
    t = range(0, 3, length = n + 1)
    xobs = 10 .* exp.(-0.4 .* t) .* (1 .+ 0.05 .* randn(rng, n + 1))
    dt = diff(t)
    s = 0.05^2 .* xobs .^ 2
    X = reshape(xobs[1:end-1], :, 1)
    y = diff(xobs) ./ dt
    Σxx = Matrix(Diagonal(s[1:end-1]))
    Σyy = Matrix(SymTridiagonal((s[1:end-1] .+ s[2:end]) ./ dt .^ 2,
                                -s[2:end-1] ./ (dt[1:end-1] .* dt[2:end])))
    Σxy = Matrix(Bidiagonal(-s[1:end-1] ./ dt, s[2:end-1] ./ dt[1:end-1], :L))
    l = @test_logs (:warn, r"singular") XProfileDenseLikelihood(X, y, Σxx, Σxy, Σyy;
                                                                 intrinsic_scatter = true)
    @test all(>=(0), l.schur_eigvals)
    m = Model(Float64, :((x, p) -> p[1] * x[1]), 1)
    nlls = [evaluate_nll(l, m, [-0.4, s2]) for s2 in 10.0 .^ (-14:-2)]
    @test all(isfinite, nlls)

    # with a jitter the covariance is regular and nothing is reported
    @test_logs XProfileDenseLikelihood(X, y, Σxx + 1e-6I, Σxy, Σyy + 1e-6I;
                                       intrinsic_scatter = true)
    # an indefinite covariance is rejected
    @test_throws ArgumentError XProfileDenseLikelihood(X, y, Σxx, Σxy, Σyy - 10I;
                                                       intrinsic_scatter = true)
end

# Covariances of finite-difference data for MNRGeneralBandedLikelihood: inputs
# A x̃ and derivatives D x̃_k of a trajectory with independent noise variances s,
# for forward differences paired with x_i or with the midpoint, and central
# differences.
function _fd_stencil(scheme, t)
    N = length(t)
    n = scheme === :central ? N - 2 : N - 1
    A = zeros(n, N); D = zeros(n, N)
    for i in 1:n
        if scheme === :central
            A[i, i + 1] = 1
            D[i, i] = -1 / (t[i + 2] - t[i]); D[i, i + 2] = 1 / (t[i + 2] - t[i])
        else
            if scheme === :forward
                A[i, i] = 1
            else
                A[i, i] = A[i, i + 1] = 0.5
            end
            D[i, i] = -1 / (t[i + 1] - t[i]); D[i, i + 1] = 1 / (t[i + 1] - t[i])
        end
    end
    A, D
end

function _fd_mnr_data(scheme, t, xobs, s, k; jitter = 1e-8)
    A, D = _fd_stencil(scheme, t)
    n, d = size(A, 1), size(xobs, 2)
    Σxx = zeros(n * d, n * d); Σxy = zeros(n * d, n)
    for j in 1:d
        rg = (j - 1) * n + 1:j * n
        Σxx[rg, rg] = A * Diagonal(s[:, j]) * A' + jitter * sum(s[:, j]) / size(s, 1) * I
        j == k && (Σxy[rg, :] = A * Diagonal(s[:, k]) * D')
    end
    Σyy = D * Diagonal(s[:, k]) * D'
    Σyy += jitter * tr(Σyy) / n * I
    A * xobs, D * xobs[:, k], Σxx, Σxy, Σyy
end

@testset "general banded MNR likelihood" begin
    rng = MersenneTwister(3)
    N = 30
    t = collect(range(0, 4, length = N))
    traj = [3 .* exp.(-0.5 .* t) .+ 0.2  sin.(1.3 .* t)]
    xobs = traj .* (1 .+ 0.05 .* randn(rng, N, 2))
    s = 0.05^2 .* xobs .^ 2
    models = Dict(1 => Model(Float64, :((x, p) -> p[1] * x[1] + p[2] * sin(x[1])), 2),
                  2 => Model(Float64, :((x, p) -> p[1] * x[1] * x[2] + p[2] * x[2]), 2))

    # equal to the dense likelihood for every stencil, also for a wide prior
    for scheme in (:forward, :midpoint, :central), d in (1, 2), k in 1:d, scatter in (true, false)
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(scheme, t, xobs[:, 1:d], s[:, 1:d], k)
        lg = MNRGeneralBandedLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = scatter)
        ld = MNRDenseLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = scatter)
        @test lg.cov.h == (scheme === :central ? 2 : 1)
        @test n_param(lg, models[d]) == n_param(ld, models[d])
        for w2 in (2.0, 1e13)
            p = [-0.4, 0.3, (scatter ? [0.05] : Float64[])..., fill(0.5, d)..., fill(w2, d)...]
            nll = evaluate_nll(ld, models[d], p)
            @test evaluate_nll(lg, models[d], p) ≈ nll rtol = 1e-9
            g = zeros(length(p)); gd = zeros(length(p)); gfd = zeros(length(p))
            @test evaluate_nll_grad!(g, lg, models[d], p) ≈ nll rtol = 1e-9
            @test evaluate_nll(prepare(lg, models[d]), p) ≈ nll rtol = 1e-9
            evaluate_nll_grad!(gd, ld, models[d], p)
            @test g ≈ gd rtol = 1e-7
            evaluate_nll_grad_fd!(gfd, lg, models[d], p)
            @test g ≈ gfd rtol = 1e-7
            # the analytic gradient accepts duals (Jacobian = Fisher information)
            w2 < 10 && @test information_matrix(lg, models[d], p) ≈ information_matrix(ld, models[d], p) rtol = 1e-6
        end
        # an invalid variance gives the worst loss
        pbad = [-0.4, 0.3, (scatter ? [0.05] : Float64[])..., fill(0.5, d)..., fill(-1.0, d)...]
        @test evaluate_nll(lg, models[d], pbad) == floatmax(Float64)
    end

    # a weighted sum of covariance patterns Σ = Σ_r θ_r P_r (here the noise of
    # each variable as one pattern, plus the jitter) equals the summed matrices
    for scheme in (:midpoint, :central)
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(scheme, t, xobs, s, 2)
        pats = map(1:2) do j
            sj = zeros(size(s)); sj[:, j] = s[:, j]
            _fd_mnr_data(scheme, t, xobs, sj, 2; jitter = 0.0)[3:5]
        end
        jit = (Σxx - pats[1][1] - pats[2][1], Σxy - pats[1][2] - pats[2][2], Σyy - pats[1][3] - pats[2][3])
        θ = [0.7, 1.9]
        lw = MNRGeneralBandedLikelihood(X, y, [BandedCovariance(pats[1]...), BandedCovariance(pats[2]...),
                                               BandedCovariance(jit...)], [θ; 1.0])
        lm = MNRGeneralBandedLikelihood(X, y, (θ[1] .* pats[1] .+ θ[2] .* pats[2] .+ jit)...)
        p = [-0.4, 0.3, 0.05, 0.5, 0.5, 3.0, 3.0]
        @test lw.cov.h == lm.cov.h
        @test evaluate_nll(lw, models[2], p) ≈ evaluate_nll(lm, models[2], p) rtol = 1e-9
    end

    # equal to MNRBandedLikelihood for forward differences paired with x_i
    for d in (1, 2), k in 1:d
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(:forward, t, xobs[:, 1:d], s[:, 1:d], k)
        lg = MNRGeneralBandedLikelihood(X, y, Σxx, Σxy, Σyy)
        lb = MNRBandedLikelihood(X, y, Diagonal(diag(Σxx)), Σxy, SymTridiagonal(diag(Σyy), diag(Σyy, 1)))
        p = [-0.4, 0.3, 0.05, fill(0.5, d)..., fill(3.0, d)...]
        @test evaluate_nll(lg, models[d], p) ≈ evaluate_nll(lb, models[d], p) rtol = 1e-10
        @test information_matrix(lg, models[d], p) ≈ information_matrix(lb, models[d], p) rtol = 1e-6
    end

    # exact for a wide latent prior (reference: the dense likelihood in BigFloat)
    setprecision(BigFloat, 256) do
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(:midpoint, t, xobs[:, 1:1], s[:, 1:1], 1)
        lg = MNRGeneralBandedLikelihood(X, y, Σxx, Σxy, Σyy)
        lbig = MNRDenseLikelihood(big.(X), big.(y), big.(Σxx), big.(Σxy), big.(Σyy))
        m = Model(Float64, :((x, p) -> p[1] * x[1] + p[2] * x[1] * x[1]), 2)
        mb = Model(BigFloat, :((x, p) -> p[1] * x[1] + p[2] * x[1] * x[1]), 2)
        for (a, w2) in ((-0.4, 5.0), (-3e4, 1e13), (-6e8, 1e15))
            p = [a, 0.01, 1e-6, 1.5, w2]
            @test evaluate_nll(lg, m, p) ≈ Float64(evaluate_nll(lbig, mb, big.(p))) rtol = 1e-9
        end
    end
end

@testset "banded uniform-prior likelihood" begin
    rng = MersenneTwister(5)
    N = 30
    t = collect(range(0, 4, length = N))
    traj = [3 .* exp.(-0.5 .* t) .+ 0.2  sin.(1.3 .* t)]
    xobs = traj .* (1 .+ 0.05 .* randn(rng, N, 2))
    s = 0.05^2 .* xobs .^ 2
    models = Dict(1 => Model(Float64, :((x, p) -> p[1] * x[1] + p[2] * sin(x[1])), 2),
                  2 => Model(Float64, :((x, p) -> p[1] * x[1] * x[2] + p[2] * x[2]), 2))
    jacx(m, X, p) = (n = size(X, 1); f = zeros(n); J = zeros(n, size(X, 2));
                     SymRegInterpreter.interpret_jac!(f, nothing, J, m, X, p); (f, J))

    for scheme in (:forward, :midpoint, :central), d in (1, 2), k in 1:d, scatter in (true, false)
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(scheme, t, xobs[:, 1:d], s[:, 1:d], k)
        n = size(X, 1)
        lu = XUniformBandedLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = scatter)
        m = models[d]
        @test n_param(lu, m) == 2 + Int(scatter)
        p = [-0.4, 0.3, (scatter ? [0.05] : Float64[])...]
        # dense reference: C = D + σ²I, D = Σyy + J Σxx J' - J Σxy - (J Σxy)'
        f, g = jacx(m, X, p[1:2])
        J = zeros(n, n * d)
        for i in 1:n, kk in 1:d
            J[i, (kk - 1) * n + i] = g[i, kk]
        end
        JΣxy = J * Σxy
        C = Symmetric(Σyy + J * Σxx * J' - JΣxy - JΣxy' + (scatter ? p[3] : 0.0) * I)
        r = f - y
        nll = (logdet(C) + n * log(2π) + dot(r, C \ r)) / 2
        @test evaluate_nll(lu, m, p) ≈ nll rtol = 1e-10
        @test evaluate_nll(prepare(lu, m), p) ≈ nll rtol = 1e-10
        gr = zeros(length(p)); gfd = zeros(length(p))
        @test evaluate_nll_grad!(gr, lu, m, p) ≈ nll rtol = 1e-10
        evaluate_nll_grad_fd!(gfd, lu, m, p)
        @test gr ≈ gfd rtol = 1e-7
        # the analytic gradient accepts duals: Jacobian = Hessian of the NLL
        @test information_matrix(lu, m, p) ≈ ForwardDiff.hessian(q -> evaluate_nll(lu, m, q), p) rtol = 1e-6
        if scatter
            pbad = copy(p); pbad[3] = -1.0
            @test evaluate_nll(lu, m, pbad) == floatmax(Float64)
        end
    end

    # limit of the banded MNR likelihood for a wide latent prior: NLL differences agree
    X, y, Σxx, Σxy, Σyy = _fd_mnr_data(:midpoint, t, xobs, s, 2)
    lu = XUniformBandedLikelihood(X, y, Σxx, Σxy, Σyy)
    lg = MNRGeneralBandedLikelihood(X, y, Σxx, Σxy, Σyy)
    pa, pb = [-0.4, 0.3, 0.05], [0.2, -0.7, 0.02]
    mnr(p) = evaluate_nll(lg, models[2], [p; 0.5; 0.5; 1e9; 1e9])
    @test mnr(pa) - mnr(pb) ≈ evaluate_nll(lu, models[2], pa) - evaluate_nll(lu, models[2], pb) rtol = 1e-4

    # noise inference: covariance patterns with unknown weights θ (likelihood
    # parameters after σ²) give the same NLL as the patterns added with weight θ
    for scheme in (:midpoint, :central), scatter in (true, false)
        X, y, Σxx, Σxy, Σyy = _fd_mnr_data(scheme, t, xobs, s, 2)
        pats = map(1:2) do j
            sj = zeros(size(s)); sj[:, j] = s[:, j]
            BandedCovariance(_fd_mnr_data(scheme, t, xobs, sj, 2; jitter = 0.0)[3:5]...)
        end
        jit = BandedCovariance(Σxx, Σxy, Σyy) + (-1.0) * (pats[1] + pats[2])
        lf = XUniformBandedLikelihood(X, y, 0.8 * pats[1] + jit; intrinsic_scatter = scatter,
                                      free_patterns = [pats[2]])
        @test n_param(lf, models[2]) == 2 + Int(scatter) + 1
        θ = 1.7
        lk = XUniformBandedLikelihood(X, y, 0.8 * pats[1] + θ * pats[2] + jit; intrinsic_scatter = scatter)
        pk = [-0.4, 0.3, (scatter ? [0.05] : Float64[])...]
        pf = [pk; θ]
        @test evaluate_nll(lf, models[2], pf) ≈ evaluate_nll(lk, models[2], pk) rtol = 1e-10
        gr = zeros(length(pf)); gfd = zeros(length(pf))
        evaluate_nll_grad!(gr, lf, models[2], pf)
        evaluate_nll_grad_fd!(gfd, lf, models[2], pf)
        @test gr ≈ gfd rtol = 1e-7
        @test information_matrix(lf, models[2], pf) ≈ ForwardDiff.hessian(q -> evaluate_nll(lf, models[2], q), pf) rtol = 1e-6
        pbad = copy(pf); pbad[end] = -0.1
        @test evaluate_nll(lf, models[2], pbad) == floatmax(Float64)
    end

    # equal to XUniformDiagonalLikelihood for one input and diagonal covariances
    x = xobs[:, 1]; xerr = 0.05 .* abs.(x); yv = 0.5 .* x .+ 0.1 .* randn(rng, N); yerr = fill(0.1, N)
    ld = XUniformDiagonalLikelihood(x, xerr, yv, yerr; intrinsic_scatter = true)
    lu1 = XUniformBandedLikelihood(x, yv, Matrix(Diagonal(xerr .^ 2)), zeros(N, N), Matrix(Diagonal(yerr .^ 2)))
    p = [0.4, -0.2, 0.03]
    @test evaluate_nll(lu1, models[1], p) ≈ evaluate_nll(ld, models[1], p) rtol = 1e-12
end

@testset "MNRBandedLikelihood rejects a Σxy it cannot exploit" begin
    n, d = 6, 2
    nd = n * d
    X = [collect(1.0:n) collect(2.0:n+1)]
    y = collect(1.0:n)
    Σxx = Diagonal(fill(0.5, nd))
    Σyy = SymTridiagonal(fill(1.0, n), fill(0.1, n - 1))

    ok = zeros(nd, n)
    for k in 1:d, i in 1:n
        ok[(k - 1) * n + i, i] = 0.2
        i > 1 && (ok[(k - 1) * n + i, i - 1] = 0.05)
    end
    @test MNRBandedLikelihood(X, y, Σxx, ok, Σyy) isa MNRBandedLikelihood

    # two places off the diagonal: the Schur complement would be pentadiagonal
    far = copy(ok); far[1, 3] = 0.05
    @test_throws ArgumentError MNRBandedLikelihood(X, y, Σxx, far, Σyy)

    # both sides within one block: same problem
    both = copy(ok); both[1, 2] = 0.05
    @test_throws ArgumentError MNRBandedLikelihood(X, y, Σxx, both, Σyy)

    # the dense implementation takes either of them
    @test MNRDenseLikelihood(X, y, Σxx, far, Σyy) isa MNRDenseLikelihood

    # `Σxy` is a type parameter, so a Bidiagonal stays a Bidiagonal and the
    # field is not accessed through an abstract type
    x1 = collect(1.0:n)
    l1 = MNRBandedLikelihood(x1, y, Diagonal(fill(0.5, n)),
                             Bidiagonal(fill(0.2, n), fill(0.05, n - 1), :L), Σyy)
    @test l1.Σxy isa Bidiagonal{Float64,Vector{Float64}}
    @test isconcretetype(fieldtype(typeof(l1), :Σxy))
    @test isconcretetype(fieldtype(typeof(l1), :Σxx))
    @test isconcretetype(fieldtype(typeof(l1), :Σyy))
end

@testset "XProfileDense likelihood" begin
    rng = MersenneTwister(13)
    n = 8
    xcol = reshape(rand(rng, n) .+ 1, :, 1)
    y = (xcol[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    Σxx = Matrix(0.3*I, n, n) + 0.01*randn(rng, n, n); Σxx = Σxx*Σxx'
    Σyy = Matrix(0.6*I, n, n) + 0.01*randn(rng, n, n); Σyy = Σyy*Σyy'
    Σxy = 0.01*randn(rng, n, n)
    mm = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)
    l = XProfileDenseLikelihood(xcol, y, Σxx, Σxy, Σyy)
    p = [1.0, -1.0]
    nll = evaluate_nll(l, mm, p)
    @test isfinite(nll)
    grad = zeros(2)
    nll2 = evaluate_nll_grad!(grad, l, mm, p)
    @test nll2 ≈ nll
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(l, mm, q), p) atol=1e-5
    step = 1e-6
    for j in 1:2
        p1 = copy(p); p1[j] += step
        p2 = copy(p); p2[j] -= step
        fd = (evaluate_nll(l, mm, p1) - evaluate_nll(l, mm, p2)) / (2*step)
        @test grad[j] ≈ fd rtol=1e-4
    end
end

@testset "XProfileDense == reference on small systems" begin
    # XProfileDenseLikelihood assembles D from the n x d input partials without
    # forming J and splits off the σ² part of logdet(2πΣ).  The reference forms
    # J explicitly and evaluates Eq. 39 of arXiv:2309.00948 as written:
    #   nll = ½ logdet(2πΣ) + ½ r' D⁻¹ r,   r = f - y,
    #   D = Σyy + σ²I + JΣxxJ' - JΣxy - (JΣxy)',  Σ = [Σxx Σxy; Σxy' Σyy + σ²I]
    # Σxx is fully dense (including the blocks between variables), and the
    # covariances are drawn as one joint covariance, so that Σ is positive definite.
    function reference_nll(X, y, Σxx, Σxy, Σyy, m, β, s2)
        n, d = size(X)
        TE = promote_type(eltype(β), typeof(s2))
        f = zeros(TE, n); jacx = zeros(TE, n, d)
        interpret_jac!(f, nothing, jacx, m, X, β)
        J = zeros(TE, n, n * d)
        for k in 1:d, i in 1:n
            J[i, (k - 1) * n + i] = jacx[i, k]
        end
        Σyys = Σyy + s2 * I
        D = Σyys + J * Σxx * J' - J * Σxy - (J * Σxy)'
        Σ = [Σxx Σxy; Σxy' Σyys]
        r = f - y
        (logabsdet(2π * Σ)[1] + dot(r, D \ r)) / 2
    end

    bodies = Dict(
        1 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[1])), 3),
        2 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[2]) + p[4] * x[1] * x[2]), 4),
        3 => (:(p[1] * x[1]^2 + p[2] * sin(p[3] * x[2]) + p[4] * x[1] * x[3]), 4))
    rng = MersenneTwister(815)
    for d in 1:3, n in (1, 2, 3, 5), scatter in (true, false)
        nd = n * d
        X = randn(rng, n, d)
        body, nmod = bodies[d]
        m = Model(Float64, Expr(:->, :(x, p), body), nmod)
        β = 0.5 .+ rand(rng, nmod)
        y = interpret_vec(m, X, β) .+ 0.3 .* randn(rng, n)

        L = 0.4 .* randn(rng, nd + n, nd + n)
        Σ = Symmetric(L * L' + 0.5I)
        Σxx = Matrix(Σ[1:nd, 1:nd]); Σxy = Matrix(Σ[1:nd, nd+1:end]); Σyy = Matrix(Σ[nd+1:end, nd+1:end])

        l = XProfileDenseLikelihood(X, y, Σxx, Σxy, Σyy; intrinsic_scatter = scatter)
        s2 = scatter ? 0.1 + 0.2rand(rng) : 0.0
        p = scatter ? [β; s2] : β
        np = length(p)
        @test np == n_param(l, m)
        ref(q) = reference_nll(X, y, Σxx, Σxy, Σyy, m, q[1:nmod], scatter ? q[end] : 0.0)

        nll_r = ref(p)
        @test isfinite(nll_r)
        @test evaluate_nll(l, m, p) ≈ nll_r rtol = 1e-10
        @test evaluate_nll(prepare(l, m), p) ≈ nll_r rtol = 1e-10

        g = zeros(np)
        @test evaluate_nll_grad!(g, l, m, p) ≈ nll_r rtol = 1e-10
        @test g ≈ ForwardDiff.gradient(ref, p) rtol = 1e-8
        @test information_matrix(l, m, p) ≈ ForwardDiff.hessian(ref, p) rtol = 1e-7

        # a negative σ² gives the worst loss and a zero gradient
        if scatter
            pbad = copy(p); pbad[end] = -0.1
            @test evaluate_nll(l, m, pbad) == floatmax(Float64)
            @test evaluate_nll_grad!(g, l, m, pbad) == floatmax(Float64)
            @test all(iszero, g)
        end
    end
end

@testset "analytic gradient vs ForwardDiff fallback" begin
    rng = MersenneTwister(21)
    mlin = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    pp = [1.9, 0.4]

    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    lrar = XUniformDiagonalLikelihood(log10.(gbar), elgbar, log10.(gobs), elgobs)
    mrar = Model(Float64, Expr(:->, :(x, p), :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))), 3)
    mrar = to_logspace_xy_model(mrar)
    prar = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]

    # simple MNR on the fixture data (logspace model)
    Z = log10.(gbar)
    lmnr = MNRDiagonalLikelihood(Z, elgbar, log10.(gobs), elgobs)
    mmnr = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p),
        :(p[1] * abs(x[1])^(-1 / (p[2] + abs(p[3] + x[1])^p[4])))), 4))
    pmnrf = [-1.7302245173940842, -2.40586258570395, -0.017892186956077034,
             0.051284150873232376, 0.07915900803213895^2, -0.5024228473024372,
             0.7397713386530679^2]

    nx = 12
        xmm = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1]^2 + p[2])), 2)
    xcc = rand(rng, nx) .+ 2
    lcc = CosmicChronometerLikelihood(xcc, sqrt.(1.0 .+ 0.2 .* xcc), 0.1 .* ones(nx))
    mcc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)

    cases = [
        ("Gaussian fixed sigma", GaussianLikelihood(X, y, 0.4), mlin, pp),
        ("Gaussian profiled", GaussianProfiledLikelihood(X, y), mlin, [pp; 0.3]),
        ("Laplace fixed scalar", LaplaceLikelihood(X, y, 0.3), mlin, pp),
        ("Laplace per-point", LaplaceLikelihood(X, y, 0.1 .* (1 .+ abs.(randn(rng, size(X, 1))))), mlin, pp),
        ("Laplace profiled", LaplaceProfiledLikelihood(X, y), mlin, [pp; 0.3]),
        ("XUniformDiagonal", lrar, mrar, prar),
        ("Diagonal MNR", lmnr, mmnr, pmnrf),
        ("Cosmic chronometer", lcc, mcc, [1.0, 0.2]),
    ]
    for (tag, l, model, p) in cases
        nmod = n_param(l, model)
        ga = zeros(length(p))
        gf = zeros(length(p))
        nll_a = evaluate_nll_grad!(ga, l, model, p)
        nll_f = evaluate_nll_grad_fd!(gf, l, model, p)
        @test nll_a ≈ nll_f atol=1e-10
        @test ga ≈ gf rtol=1e-6 atol=1e-10
    end

    # prepared RAR object hits the same fast path as the model-agnostic call
    pl = prepare(lrar, mrar)
    ga = zeros(3); gf = zeros(3); gp = zeros(3)
    evaluate_nll_grad!(ga, pl, prar)
    evaluate_nll_grad!(gp, lrar, mrar, prar)
    evaluate_nll_grad_fd!(gf, lrar, mrar, prar)
    @test ga ≈ gp rtol=1e-10
    @test ga ≈ gf rtol=1e-6
end


@testset "ForwardDiff gradient and Hessian for all likelihoods" begin
    rng = MersenneTwister(123)
    X1 = reshape(Float64[1, 2, 3, 4], :, 1)
    y1 = 1.5 .* vec(X1) .- 0.7
    y1n = y1 .+ 0.05 .* randn(rng, length(y1))   # noisy: profiled sigma/scale > 0
    mlin = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)

    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    Z = log10.(gbar)
    lrar = XUniformDiagonalLikelihood(log10.(gbar), elgbar, log10.(gobs), elgobs)
    mrar = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p),
        :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))), 3))
    prar = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]
    lsmnr = MNRDiagonalLikelihood(Z, elgbar, log10.(gobs), elgobs)
    msmnr = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p),
        :(p[1] * abs(x[1])^(-1 / (p[2] + abs(p[3] + x[1])^p[4])))), 4))
    psmnr = [-1.7302245173940842, -2.40586258570395, -0.017892186956077034,
             0.051284150873232376, 0.07915900803213895^2, -0.5024228473024372,
             0.7397713386530679^2]

    n, d = 10, 2
    nd = n * d
    Xm = [randn(rng, n) .* 2  randn(rng, n)]
    ym = (Xm[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    Σxx = Diagonal(rand(rng, nd) .+ 0.5)
    Σxy = zeros(nd, n)
    for i in 1:n, k in 1:d
        Σxy[(k - 1) * n + i, i] = 0.2 * (rand(rng) - 0.5)
    end
    Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, n - 1) .* 0.4)
    mmv = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
    pmv = [1.0, -1.0, 0.4^2, 0.3, -0.2, 1.1^2, 0.8^2]
    lstr = MNRBandedLikelihood(Xm, ym, Σxx, Σxy, Σyy)
    ldense = MNRDenseLikelihood(Xm, ym, Matrix(Σxx), Σxy, Matrix(Σyy))

    xcc = rand(rng, n) .+ 2
    lcc = CosmicChronometerLikelihood(xcc, sqrt.(1.0 .+ 0.2 .* xcc), 0.1 .* ones(n))
    mcc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
    pcc = [1.0, 0.2]

    Σxxp = Matrix(0.3I, n, n) + 0.01*randn(rng, n, n); Σxxp = Σxxp * Σxxp'
    Σyyp = Matrix(0.6I, n, n) + 0.01*randn(rng, n, n); Σyyp = Σyyp * Σyyp'
    Σxyp = 0.01*randn(rng, n, n)
    Xp = reshape(rand(rng, n) .+ 1, :, 1)
    yp = (Xp[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    lprof = XProfileDenseLikelihood(Xp, yp, Σxxp, Σxyp, Σyyp)
    mprof = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)
    pprof = [1.0, -1.0]

    cases = [
        ("Gaussian fixed", GaussianLikelihood(X1, y1, 0.4), mlin, [1.5, -0.7]),
        ("Gaussian profiled", GaussianProfiledLikelihood(X1, y1n), mlin, [1.5, -0.7, 0.01]),
        ("Laplace fixed", LaplaceLikelihood(X1, y1, 0.3), mlin, [1.5, -0.7]),
        ("Laplace per-point", LaplaceLikelihood(X1, y1, [0.3, 0.4, 0.5, 0.6]), mlin, [1.5, -0.7]),
        ("Laplace profiled", LaplaceProfiledLikelihood(X1, y1n), mlin, [1.5, -0.7, 0.01]),
        ("XUniformDiagonal", lrar, mrar, prar),
        ("Diagonal MNR", lsmnr, msmnr, psmnr),
        ("MNR structured", lstr, mmv, pmv),
        ("MNR dense", ldense, mmv, pmv),
        ("XProfileDense", lprof, mprof, pprof),
        ("Cosmic chronometer", lcc, mcc, pcc),
    ]

    with_fit_buffer_cache() do
        for (tag, l, model, p) in cases
            g = ForwardDiff.gradient(q -> evaluate_nll(l, model, q), p)
            @test all(isfinite, g)
            H = information_matrix(l, model, p)
            @test size(H) == (length(p), length(p))
            @test all(isfinite, H)
            @test H ≈ H'
            # rtol because the RAR entries reach 7e7
            Hfd = ForwardDiff.hessian(θ -> evaluate_nll(l, model, θ), p)
            # the Laplace likelihoods use the expected Fisher information instead
            l isa LaplaceLike || @test H ≈ Hfd rtol=1e-10 atol=1e-8

            # a clamped NLL keeps the Dual element type
            pnan = copy(p)
            pnan[1] = NaN
            qnan = [ForwardDiff.Dual{Nothing}(v, 1.0) for v in pnan]
            nll_nan = evaluate_nll(l, model, qnan)
            @test nll_nan isa eltype(qnan)
            @test ForwardDiff.value(nll_nan) == floatmax(Float64)

            # a non-finite parameter (an unfactorizable system for the MNR and
            # profile likelihoods) is also the worst loss on the gradient path,
            # with a zero gradient
            for (i, v) in ((1, NaN), (1, Inf), (n_param(model), Inf))
                pbad = copy(p)
                pbad[i] = v
                gbad = fill(NaN, length(p))
                @test evaluate_nll_grad!(gbad, l, model, pbad) == floatmax(Float64)
                @test all(iszero, gbad)
            end
        end

        # an infinite prediction with finite input derivatives passes the
        # factorization of the banded MNR system and is caught on the NLL
        mmv3 = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2] + p[3])), 3)
        gbad = fill(NaN, length(pmv) + 1)
        @test evaluate_nll_grad!(gbad, lstr, mmv3, [pmv[1:2]; Inf; pmv[3:end]]) == floatmax(Float64)
        @test all(iszero, gbad)
    end
end

@testset "hot paths do not allocate temporary buffers" begin
    cases = let
        rng = MersenneTwister(321)
        X1 = reshape(Float64[1, 2, 3, 4], :, 1)
        y1 = 1.5 .* vec(X1) .- 0.7
        mlin = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1] + p[2])), 2)
        p1 = [1.5, -0.7]

        n, d = 10, 2
        nd = n * d
        Xm = [randn(rng, n) .* 2  randn(rng, n)]
        ym = (Xm[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
        Σxx = Diagonal(rand(rng, nd) .+ 0.5)
        Σxy = zeros(nd, n)
        for i in 1:n, k in 1:d
            Σxy[(k - 1) * n + i, i] = 0.2 * (rand(rng) - 0.5)
        end
        Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, n - 1) .* 0.4)
        mmv = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
        pmv = [1.0, -1.0, 0.4^2, 0.3, -0.2, 1.1^2, 0.8^2]
        lstr = prepare(MNRBandedLikelihood(Xm, ym, Σxx, Σxy, Σyy), mmv)
        ldense = MNRDenseLikelihood(Xm, ym, Matrix(Σxx), Σxy, Matrix(Σyy))

        Σxxp = Matrix(0.3I, n, n) + 0.01*randn(rng, n, n); Σxxp = Σxxp * Σxxp'
        Σyyp = Matrix(0.6I, n, n) + 0.01*randn(rng, n, n); Σyyp = Σyyp * Σyyp'
        Σxyp = 0.01*randn(rng, n, n)
        Xp = reshape(rand(rng, n) .+ 1, :, 1)
        yp = (Xp[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
        mprof = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)
        lprof = prepare(XProfileDenseLikelihood(Xp, yp, Σxxp, Σxyp, Σyyp), mprof)
        pprof = [1.0, -1.0]

        xcc = rand(rng, n) .+ 2
        mcc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
        lcc = CosmicChronometerLikelihood(xcc, sqrt.(1.0 .+ 0.2 .* xcc), 0.1 .* ones(n))
        pcc = [1.0, 0.2]

        [
            ("Gaussian fixed nll", () -> evaluate_nll(GaussianLikelihood(X1, y1, 0.4), mlin, p1)),
            ("Gaussian profiled nll", () -> evaluate_nll(GaussianProfiledLikelihood(X1, y1), mlin, [p1; 0.1])),
            ("Gaussian nll+grad", () -> evaluate_nll_grad!(zeros(2), GaussianLikelihood(X1, y1, 0.4), mlin, p1)),
            ("Laplace fixed nll", () -> evaluate_nll(LaplaceLikelihood(X1, y1, 0.3), mlin, p1)),
            ("Laplace profiled nll", () -> evaluate_nll(LaplaceProfiledLikelihood(X1, y1), mlin, [p1; 0.1])),
            ("Laplace nll+grad", () -> evaluate_nll_grad!(zeros(2), LaplaceLikelihood(X1, y1, 0.3), mlin, p1)),
            ("MNR structured nll", () -> evaluate_nll(lstr, pmv)),
            ("MNR structured nll+grad", () -> evaluate_nll_grad!(zeros(7), lstr, pmv)),
            ("MNR dense nll", () -> evaluate_nll(ldense, mmv, pmv)),
            ("XProfileDense nll", () -> evaluate_nll(lprof, pprof)),
            ("XProfileDense nll+grad", () -> evaluate_nll_grad!(zeros(2), lprof, pprof)),
            ("Cosmic nll", () -> evaluate_nll(lcc, mcc, pcc)),
            ("Cosmic nll+grad", () -> evaluate_nll_grad!(zeros(2), lcc, mcc, pcc)),
        ]
    end

    # tighter bounds where the allocation must not depend on n and d
    limits = Dict("MNR structured nll" => 256, "MNR structured nll+grad" => 768,
                  "Profile dense nll+grad" => 704)

    with_fit_buffer_cache() do
        for (tag, f) in cases
            f()                          # warm up the cache
            bytes = @allocated f()
            @test bytes <= get(limits, tag, 4096)
        end
    end
end

@testset "prediction_intervals on a grid different from the training data" begin
    Xtr = reshape(collect(range(0.5, 3.0; length = 12)), :, 1)
    ytr = 1.7 .* vec(Xtr) .+ 0.4
    mlin = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    l = GaussianLikelihood(Xtr, ytr, 0.05)
    popt = [1.7, 0.4]

    # a prediction grid with a different number of rows than the training data
    Xnew = reshape(collect(range(0.0, 4.0; length = 5)), :, 1)
    @test size(Xnew, 1) != n_observations(l)
    y_pred, low, high, resStdErr = prediction_intervals(l, mlin, Xnew, popt)

    @test length(y_pred) == size(Xnew, 1)
    @test length(low) == length(high) == length(resStdErr) == size(Xnew, 1)
    @test y_pred ≈ interpret_vec(mlin, Xnew, popt)
    @test all(isfinite, resStdErr)
    @test all(low .<= y_pred .<= high)

    # resStdErr is the delta-method standard error of the fitted *mean*:
    # sqrt(diag(J * cov * J')) with J = d(model)/d(p) at the rows of Xnew.
    cov = SymRegLikelihoods.covariance(l, mlin, popt)
    J = ForwardDiff.jacobian(q -> interpret_vec(mlin, Xnew, q) |> collect, popt)
    @test resStdErr ≈ sqrt.(diag(J * cov * J'))

    # the interval adds the likelihood's scatter in quadrature
    sig = observation_scatter(l, mlin, popt)
    @test sig ≈ 0.05                       # the sigma the likelihood was built with
    t = -quantile(TDist(n_observations(l) - length(popt)), 0.005)
    @test high .- y_pred ≈ t .* sqrt.(resStdErr .^ 2 .+ sig^2)
    @test y_pred .- low ≈ t .* sqrt.(resStdErr .^ 2 .+ sig^2)
    # narrower than the linear sum, wider than the mean-only band
    @test all(high .- y_pred .< t .* (resStdErr .+ sig))
    @test all(high .- y_pred .> t .* resStdErr)

    # a larger scatter widens the interval
    lwide = GaussianLikelihood(Xtr, ytr, 0.5)
    _, low2, high2, _ = prediction_intervals(lwide, mlin, Xnew, popt)
    @test all(low2 .< low) && all(high2 .> high)

    # a profiled sigma supplies the scatter it was fitted with
    lprof = GaussianProfiledLikelihood(Xtr, ytr .+ 0.1 .* sin.(1:12))
    @test observation_scatter(lprof, mlin, natural_parameters(lprof, mlin, popt)) ≈
          sqrt(sum(abs2, interpret_vec(mlin, Xtr, popt) .- lprof.y) / 12)

    # a Laplace(0, b) deviate has standard deviation b * sqrt(2)
    @test observation_scatter(LaplaceLikelihood(Xtr, ytr, 0.3), mlin, popt) ≈ 0.3 * sqrt(2)
    # b² = b_y² + b_int² with an intrinsic scatter, and the profiled b²
    @test observation_scatter(LaplaceScatterLikelihood(Xtr, ytr, 0.3), mlin, [popt; 0.4^2]) ≈
          sqrt(2 * (0.3^2 + 0.4^2))
    @test observation_scatter(LaplaceProfiledLikelihood(Xtr, ytr), mlin, [popt; 0.3^2]) ≈ 0.3 * sqrt(2)
    @test_throws ArgumentError observation_scatter(
        LaplaceScatterLikelihood(Xtr, ytr, collect(range(0.1, 0.3; length = 12))), mlin, [popt; 0.1])

    # no scalar scatter for a per-observation sigma or for correlated noise
    lhet = GaussianLikelihood(Xtr, ytr, collect(range(0.02, 0.08; length = 12)))
    @test_throws ArgumentError observation_scatter(lhet, mlin, popt)
    @test_throws ArgumentError prediction_intervals(lhet, mlin, Xnew, popt)
    lrar = XUniformDiagonalLikelihood(vec(Xtr), fill(0.1, 12), ytr, fill(0.2, 12))
    @test_throws ArgumentError observation_scatter(lrar, mlin, popt)

    # a grid with more rows than observations
    Xbig = reshape(collect(range(0.0, 4.0; length = 40)), :, 1)
    yb, lb, hb, sb = prediction_intervals(l, mlin, Xbig, popt)
    @test length(yb) == length(lb) == length(hb) == length(sb) == 40
    @test all(isfinite, sb)

    # fewer observations than parameters cannot support a t interval
    Xtiny = reshape([1.0, 2.0], :, 1)
    ltiny = GaussianLikelihood(Xtiny, [1.0, 2.0], 0.1)
    m3 = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2] * x[1] + p[3])), 3)
    @test_throws ArgumentError prediction_intervals(ltiny, m3, Xtiny, [1.0, 1.0, 1.0])
    @test_throws ArgumentError confidence_intervals(ltiny, m3, [1.0, 1.0, 1.0])
end

@testset "confidence_intervals of a linear model" begin
    # with a known sigma the covariance is exactly sigma² (A'A)⁻¹
    X = reshape(collect(range(0.5, 3.0; length = 12)), :, 1)
    y = 1.7 .* vec(X) .+ 0.4
    mlin = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    l = GaussianLikelihood(X, y, 0.05)
    popt = [1.7, 0.4]
    A = [X ones(12)]
    covref = 0.05^2 * inv(A' * A)
    seref = sqrt.(diag(covref))

    low, high, se, z, cov, corr = confidence_intervals(l, mlin, popt)
    @test cov ≈ covref
    @test se ≈ seref
    @test z ≈ abs.(popt ./ seref)
    @test corr ≈ covref ./ (seref * seref')
    @test diag(corr) ≈ ones(2)
    t = quantile(TDist(12 - 2), 1 - 0.01 / 2)
    @test low ≈ popt .- t .* seref
    @test high ≈ popt .+ t .* seref

    # a larger alpha gives a narrower interval
    low5, high5 = confidence_intervals(l, mlin, popt; alpha = 0.05)
    @test all(low .< low5) && all(high5 .< high)
end


@testset "pairwise confidence regions are marginal" begin
    # three strongly correlated parameters
    Xc = reshape(collect(range(0.5, 3.0; length = 20)), :, 1)
    yc = 1.7 .* vec(Xc) .+ 0.4
    mc = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1]^2 + p[2] * x[1] + p[3])), 3)
    lc = GaussianLikelihood(Xc, yc, 0.05)
    pc = [0.0, 1.7, 0.4]

    regs = pairwise_confidence_regions_laplace(lc, mc, pc; alpha = 0.95)
    @test length(regs) == 3                       # (1,2), (1,3), (2,3)
    @test all(r -> size(r, 2) == 2, regs)

    cov = SymRegLikelihoods.covariance(lc, mc, pc)
    n = n_observations(lc)
    d = length(pc)
    f = sqrt(d * quantile(FDist(d, n - d), 0.95))

    # the extent of the (i, j) region along i is the marginal half-width f * sqrt(cov[i, i])
    for (k, (i, j)) in enumerate([(1, 2), (1, 3), (2, 3)])
        @test maximum(regs[k][:, 1]) ≈ pc[i] + f * sqrt(cov[i, i]) rtol = 5e-3
        @test minimum(regs[k][:, 1]) ≈ pc[i] - f * sqrt(cov[i, i]) rtol = 5e-3
        @test maximum(regs[k][:, 2]) ≈ pc[j] + f * sqrt(cov[j, j]) rtol = 5e-3
    end
    # the parameters really are correlated, so this is not a vacuous check
    @test abs(cov[1, 3] / sqrt(cov[1, 1] * cov[3, 3])) > 0.5

    # two parameters
    m2 = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    p2 = [1.7, 0.4]
    r2 = pairwise_confidence_regions_laplace(lc, m2, p2; alpha = 0.95)
    @test length(r2) == 1
    cov2 = SymRegLikelihoods.covariance(lc, m2, p2)
    f2 = sqrt(2 * quantile(FDist(2, n - 2), 0.95))
    @test maximum(r2[1][:, 1]) ≈ p2[1] + f2 * sqrt(cov2[1, 1]) rtol = 5e-3

    @test_throws ArgumentError pairwise_confidence_regions_laplace(lc, m2, [1.7])
end

@testset "prepare handles a derivative that is not the last instruction" begin
    # `differentiate` hashconses, so the derivative can be an earlier node
    xs = collect(range(0.5, 3.0; length = 9))
    ys = 1.3 .* xs .+ 0.2
    l = XUniformDiagonalLikelihood(xs, fill(0.05, 9), ys, fill(0.05, 9))

    mz = Model(Float64, Expr(:->, :(x, p), :(p[1] + 0.0)), 1)
    _, _, dfidx_z = differentiate(code(mz), 1)
    @test dfidx_z < length(code(mz)) + 2      # not the last instruction

    pl = prepare(l, mz)
    @test pl.dfidx < length(pl.dcode)
    p = [1.1]
    nll = evaluate_nll(pl, p)
    @test isfinite(nll)
    grad = zeros(1)
    nllg = evaluate_nll_grad!(grad, pl, p)
    @test nllg ≈ nll
    @test grad ≈ ForwardDiff.gradient(q -> evaluate_nll(pl, q), p) rtol=1e-6

    # a model that does depend on x[1] but still hashconses its zero constant
    mz2 = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + 0.0)), 1)
    pl2 = prepare(l, mz2)
    p2 = [1.3]
    g2 = zeros(1)
    nll2 = evaluate_nll_grad!(g2, pl2, p2)
    @test nll2 ≈ evaluate_nll(pl2, p2)
    @test g2 ≈ ForwardDiff.gradient(q -> evaluate_nll(pl2, q), p2) rtol=1e-6
end

@testset "optimize a likelihood that is not Float64" begin
    core = :(p[1] * x[1] + p[2])
    X32 = reshape(Float32[1, 2, 3, 4], :, 1)
    y32 = Float32[1.3, 2.8, 4.2, 5.7]
    m32 = Model(Float32, Expr(:->, :(x, p), core), 2)
    m64 = Model(Float64, Expr(:->, :(x, p), core), 2)

    # reference: the same problem at full precision
    ref = optimize(NLoptOptimizer(), GaussianLikelihood(Float64.(X32), Float64.(y32), 0.4),
                   m64, [1.0, 0.0])

    l32 = GaussianLikelihood(X32, y32, 0.4f0)
    for opt in (NLoptOptimizer(), NLoptOptimizer(autodiff = true))
        r = optimize(opt, l32, m32, Float32[1.0, 0.0])
        @test all(isfinite, r.x)
        @test r.num_fevals > 0
        # `OptResult` stores Float64 whatever the likelihood's element type
        @test r.x isa Vector{Float64}
        # the search runs in Float64 but the objective is only evaluated to
        # Float32 accuracy, so the optimum agrees to about that precision
        @test r.x ≈ ref.x rtol = 1e-4
        @test r.f ≈ ref.f rtol = 1e-4
    end

    # the profiled and the non-least-squares likelihoods take the same path
    @test optimize(NLoptOptimizer(), GaussianProfiledLikelihood(X32, y32), m32,
                   Float32[1.0, 0.0, 1.0]).f |> isfinite
    @test optimize(NLoptOptimizer(), LaplaceLikelihood(X32, y32, 0.3f0), m32,
                   Float32[1.0, 0.0]).f |> isfinite

    # a Float64 starting point
    @test optimize(NLoptOptimizer(), l32, m32, [1.0, 0.0]).f ≈ ref.f rtol = 1e-4

    @test optimize(l32, m32; num_starts = 3, rng = MersenneTwister(7)).f ≈ ref.f rtol = 1e-4
    @test optimize(LaplaceLikelihood(X32, y32, 0.3f0), m32;
                   num_starts = 3, rng = MersenneTwister(7)).f |> isfinite
end


@testset "NLopt stopping criteria" begin
    core = :(p[1] * x[1] + p[2])
    X = reshape(Float64[1, 2, 3, 4], :, 1)
    y = Float64[1.3, 2.8, 4.2, 5.7]
    m64 = Model(Float64, Expr(:->, :(x, p), core), 2)
    m32 = Model(Float32, Expr(:->, :(x, p), core), 2)

    # the Laplace NLL has a kink at a zero residual
    llap = LaplaceLikelihood(X, y, 0.3)
    r = optimize(NLoptOptimizer(), llap, m64, [1.0, 0.0])
    @test r.returnvalue in (:FTOL_REACHED, :XTOL_REACHED)
    r_none = optimize(NLoptOptimizer(), llap, m64, [1.0, 0.0];
                      ftol_rel = 0.0, xtol_rel = 0.0)
    @test r_none.num_fevals > r.num_fevals          # the keywords do something
    @test r_none.f ≈ r.f rtol = 1e-6                # and stop at the same place

    # with a Float32 model the line search fails at Float32 resolution
    l32 = GaussianLikelihood(reshape(Float32.(vec(X)), :, 1), Float32.(y), 0.4f0)
    r32 = optimize(NLoptOptimizer(), l32, m32, Float32[1.0, 0.0])
    @test r32.returnvalue != :FAILURE
    @test r32.returnvalue == :ROUNDOFF_LIMITED
    ref = optimize(NLoptOptimizer(), GaussianLikelihood(X, y, 0.4), m64, [1.0, 0.0])
    @test r32.x ≈ ref.x rtol = 1e-4

    # the distinction that reinterpretation rests on: progress or no progress
    let f = SymRegLikelihoods._nlopt_returnvalue
        @test f(:FAILURE, 1.0, 2.0) == :ROUNDOFF_LIMITED   # improved on the start
        @test f(:FAILURE, 2.0, 2.0) == :FAILURE            # never moved
        @test f(:FAILURE, NaN, 2.0) == :FAILURE
        @test f(:FAILURE, 1.0, NaN) == :FAILURE
        @test f(:SUCCESS, 1.0, 2.0) == :SUCCESS            # every other code passes through
        @test f(:MAXEVAL_REACHED, 1.0, 2.0) == :MAXEVAL_REACHED
    end
end

@testset "evaluation limit of optimize (maxeval)" begin
    X = reshape(collect(Float64, 1:20), :, 1)
    y = 2.0 .* exp.(0.3 .* vec(X)) .+ 1.0 .+ 0.1 .* sin.(vec(X))
    m = Model(Float64, Expr(:->, :(x, p), :(p[1] * exp(p[2] * x[1]) + p[3])), 3)

    # Levenberg-Marquardt: residual evaluations, exact from 2 on (the first
    # trial step is always evaluated)
    lp = GaussianProfiledLikelihood(X, y)
    θ0 = [1.0, 0.1, 0.0, 1.0]
    full = optimize(lp, m, θ0)
    @test full.num_fevals > 5
    for maxeval in (2, 5)
        r = optimize(lp, m, θ0; maxeval)
        @test r.num_fevals == maxeval
        @test r.returnvalue == :MAXEVAL_REACHED
        @test r.f <= evaluate_nll(lp, m, θ0)          # the best point found is kept
    end

    # NLopt: evaluations of the objective and its gradient; LBFGS checks the
    # limit between line searches, so it is not exact
    ll = LaplaceProfiledLikelihood(X, y)
    θl = [1.0, 0.1, 0.0, 1.0]
    full = optimize(ll, m, θl)
    r = optimize(ll, m, θl; maxeval = 10)
    @test r.num_fevals < full.num_fevals
    @test r.returnvalue == :MAXEVAL_REACHED
end

# Each likelihood with intrinsic scatter against the same likelihood without
# it, with the scatter added to the measurement errors by hand.
@testset "intrinsic scatter" begin
    rng = MersenneTwister(77)
    X1 = reshape(collect(1.0:10.0), :, 1)
    y1 = 1.5 .* vec(X1) .- 0.7 .+ 0.3 .* randn(rng, 10)
    mlin = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    pm = [1.4, -0.5]
    sint = 0.25
    sy = 0.1 .+ 0.02 .* collect(1.0:10.0)

    csv = joinpath(@__DIR__, "RAR.csv")
    m, header = readdlm(csv, ',', Float64; header=true)
    header = vec(header)
    gbar = m[:, findfirst(==("gbar"), header)]
    gobs = m[:, findfirst(==("gobs"), header)]
    elgbar = m[:, findfirst(==("e_gbar"), header)] ./ (gbar * log(10))
    elgobs = m[:, findfirst(==("e_gobs"), header)] ./ (gobs * log(10))
    mrar = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p),
        :(p[1] * (abs(p[2] + x[1])^p[3] + x[1]))), 3))
    prar = [0.8395593155955371, -0.022204122733443278, 0.38093147640767777]

    Z = log10.(gbar)
    msmnr = to_logspace_xy_model(Model(Float64, Expr(:->, :(x, p),
        :(p[1] * abs(x[1])^(-1 / (p[2] + abs(p[3] + x[1])^p[4])))), 4))
    psmnr = [-1.7302245173940842, -2.40586258570395, -0.017892186956077034,
             0.051284150873232376, 0.07915900803213895^2, -0.5024228473024372,
             0.7397713386530679^2]

    n, d = 10, 2
    nd = n * d
    Xm = [randn(rng, n) .* 2  randn(rng, n)]
    ym = (Xm[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    Σxx = Diagonal(rand(rng, nd) .+ 0.5)
    Σxy = zeros(nd, n)
    for i in 1:n, k in 1:d
        Σxy[(k - 1) * n + i, i] = 0.2 * (rand(rng) - 0.5)
    end
    Σyy = SymTridiagonal(rand(rng, n) .* 3 .+ 1.0, rand(rng, n - 1) .* 0.4)
    mmv = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
    pmv = [1.0, -1.0, 0.4^2, 0.3, -0.2, 1.1^2, 0.8^2]
    pmv_off = pmv[[1, 2, 4, 5, 6, 7]]     # the same point without `sigma²`

    Σxxp = Matrix(0.3I, n, n) + 0.01*randn(rng, n, n); Σxxp = Σxxp * Σxxp'
    Σyyp = Matrix(0.6I, n, n) + 0.01*randn(rng, n, n); Σyyp = Σyyp * Σyyp'
    Σxyp = 0.01*randn(rng, n, n)
    Xp = reshape(rand(rng, n) .+ 1, :, 1)
    yp = (Xp[:, 1] .^ 2 .- 1) .+ 0.4 .* randn(rng, n)
    mprof = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2])), 2)

    xcc = rand(rng, n) .+ 2
    agecc = sqrt.(1.0 .+ 0.2 .* xcc) .+ 0.05 .* randn(rng, n)
    sigcc = 0.1 .* ones(n)
    mcc = Model(Float64, Expr(:->, :(x, p), :(p[1] + p[2]*x[1])), 2)
    pcc = [1.0, 0.2]

    # (tag, on, off-equivalent, model, p for `on`, p for `off`, nll offset)
    equivalents = [
        ("Gaussian scalar",
         GaussianScatterLikelihood(X1, y1, 0.2),
         GaussianLikelihood(X1, y1, sqrt(0.2^2 + sint^2)), mlin, [pm; sint^2], pm, 0.0),
        ("Gaussian per-point",
         GaussianScatterLikelihood(X1, y1, sy),
         GaussianLikelihood(X1, y1, sqrt.(sy .^ 2 .+ sint^2)), mlin, [pm; sint^2], pm, 0.0),
        ("Laplace scalar",
         LaplaceScatterLikelihood(X1, y1, 0.2),
         LaplaceLikelihood(X1, y1, sqrt(0.2^2 + sint^2)), mlin, [pm; sint^2], pm, 0.0),
        ("Laplace per-point",
         LaplaceScatterLikelihood(X1, y1, sy),
         LaplaceLikelihood(X1, y1, sqrt.(sy .^ 2 .+ sint^2)), mlin, [pm; sint^2], pm, 0.0),
        ("XUniformDiagonal",
         XUniformDiagonalLikelihood(log10.(gbar), elgbar, log10.(gobs), elgobs; intrinsic_scatter = true),
         XUniformDiagonalLikelihood(log10.(gbar), elgbar, log10.(gobs), sqrt.(elgobs .^ 2 .+ 0.05^2)),
         mrar, [prar; 0.05^2], prar, 0.0),
        # the scatter-free CC NLL omits the normalization, so the folded
        # version differs by the part of it that sigma_int changes
        ("Cosmic chronometer",
         CosmicChronometerLikelihood(xcc, agecc, sigcc; intrinsic_scatter = true),
         CosmicChronometerLikelihood(xcc, agecc, sqrt.(sigcc .^ 2 .+ sint^2)),
         mcc, [pcc; sint^2], pcc, sum(log.((sigcc .^ 2 .+ sint^2) ./ sigcc .^ 2)) / 2),
        ("MNR structured",
         MNRBandedLikelihood(Xm, ym, Σxx, Σxy, Σyy),
         MNRBandedLikelihood(Xm, ym, Σxx, Σxy, Σyy + 0.4^2 * I; intrinsic_scatter = false),
         mmv, pmv, pmv_off, 0.0),
        ("MNR dense",
         MNRDenseLikelihood(Xm, ym, Matrix(Σxx), Σxy, Matrix(Σyy)),
         MNRDenseLikelihood(Xm, ym, Matrix(Σxx), Σxy, Matrix(Σyy) + 0.4^2 * I;
                            intrinsic_scatter = false),
         mmv, pmv, pmv_off, 0.0),
        ("XProfileDense",
         XProfileDenseLikelihood(Xp, yp, Σxxp, Σxyp, Σyyp; intrinsic_scatter = true),
         XProfileDenseLikelihood(Xp, yp, Σxxp, Σxyp, Σyyp + sint^2 * I),
         mprof, [1.0, -1.0, sint^2], [1.0, -1.0], 0.0),
    ]
    for (tag, lon, loff, model, pon, poff, offset) in equivalents
        @testset "$tag" begin
            @test has_intrinsic_scatter(lon)
            @test !has_intrinsic_scatter(loff)
            @test n_param(lon, model) == length(pon)
            @test n_param(loff, model) == length(poff)
            @test 1 in positive_params(lon)
            @test evaluate_nll(lon, model, pon) ≈ evaluate_nll(loff, model, poff) + offset rtol = 1e-10
            @test intrinsic_scatter(lon, model, pon) ≈ sqrt(pon[n_param(model) + 1])
            @test intrinsic_scatter(loff, model, poff) == 0

            # a negative variance is the worst loss
            pneg = copy(pon); pneg[n_param(model) + 1] = -0.01
            @test evaluate_nll(lon, model, pneg) == floatmax(Float64)

            # analytic gradients, prepared and unprepared, against ForwardDiff
            for l in (lon, loff), p in (l === lon ? pon : poff,)
                ga = zeros(length(p)); gp = zeros(length(p)); gf = zeros(length(p))
                nll_a = evaluate_nll_grad!(ga, l, model, p)
                evaluate_nll_grad!(gp, prepare(l, model), p)
                nll_f = evaluate_nll_grad_fd!(gf, l, model, p)
                @test nll_a ≈ nll_f rtol = 1e-10
                @test ga ≈ gf rtol = 1e-6 atol = 1e-10
                @test gp ≈ ga rtol = 1e-10 atol = 1e-12
                @test evaluate_nll(prepare(l, model), p) ≈ nll_a rtol = 1e-10
            end

            H = information_matrix(lon, model, pon)
            @test size(H) == (length(pon), length(pon))
            @test all(isfinite, H)
        end
    end

    @testset "MNR without intrinsic scatter" begin
        # structured == dense with the scatter switched off
        ls = MNRBandedLikelihood(Xm, ym, Σxx, Σxy, Σyy; intrinsic_scatter = false)
        ld = MNRDenseLikelihood(Xm, ym, Matrix(Σxx), Σxy, Matrix(Σyy); intrinsic_scatter = false)
        @test n_likelihood_params(ls) == 2d
        @test collect(positive_params(ls)) == [d + 1, d + 2]
        @test evaluate_nll(ls, mmv, pmv_off) ≈ evaluate_nll(ld, mmv, pmv_off) rtol = 1e-10

        # simple MNR: a vanishing scatter is the same as none
        lson = MNRDiagonalLikelihood(Z, elgbar, log10.(gobs), elgobs)
        lsoff = MNRDiagonalLikelihood(Z, elgbar, log10.(gobs), elgobs; intrinsic_scatter = false)
        @test n_likelihood_params(lson) == 3
        @test n_likelihood_params(lsoff) == 2
        ptiny = copy(psmnr); ptiny[5] = 1e-35
        poff = psmnr[[1, 2, 3, 4, 6, 7]]
        @test evaluate_nll(lsoff, msmnr, poff) ≈ evaluate_nll(lson, msmnr, ptiny) rtol = 1e-12
        ga = zeros(6); gf = zeros(6)
        @test evaluate_nll_grad!(ga, lsoff, msmnr, poff) ≈ evaluate_nll_grad_fd!(gf, lsoff, msmnr, poff)
        @test ga ≈ gf rtol = 1e-6 atol = 1e-10
    end

    @testset "Gaussian and Laplace configurations" begin
        @test has_intrinsic_scatter(GaussianProfiledLikelihood(X1, y1))
        @test has_intrinsic_scatter(LaplaceProfiledLikelihood(X1, y1))
        @test !has_intrinsic_scatter(GaussianLikelihood(X1, y1, 0.2))
        @test !has_intrinsic_scatter(LaplaceLikelihood(X1, y1, 0.2))

        # zero measurement error plus a scatter is the profiled likelihood, at
        # any value of the variance
        for (lp, le) in ((GaussianProfiledLikelihood(X1, y1), GaussianScatterLikelihood(X1, y1, 0.0)),
                         (LaplaceProfiledLikelihood(X1, y1), LaplaceScatterLikelihood(X1, y1, 0.0)))
            for v in (0.05, 0.3)
                @test evaluate_nll(le, mlin, [pm; v]) ≈ evaluate_nll(lp, mlin, [pm; v]) rtol = 1e-12
            end
            θ = natural_parameters(lp, mlin, pm)
            @test intrinsic_scatter(lp, mlin, θ) ≈ sqrt(θ[3])
        end

        # a constant measurement error plus scatter has the optimum
        # σ_y² + σ_int² = RSS/n, but is fitted jointly, not by least squares
        lg = GaussianScatterLikelihood(X1, y1, 0.2)
        @test !(lg isa AbstractLeastSquaresLikelihood)
        @test_throws MethodError optimize(LevenbergMarquardtOptimizer(), lg, mlin, [1.0, 0.0, 0.1])
        @test_throws MethodError residual!(zeros(10), lg, mlin, [1.0, 0.0])
        r = optimize(lg, mlin; num_starts = 3, rng = MersenneTwister(2))
        rss = sum(abs2, y1 .- (r.x[1] .* vec(X1) .+ r.x[2]))
        @test r.x[3] ≈ rss / 10 - 0.2^2 rtol = 1e-4
        # and the model parameters are the ordinary least-squares fit
        rls = optimize(GaussianLikelihood(X1, y1, 0.2), mlin, [1.0, 0.0])
        @test r.x[1:2] ≈ rls.x rtol = 1e-4

        # a new observation scatters by the measurement error and the
        # intrinsic scatter in quadrature
        @test observation_scatter(lg, mlin, [pm; sint^2]) ≈ sqrt(0.2^2 + sint^2)
        _, lo, hi, _ = prediction_intervals(lg, mlin, X1[1:3, :], r.x)
        @test all(lo .< hi)
    end
end

@testset "every public name has a docstring" begin
    @test isempty(Docs.undocumented_names(SymRegLikelihoods))
end

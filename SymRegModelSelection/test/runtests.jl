using Test
using SymRegModelSelection
using SymRegInterpreter
using SymRegLikelihoods
using SymRegInterpreter: Instruction, ADD, MUL, POWCONST, code, core_expr,
    create_const_instruction, create_var_instruction, create_scaledvar_instruction,
    replace_param_with_const
using SymRegLikelihoods: n_opt_params, natural_parameters
using SymRegModelSelection: description_length_terms, description_length_complexity_penalty,
    fractional_bayes_factor_terms, fractional_bayes_factor_complexity_penalty,
    BIC_funccompl_penalty, param_complexity_det, param_complexity_rot,
    param_complexity_rot_scaled, find_best_snap, find_best_snap_for_pidx, rational_approx
using LinearAlgebra
using Logging
@testset "func_complexity (Expr path)" begin
    # parameters are not counted: nodes {*, +, x1, x2}, symbols {*, +, x1, x2}
    @test func_complexity(:(x[1] * p[1] + x[2])) == 4 * log(4)
    # parameters counted only with count_params
    @test func_complexity(:(x[1] * p[1] + x[2]); count_params = true) == 5 * log(5)
    # integer constants count as log(|c| + 1)
    @test func_complexity(:(3 * x[1] + p[1])) == 3 * log(3) + log(4)
    # sqrt counts as a single symbol (no ^ + 1//2 decomposition)
    @test func_complexity(:(sqrt(x[1]))) == 2 * log(2)
    # negative integer literal contributes the :- symbol and |c| as constant
    @test func_complexity(:(-2 + x[1])) == 3 * log(3) + log(3)
    # float constants are not allowed, not even integer-valued ones
    @test_throws ArgumentError func_complexity(:(2.5 * x[1] + p[1]))
    @test_throws ArgumentError func_complexity(:(2.0 * x[1] + p[1]))
    # single symbol: N * log(N) = 1 * log(1) = 0
    @test func_complexity(:(x[1])) == 0.0
end

@testset "func_complexity (Model)" begin
    m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    # nodes/symbols: *, +, x1 (p1, p2 are parameters, not counted)
    @test func_complexity(m) == 3 * log(3)
    @test func_complexity(m) == func_complexity(core_expr(m))
end

@testset "func_complexity (CodeModel dispatch)" begin
    # a code-backed model computes complexity from its code (DAG path) without
    # ever constructing an expression
    codevec = [create_var_instruction(Float64, 1), create_var_instruction(Float64, 2),
               Instruction{Float64}(MUL, UInt32(1), UInt32(2), 0.0)]
    cm = Model{Float64}(codevec, 0)
    @test cm isa CodeModel
    @test func_complexity(cm) == func_complexity(code(cm)) == 3 * log(3)
end

@testset "func_complexity of rational constants" begin
    # n/d costs log(|n| + 1) + log(d) plus one `/` symbol, a negative one also
    # a `-` symbol, the same way as integer constants
    rat(c) = Expr(:call, :+, Expr(:call, :*, c, :(x[1])), :(x[2]))   # c * x1 + x2
    @test func_complexity(rat(1//2)) ≈ 5 * log(5) + log(2) + log(2)
    @test func_complexity(rat(-1//2)) ≈ 6 * log(6) + log(2) + log(2)
    @test func_complexity(rat(355//113)) ≈ 5 * log(5) + log(356) + log(113)
    # an integer rational is an integer constant, without `/`
    @test func_complexity(rat(2//1)) == func_complexity(rat(2)) == 4 * log(4) + log(3)
    @test func_complexity(rat(-2//1)) == func_complexity(rat(-2))
    # a division of two integer literals is the same rational
    @test func_complexity(rat(:(1 / 2))) ≈ func_complexity(rat(1//2))
    @test func_complexity(rat(:(-1 / 2))) ≈ func_complexity(rat(-1//2))
    # but a division by a non-constant is not
    @test func_complexity(:(x[1] / 2)) ≈ 2 * log(2) + log(3)

    # the bytecode path agrees, both for a model built from the expression and
    # for one with a parameter snapped to the rational
    m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + x[2])), 1)
    for c in (1//2, -1//2, 355//113, 2//1, -3//1)
        @test func_complexity(Model(Float64, Expr(:->, :(x, p), rat(c)), 0)) ≈ func_complexity(rat(c))
        @test func_complexity(replace_param_with_const(m, 1, c)) ≈ func_complexity(rat(c))
    end
    # non-integer floats are not allowed in the bytecode path
    @test_throws ArgumentError func_complexity(replace_param_with_const(m, 1, 0.5))
    @test_throws ArgumentError func_complexity([create_const_instruction(2.5)])
end

@testset "func_complexity (bytecode path): parameters and scaled variables" begin
    # count_params counts parameters as one symbol in both paths
    m = Model(Float64, Expr(:->, :(x, p), :(x[1] * p[1] + p[2] * x[2])), 2)
    cm = CodeModel(code(m), 2)
    for count_params in (false, true)
        @test func_complexity(cm; count_params) ≈ func_complexity(core_expr(m); count_params)
    end
    @test func_complexity(cm; count_params = true) ≈ 7 * log(5)

    # a scaled variable p * x[i] is one node, distinct from x[i] and from a
    # scaled variable of another input
    sv(p, v) = create_scaledvar_instruction(Float64, p, v)
    add(a, b) = Instruction{Float64}(ADD, UInt32(a), UInt32(b), 0.0)
    @test func_complexity([sv(1, 1), sv(2, 1), add(1, 2)]) ≈ 3 * log(2)
    @test func_complexity([sv(1, 1), create_var_instruction(Float64, 1), add(1, 2)]) ≈ 3 * log(3)
    @test func_complexity([sv(1, 1), sv(2, 2), add(1, 2)]) ≈ 3 * log(3)
end

@testset "func_complexity (bytecode path)" begin
    # no repeated subexpressions: Expr and bytecode counts agree
    m = Model(Float64, Expr(:->, :(x, p), :(x[1] + x[2] / x[3])), 0)
    @test func_complexity(core_expr(m)) == 5 * log(5)
    @test func_complexity(code(m)) == 5 * log(5)
    # constants contribute log(|c| + 1) in both paths
    m2 = Model(Float64, Expr(:->, :(x, p), :(3 * x[1] + x[2])), 0)
    @test func_complexity(core_expr(m2)) == 4 * log(4) + log(4)
    @test func_complexity(code(m2)) == 4 * log(4) + log(4)
    # negative constant: :- symbol plus |c| as constant (both paths)
    m3 = Model(Float64, Expr(:->, :(x, p), :(-2 + x[1])), 0)
    @test func_complexity(core_expr(m3)) ≈ 3 * log(3) + log(3)
    @test func_complexity(code(m3)) ≈ 3 * log(3) + log(3)
    # the integer exponent of a power is a constant in both paths, a negative
    # one with a `-` symbol
    for (body, ref) in ((:(x[1]^2 + x[2]), 4 * log(4) + log(3)),
                        (:(x[1]^-2 + x[2]), 5 * log(5) + log(3)))
        mp = Model(Float64, Expr(:->, :(x, p), body), 0)
        @test any(instr -> instr.opcode == POWCONST, code(mp))
        @test func_complexity(core_expr(mp)) ≈ ref
        @test func_complexity(code(mp)) ≈ ref
    end
    # a non-integer exponent is not allowed
    @test_throws ArgumentError func_complexity([create_var_instruction(Float64, 1),
        Instruction{Float64}(POWCONST, UInt32(1), UInt32(0), 0.5)])
    # program that only returns a constant: constant complexity only
    @test func_complexity([create_const_instruction(2.0)]) == log(3.0)
    # empty program has zero complexity
    @test func_complexity(Instruction{Float64}[]) == 0.0
end

# ---------------------------------------------------------------------------
# Fixtures: y = b1 * x + b0 with Gaussian noise, sigma known / profiled
# ---------------------------------------------------------------------------
X = reshape([1.0, 1.0, 2.0, 2.0], 4, 1)
y = [2.2, 1.8, 4.9, 5.1]
m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)

# least-squares optimum (needed for the profiled-sigma properties)
Xtilde = [X[:, 1] ones(4)]
p_ls = Xtilde \ y

sigma = 0.5
l_fixed = GaussianLikelihood(X, y, sigma)
l_prof = GaussianProfiledLikelihood(X, y) # profiled sigma2
θ_ls = natural_parameters(l_prof, m, p_ls)  # [p_ls; sigma2 at the profile optimum]

@testset "parameter counts" begin
    @test n_param(l_fixed, m) == 2
    @test n_param(l_prof, m) == 3 # profiled sigma2 is a parameter
    @test length(θ_ls) == 3
end

@testset "information_matrix (fixed sigma)" begin
    FI = information_matrix(l_fixed, m, p_ls)
    # analytic: FI = J' * J / sigma^2
    ypred, J = interpret_jac(m, X, p_ls)
    @test FI ≈ (J' * J) ./ sigma^2
    @test FI ≈ FI' # symmetric
end

@testset "information_matrix (profiled sigma)" begin
    FI = information_matrix(l_prof, m, θ_ls)
    @test size(FI) == (3, 3)
    n = length(y)
    ypred = interpret_vec(m, X, p_ls)
    s2 = sum(abs2, ypred .- y) / n
    @test θ_ls ≈ vcat(p_ls, s2)
    # at the least-squares optimum the residuals are orthogonal to the model
    # Jacobian, so the theta-sigma2 cross block vanishes exactly ...
    @test FI[1:2, 3] ≈ zeros(2) atol = 1e-8
    @test FI[3, 1:2] ≈ zeros(2) atol = 1e-8
    # ... and the sigma2 diagonal is n / (2 * s2^2)
    @test FI[3, 3] ≈ n / (2 * s2^2) rtol = 1e-8
end

@testset "param_complexity (pure vector math)" begin
    # unclamped identity: max(0, log|p| + log(prec)/2 - log(12)/2) + log(2)
    @test param_complexity([100.0], [1000.0]) ≈
          log(100) + 0.5 * log(1000) - 0.5 * log(12) + log(2)
    # negative contributions of uncertain parameters are clamped (sign remains)
    @test param_complexity([0.001], [1e-6]) == log(2)
    @test param_complexity(Float64[], Float64[]) == 0.0
end

@testset "param_complexity unrotated vs rotated" begin
    # orthogonal design -> diagonal information matrix -> rot == unrot
    X2 = [1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
    y2 = [2.1, 1.9, 5.2, 4.8]
    m2 = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2] * x[2])), 2)
    l2 = GaussianLikelihood(X2, y2, sigma)
    p2 = [3.0, -2.0]
    prec = 2 ./ sigma^2
    expected = sum(max(0.0, log(abs(p2[i]) / sqrt(12 / prec))) + log(2) for i in 1:2)
    @test param_complexity(p2, l2, m2) ≈ expected
    @test param_complexity_rot(p2, l2, m2) ≈ expected
    @test param_complexity(p2, l2, m2) ≈ param_complexity_rot(p2, l2, m2)
    # uncorrelated parameters: the scaled rotation is the unrotated term too
    @test param_complexity_rot_scaled(p2, l2, m2) ≈ expected
    @test param_complexity_rot(p2, l2, m2; scaled = true) == param_complexity_rot_scaled(p2, l2, m2)
end

@testset "scaled rotation: units of y, ridges of optima, the unrotated bound" begin
    # The scaled rotation finds stiff and sloppy directions in relative
    # coordinates z = θ/|θ̂|, i.e. in significant digits.  The unscaled one finds
    # them in the parameters' own units, so a change of units, or of where an
    # optimizer lands on a ridge of equivalent optima, rotates it.  Deterministic
    # "noise" keeps the fixture free of an RNG.
    x = collect(range(0.1, 4.0; length = 60))
    X = reshape(x, :, 1)
    y = 2.5 .* exp.(-0.7 .* x) .+ 0.3 .* x .+ 0.2 .* sin.(37 .* x)
    # the profiled sigma² of the start is not read
    fit(l, m, x0) = optimize(LevenbergMarquardtOptimizer(), l, m, [x0; ones(n_likelihood_params(l))]).x
    body_exp = :(p[1] * exp(-p[2] * x[1]) + p[3] * x[1])
    mexp = Model(Float64, Expr(:->, :(x, p), body_exp), 3)
    pexp = fit(GaussianProfiledLikelihood(X, y), mexp, [2.5, 0.7, 0.3])

    # A change of the units of y multiplies the linear coefficients and σ and
    # leaves the rate alone.  NLL shifts by the same n·log c for every model,
    # so the ranking is unit-independent only if the parameter complexity is.
    ref = param_complexity_rot_scaled(pexp, GaussianProfiledLikelihood(X, y), mexp)
    for c in (1e-2, 1e2)
        pc = [c * pexp[1], pexp[2], c * pexp[3], c^2 * pexp[4]]
        @test param_complexity_rot_scaled(pc, GaussianProfiledLikelihood(X, c .* y), mexp) ≈ ref rtol = 1e-8
        # with measurement errors and an intrinsic scatter too
        l1 = GaussianScatterLikelihood(X, y, fill(0.15, 60))
        lc = GaussianScatterLikelihood(X, c .* y, fill(0.15c, 60))
        @test param_complexity_rot_scaled([pc[1:3]; (0.2c)^2], lc, mexp) ≈
              param_complexity_rot_scaled([pexp[1:3]; 0.2^2], l1, mexp) rtol = 1e-8
    end

    # Principle 8: an overparameterized model's optimum is a ridge, and the
    # point on it is an accident of the optimizer.  On a product-type ridge
    # (two merged constants) the fitted function, and the charge, stay put.
    mprod = Model(Float64, Expr(:->, :(x, p), :(p[1] * p[2] * exp(-p[3] * x[1]) + p[4] * x[1])), 4)
    lprof = GaussianProfiledLikelihood(X, y)
    pprod = fit(lprof, mprod, [1.25, 2.0, 0.7, 0.3])
    ridge(t) = [pprod[1] * t, pprod[2] / t, pprod[3], pprod[4], pprod[5]]
    for t in (0.1, 10.0)
        @test evaluate_nll(lprof, mprod, ridge(t)) ≈ evaluate_nll(lprof, mprod, pprod)
        @test param_complexity_rot_scaled(ridge(t), lprof, mprod) ≈
              param_complexity_rot_scaled(pprod, lprof, mprod) rtol = 1e-8
    end

    # Rotation should never cost more than the unrotated charge.  Before the
    # clamp that is a theorem for the scaled rotation at an optimum (Hadamard
    # and AM-GM); at these fitted optima it holds after the clamp as well.
    for (m, x0) in ((mexp, [2.5, 0.7, 0.3]), (mprod, [1.25, 2.0, 0.7, 0.3]),
                    (Model(Float64, Expr(:->, :(x, p), :(p[1] * exp(-p[2] * x[1] + p[3]) + p[4] * x[1])), 4),
                     [2.0, 0.7, 0.2, 0.3]),
                    (Model(Float64, Expr(:->, :(x, p), :(p[1] / (1 + p[2] * x[1]) + p[3] * x[1])), 3),
                     [2.5, 1.0, 0.3]))
        p = fit(lprof, m, x0)
        @test param_complexity_rot_scaled(p, lprof, m) <= param_complexity(p, lprof, m) + 1e-9
    end
end

@testset "param_complexity_det sums its two terms" begin
    # Eq. 6 of arXiv:2304.06333 replaces the `Σ_α ½ log(I_αα) - (p/2) log(3)`
    # group of Eq. 5 by `½ log det(I) + (p/2) log(v_p)`, two separate summands
    # with `log(v_p) = 1 - log(3)`.  They used to be grouped as
    # `(log det I + p)/2 * log(v_p)`, which scales the log-determinant by
    # `log(v_p) ≈ -0.0986` as well and turns the complexity negative.
    pc = param_complexity(p_ls, l_fixed, m)
    pcd = param_complexity_det(p_ls, l_fixed, m)
    @test pcd > 0
    @test 0.5 < pcd / pc < 2      # the three estimators stay in the same range

    # the two summands, formed here without going through the implementation
    sv = svdvals(information_matrix(l_fixed, m, p_ls))
    @test length(sv) == 2                              # full rank, so r == p
    @test pcd ≈ sum(log, sv) / 2 + length(sv) / 2 * (1 - log(3))
    # and the grouping that was there before is a different number
    @test !isapprox(pcd, (sum(log, sv) + length(sv)) / 2 * (1 - log(3)))
end

@testset "param_complexity_det reports a reduced rank through @debug" begin
    # `p[1]` and `p[2]` multiply the same column, so the information matrix is
    # exactly rank one and the rank-reduction branch is taken.
    md = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2] * x[1])), 2)
    ld = GaussianLikelihood(X, y, sigma)
    pd = [0.5, 0.5]

    out = tempname()
    v = open(out, "w") do io
        redirect_stdout(() -> param_complexity_det(pd, ld, md), io)
    end
    @test isfinite(v)
    # a model search calls this once per candidate, so the branch must not
    # write to stdout ...
    @test isempty(read(out, String))
    rm(out; force = true)
    # ... but the information is still there for whoever asks for it
    @test_logs (:debug, "Fisher information matrix is rank deficient") min_level =
        Logging.Debug match_mode = :any param_complexity_det(pd, ld, md)
end

@testset "negative eigenvalues of the information matrix count as zero precision" begin
    # far from the optimum the Hessian of the NLL is indefinite; a direction of
    # negative curvature costs only its sign (log 2)
    Xn = reshape(collect(range(0.0, 3.0; length = 30)), :, 1)
    yn = 2 .* exp.(0.5 .* Xn[:, 1])
    mn = Model(Float64, Expr(:->, :(x, p), :(p[1] * sin(p[2] * x[1]) + p[3])), 3)
    ln = GaussianLikelihood(Xn, yn, 0.1)
    pn = [1.0, 2.0, 1.0]
    FI = information_matrix(ln, mn, pn)
    e = eigen(Symmetric(FI))
    @test count(<(0), e.values) == 1
    λ = max.(e.values, 0.0)

    # rotated: the negative direction contributes exactly log(2)
    pproj = e.vectors' * pn
    terms = [log(2) + max(0.0, log(abs(pproj[i])) + log(λ[i]) / 2 - log(3) / 2 - log(2))
             for i in eachindex(λ)]
    @test terms[findfirst(<(0), e.values)] == log(2)
    @test param_complexity_rot(pn, ln, mn) ≈ sum(terms)
    # charging |λ| as precision, as before, costs more
    @test param_complexity_rot(pn, ln, mn) <
          sum(log(2) + max(0.0, log(abs(pproj[i])) + log(abs(e.values[i])) / 2 - log(3) / 2 - log(2))
              for i in eachindex(λ))
    # determinant: the negative direction is a rank deficiency
    @test param_complexity_det(pn, ln, mn) ≈ sum(log, filter(>(0), λ)) / 2 + 2 / 2 * (1 - log(3))
    # unrotated: the diagonal of the information matrix without the negative part
    @test param_complexity(pn, ln, mn) ≈
          SymRegModelSelection.param_complexity(pn, diag(e.vectors * Diagonal(λ) * e.vectors'))
    # scaled rotation is finite as well
    @test isfinite(param_complexity_rot_scaled(pn, ln, mn))

    # a positive definite information matrix is unaffected
    @test param_complexity(p_ls, l_fixed, m) ≈
          SymRegModelSelection.param_complexity(p_ls, diag(information_matrix(l_fixed, m, p_ls)))
end

@testset "symmetric eigensolver (Jacobi)" begin
    eig = SymRegModelSelection._symmetric_eigen

    # agrees with LAPACK on well-conditioned matrices, in both precisions
    for T in (Float64, Float32), n in (1, 2, 5, 9)
        B = T[sin(3 * i + 7 * j) for i in 1:n, j in 1:n]
        A = B + B'
        λ, V = eig(A)
        @test eltype(λ) == T && eltype(V) == T
        @test issorted(λ)
        @test λ ≈ eigvals(Symmetric(A)) atol = 100 * eps(T) * opnorm(A)
        @test V' * V ≈ I atol = 100 * eps(T)
        @test A * V ≈ V * Diagonal(λ) atol = 100 * eps(T) * opnorm(A)
    end

    # repeated eigenvalues, zero and diagonal matrices
    @test eig(Matrix(1.0I, 3, 3)) == (ones(3), Matrix(1.0I, 3, 3))
    @test eig(zeros(2, 2)) == (zeros(2), Matrix(1.0I, 2, 2))
    @test eig([3.0 0.0; 0.0 1.0]) == ([1.0, 3.0], [0.0 1.0; 1.0 0.0])

    # only the upper triangle is read, as for `Symmetric(A, :U)`
    @test first(eig([2.0 1.0; NaN 2.0])) ≈ [1.0, 3.0]

    # A graded Float32 information matrix from a NeoGP run, for which LAPACK's
    # `ssyevr` never returned (overflow in slarre, NaN loop in slarrb) and
    # `ssyevd` returned 1.4e7 for the middle eigenvalue 20.245588.
    A32 = Float32[ 3.236343f-18   0.05547311   -1.051364f-13
                   0.05547311     9.508415f14  -1803.2341
                  -1.051364f-13  -1803.2341      20.245588 ]
    λ, V = eig(A32)
    @test λ[2] ≈ 20.245588f0 rtol = 4 * eps(Float32)
    @test λ[3] ≈ 9.508415f14 rtol = 4 * eps(Float32)
    @test abs(λ[1]) < 1.0f-20
    @test opnorm(Float64.(A32) * V - V * Diagonal(λ)) <= 10 * eps(Float32) * opnorm(Float64.(A32))
    @test V' * V ≈ I atol = 10 * eps(Float32)

    # High relative accuracy on a graded positive definite D*C*D: every
    # eigenvalue, also the smallest, to a few ulps of a BigFloat reference.
    C = [2.0 0.5 0.1; 0.5 1.5 -0.3; 0.1 -0.3 1.0]
    d = [1.0e-6, 1.0, 1.0e6]
    for T in (Float32, Float64)
        A = T.(d .* C .* d')
        ref = setprecision(256) do
            Float64.(first(eig(BigFloat.(A))))
        end
        @test maximum(abs.(Float64.(first(eig(A))) .- ref) ./ abs.(ref)) < 20 * eps(T)
    end

    # non-finite entries end after a bounded number of sweeps
    @test length(first(eig([1.0 NaN; NaN 1.0]))) == 2
end

@testset "Laplace parameters cost more than their sign" begin
    # the observed Hessian of the Laplace NLL is zero in the model parameters,
    # which left only log(2) per parameter; the expected Fisher information
    # charges about as much as the Gaussian likelihood does on the same data
    Xl = reshape(collect(range(0.5, 3.0; length = 12)), :, 1)
    yl = 1.7 .* vec(Xl) .+ 0.4 .+ 0.1 .* sin.(1:12)
    ml = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    lg = GaussianProfiledLikelihood(Xl, yl)
    ll = LaplaceProfiledLikelihood(Xl, yl)
    pg = optimize(lg, ml, natural_parameters(lg, ml, [1.7, 0.4])).x
    pl = optimize(ll, ml, natural_parameters(ll, ml, [1.7, 0.4])).x
    pcl = param_complexity_rot(pl, ll, ml)
    pcg = param_complexity_rot(pg, lg, ml)
    @test pcl > 3 * log(2) + 2
    @test abs(pcl - pcg) < 1
end

@testset "description_length" begin
    nll = evaluate_nll(l_fixed, m, p_ls)
    fc = func_complexity(m)
    pc = param_complexity_rot(p_ls, l_fixed, m)
    dl, nll2, fc2, pc2 = description_length(l_fixed, m, p_ls)
    @test dl ≈ nll + fc + pc
    @test nll2 == nll
    @test fc2 == fc
    @test pc2 ≈ pc
    @test description_length_terms(l_fixed, m, p_ls)[1] == nll
    @test description_length_terms(l_fixed, m, p_ls)[2] == fc
    @test description_length_terms(l_fixed, m, p_ls)[3] ≈ pc
    @test description_length_complexity_penalty(l_fixed, m, p_ls) ≈ fc + pc
    # every entry point takes the same choice of parameter complexity
    for f in (param_complexity, param_complexity_rot, param_complexity_rot_scaled, param_complexity_det)
        pcf = f(p_ls, l_fixed, m)
        @test description_length(l_fixed, m, p_ls; paramCompFunc = f)[4] == pcf
        @test description_length_terms(l_fixed, m, p_ls; paramCompFunc = f)[3] == pcf
        @test description_length_complexity_penalty(l_fixed, m, p_ls; paramCompFunc = f) == fc + pcf
    end
    # failing parameter complexity -> (Inf, nll, funcComp, NaN)
    dl3, nll3, fc3, pc3 = description_length(l_fixed, m, p_ls;
        paramCompFunc = (p, l, mm) -> error("boom"))
    @test dl3 == Inf
    @test pc3 === NaN
    @test nll3 == nll && fc3 == fc

    # that failure is an ordinary outcome for one candidate in a search, so it
    # is reported at debug level.  It used to be an `@error`, which in a model
    # search is one log line per candidate for something the `NaN` already
    # says.
    out = tempname()
    open(out, "w") do io
        redirect_stderr(io) do
            description_length(l_fixed, m, p_ls; paramCompFunc = (p, l, mm) -> error("boom"))
        end
    end
    @test isempty(read(out, String))
    rm(out; force = true)
    @test_logs (:debug, "parameter complexity failed") min_level = Logging.Debug match_mode = :any description_length(
        l_fixed, m, p_ls; paramCompFunc = (p, l, mm) -> error("boom"))

    # a programming error or a Ctrl-C is not an ordinary outcome and must not
    # be absorbed into a NaN parameter complexity
    @test_throws MethodError description_length(l_fixed, m, p_ls;
        paramCompFunc = (p, l, mm) -> throw(MethodError(sin, (1,))))
    @test_throws InterruptException description_length(l_fixed, m, p_ls;
        paramCompFunc = (p, l, mm) -> throw(InterruptException()))
    @test_throws UndefVarError description_length(l_fixed, m, p_ls;
        paramCompFunc = (p, l, mm) -> throw(UndefVarError(:nope)))
end

@testset "fractional_bayes_factor" begin
    nll = evaluate_nll(l_fixed, m, p_ls)
    fc = func_complexity(m)
    n = n_observations(l_fixed)
    k = n_param(l_fixed, m)
    b = inv(sqrt(n))
    nup = exp(1 - log(3))
    expected = (1 - b) * nll - k/2 * log(b) + fc + k/2 * log(2 * pi * nup)
    @test fractional_bayes_factor(l_fixed, m, p_ls) ≈ expected
    terms = fractional_bayes_factor_terms(l_fixed, m, p_ls)
    @test sum(terms) ≈ expected
    @test terms[1] ≈ (1 - b) * nll
    @test fractional_bayes_factor_complexity_penalty(l_fixed, m, p_ls) ≈
          -k/2 * log(b) + fc + k/2 * log(2 * pi * nup)
end

@testset "AIC / BIC" begin
    nll = evaluate_nll(l_fixed, m, p_ls)
    fc = func_complexity(m)
    k = n_param(l_fixed, m)
    n = n_observations(l_fixed)
    @test AIC(l_fixed, m, p_ls) ≈ 2 * nll + 2 * k
    @test BIC(l_fixed, m, p_ls) ≈ 2 * (nll + k/2 * log(n))
    @test BIC_funccompl_penalty(l_fixed, m, p_ls) ≈ 2 * (k/2 * log(n) + fc)
    @test BIC_funccompl(l_fixed, m, p_ls) ≈ 2 * nll + 2 * (k/2 * log(n) + fc)
end

# A profiled sigma is not searched over by the optimizer but is a parameter,
# so AIC and BIC count it like the description length does.
@testset "AIC / BIC count the profiled sigma" begin
    nll = evaluate_nll(l_prof, m, θ_ls)
    fc = func_complexity(m)
    n = n_observations(l_prof)
    k = n_param(l_prof, m)
    @test n_opt_params(l_prof, m) == 2 && k == 3
    # the same count the description length and the Bayes factor use
    @test k == size(information_matrix(l_prof, m, θ_ls), 1)
    @test AIC(l_prof, m, θ_ls) ≈ 2 * nll + 2 * k
    @test BIC(l_prof, m, θ_ls) ≈ 2 * (nll + k/2 * log(n))
    @test BIC_funccompl_penalty(l_prof, m, θ_ls) ≈ 2 * (k/2 * log(n) + fc)
    @test BIC_funccompl(l_prof, m, θ_ls) ≈ 2 * nll + 2 * (k/2 * log(n) + fc)
end

@testset "rational_approx" begin
    # An exactly representable value is returned as itself.  The original port
    # pushed each convergent only after the `x == ai` break, so `rat` never
    # contained the exact value: rational_approx(2.0, 1000) returned just
    # [2001//1000] and snapping to the integer 2 was never offered.
    for v in (2.0, 3.0, -2.0, 0.0, -1.5, 0.75, 0.25, 1//8)
        _, rat = rational_approx(float(v), 1000)
        @test any(r -> float(r) == float(v), rat)
    end
    @test rational_approx(2.0, 1000)[2] == [2//1]
    @test rational_approx(-1.5, 1000)[2] == [-2//1, -3//2]

    # The final candidate is the nearest rational under the denominator bound.
    # The original compared *signed* errors; the two candidates bracket x from
    # opposite sides, so it always chose the one below x.
    for v in (0.7, sqrt(2), 0.123456789, exp(1), 1/3, -0.6180339887, 0.75)
        maxden = 100
        _, rat = rational_approx(v, maxden)
        best = rat[end]
        cand = unique([n//d for d in 1:maxden for n in (floor(Int, v*d), ceil(Int, v*d))])
        nearest = cand[argmin([abs(float(r) - v) for r in cand])]
        @test abs(float(best) - v) <= abs(float(nearest) - v) + 1e-15
    end

    # Convergents respect the denominator bound and improve monotonically.
    for v in (pi, sqrt(2), 0.001, 1234.5678, -pi)
        _, rat = rational_approx(v, 1000)
        @test all(r -> denominator(r) <= 1000, rat)
        errs = [abs(float(r) - v) for r in rat]
        @test issorted(errs; rev = true) || length(errs) <= 1
    end
    @test rational_approx(pi, 1000)[2][end] == 355//113

    # Values that cannot be expanded yield no candidates instead of throwing an
    # InexactError out of the snapping loop.
    for v in (NaN, Inf, -Inf, 1e30, -1e30)
        cf, rat = rational_approx(v, 1000)
        @test isempty(cf) && isempty(rat)
    end
    @test_throws ArgumentError rational_approx(0.5, 0)

    # concrete element type
    @test rational_approx(0.75, 1000)[2] isa Vector{Rational{Int}}
end

@testset "rational_constant_snap reaches an integer constant" begin
    # p[2] sits exactly on 2.0, so the integer is among the candidates and
    # snapping it removes a parameter.
    X = reshape(collect(range(0.5, 3.0; length = 16)), :, 1)
    y = 1.5 .* vec(X) .+ 2.0
    m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    l = GaussianLikelihood(X, y, 0.1)
    p = [1.5, 2.0]

    ri, newmodel, delta_dl, newparam = find_best_snap_for_pidx(l, m, p, 2)
    @test 2//1 in rational_approx(2.0, 1_000)[2]
    @test delta_dl <= 0
    if delta_dl < 0
        @test n_param(newmodel) == 1
        @test length(newparam) == 1
    end

    # a NaN parameter is skipped rather than throwing
    @test_nowarn find_best_snap_for_pidx(l, m, [1.5, NaN], 2)
end

@testset "rational_constant_snap" begin
    # noise-free data of 1/2 * x + 2
    X = reshape(collect(range(0.5, 3.0; length = 16)), :, 1)
    y = 0.5 .* vec(X) .+ 2.0
    m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
    p = [0.5, 2.0]

    @testset "a precise rational parameter is snapped" begin
        # the precision of p[1] makes it more expensive than the rational 1//2
        l = GaussianLikelihood(X, y, 0.001)
        idx, rational, delta_dl, newparam, newmodel = find_best_snap(l, m, p)
        @test idx in (1, 2)
        @test rational == (idx == 1 ? 1//2 : 2)
        @test delta_dl < 0
        @test n_param(newmodel) == 1 && length(newparam) == 1

        snapped, lsnap, psnap = rational_constant_snap(l, m, p)
        @test lsnap === l
        @test n_param(snapped) == 0 && isempty(psnap)
        @test interpret_vec(snapped, X, Float64[]) ≈ y
        # the snapped rational is kept exact and costed as a rational
        @test func_complexity(snapped) ≈ func_complexity(Expr(:call, :+, Expr(:call, :*, 1//2, :(x[1])), 2))
        @test description_length(l, snapped, psnap)[1] < description_length(l, m, p)[1]
        # nothing left to snap
        @test find_best_snap(l, snapped, psnap)[1] == 0
    end

    @testset "an imprecise rational parameter is kept" begin
        # with sigma = 0.1 the rational 1//2 (`/` symbol plus log(2) + log(2))
        # costs more than the parameter complexity of p[1]; only the integer
        # 2 is snapped
        l = GaussianLikelihood(X, y, 0.1)
        snapped, _, psnap = rational_constant_snap(l, m, p)
        @test n_param(snapped) == 1
        @test psnap ≈ [0.5]
        @test interpret_vec(snapped, X, psnap) ≈ y
        rational_model = replace_param_with_const(snapped, 1, 1//2)
        @test description_length(l, snapped, psnap)[1] < description_length(l, rational_model, Float64[])[1]
    end
end

@testset "every public name has a docstring" begin
    @test isempty(Docs.undocumented_names(SymRegModelSelection))
end

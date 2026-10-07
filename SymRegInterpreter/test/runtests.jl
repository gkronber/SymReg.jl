using SymRegInterpreter
using SymRegInterpreter: Instruction, Opcode, ABS, ADD, CONSTANT, COS, DIV, LOG, LOGABS,
    MUL, PARAM, POW, POWABS, POWCONST, SCALEDVAR, SIN, SQR, SQRT, SQRTABS, VAR,
    terminal_opcodes, unary_opcodes, binary_opcodes, opcode, degree,
    create_param_instruction, create_var_instruction, create_scaledvar_instruction,
    code, expr, core_expr, replace_param_with_const, differentiate,
    to_logspace_x_model, to_logspace_y_model, to_logspace_xy_model,
    with_fit_buffer_cache, reset_fit_buffer_cache!, EltypeMismatchError,
    interpret_vec2!, interpret_vecmat!, interpret_grad_seeded!
using Test
using ForwardDiff
using PreallocationTools
using Random



@testset "SymRegInterpreter" begin
    @testset "opcode" begin
        @test degree(ADD) == 2
        @test degree(LOG) == 1
        @test degree(PARAM) == 0
        @test degree(SCALEDVAR) == 0
        @test opcode(:+) == ADD
        @test opcode(:sqrt) == SQRT
        @test opcode(:logabs) == LOGABS
        @test opcode(:sqrtabs) == SQRTABS
        @test degree(SQRTABS) == 1
        @test_throws ArgumentError opcode(:nonesuch)

        # `degree` reads a table derived from the three opcode lists, so every
        # opcode must be in exactly one of them: a new opcode that is added to
        # the `@enum` but to none of the lists fails here rather than in the
        # middle of an interpreter run.
        for opc in instances(Opcode)
            @test degree(opc) in (0, 1, 2)
            @test count(l -> opc in l,
                        (terminal_opcodes, unary_opcodes, binary_opcodes)) == 1
        end
        @test degree(ADD) isa Int
        @test @inferred(degree(ADD)) == 2
    end

    @testset "Model types: ExprModel / CodeModel" begin
        using SymRegInterpreter: ExprModel, CodeModel, code, expr, core_expr
        # expression-backed model: the code is generated lazily and cached
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + x[1])), 1)
        @test m isa ExprModel
        @test m.code_cache === nothing
        c = code(m)
        @test c isa Vector{Instruction{Float64}}
        @test m.code_cache === c        # cached after first call
        @test code(m) === c             # stable across calls
        @test expr(m) === core_expr(m)  # untransformed: expr == core_expr
        X = reshape([2.0, 3.0, 4.0], 3, 1)
        p = [5.0]
        @test interpret_vec(m, X, p) ≈ p[1] .* X[:, 1] .+ X[:, 1]
        @test interpret_vec(code(m), X, p) ≈ p[1] .* X[:, 1] .+ X[:, 1]

        # code-backed model: no expression is stored or constructed
        cm = Model{Float64}(c, 1)
        @test cm isa CodeModel
        @test code(cm) === c
        @test n_param(cm) == n_param(m)
        @test interpret_vec(cm, X, p) ≈ p[1] .* X[:, 1] .+ X[:, 1]

        # the `Model(T, code, n)` / `Model{T}(code, n)` constructors also build
        # code-backed models
        @test Model(Float64, c, 1) isa CodeModel
        @test Model{Float64}(c, 1) isa CodeModel

        # deepcopy keeps both model kinds fully functional
        @test interpret_vec(deepcopy(m), X, p) ≈ interpret_vec(m, X, p)
        @test interpret_vec(deepcopy(cm), X, p) ≈ interpret_vec(cm, X, p)
    end

    @testset "log-space transforms on CodeModel (no expr)" begin
        # f = p1*x1 + p2 built from code only (1: PARAM, 2: VAR, 3: MUL, 4: PARAM, 5: ADD)
        cm = Model{Float64}([create_param_instruction(Float64, 1),
                             create_var_instruction(Float64, 1),
                             Instruction{Float64}(MUL, UInt32(1), UInt32(2), 0.0),
                             create_param_instruction(Float64, 2),
                             Instruction{Float64}(ADD, UInt32(3), UInt32(4), 0.0)], 2)
        X = reshape([2.0, 3.0], 2, 1)
        p = [5.0, 1.0]
        y = p[1] .* X[:, 1] .+ p[2]
        @test interpret_vec(to_logspace_y_model(cm), X, p) ≈ log10.(abs.(y))
        @test interpret_vec(to_logspace_x_model(cm), X, p) ≈ p[1] .* 10.0 .^ X[:, 1] .+ p[2]
        @test interpret_vec(to_logspace_xy_model(cm), X, p) ≈ log10.(abs.(p[1] .* 10.0 .^ X[:, 1] .+ p[2]))
        @test n_param(to_logspace_xy_model(cm)) == 2

        # SCALEDVAR p[k]*x[i] becomes p[k] * 10^x[i] in x-space
        sv = [create_scaledvar_instruction(Float64, 1, 1; val=2.0),
              create_var_instruction(Float64, 2),
              Instruction{Float64}(MUL, UInt32(1), UInt32(2), 0.0)]   # f = p1*x1 * x2
        svm = Model{Float64}(sv, 1)
        X2 = [2.0 10.0; 3.0 20.0]
        # every input variable is log-scaled: f(x) -> f(10^x)
        expected_x = (2.0 .* 10.0 .^ X2[:, 1]) .* 10.0 .^ X2[:, 2]
        @test interpret_vec(to_logspace_x_model(svm), X2, [2.0]) ≈ expected_x
        @test interpret_vec(to_logspace_xy_model(svm), X2, [2.0]) ≈ log10.(abs.(expected_x))

        # code-model transforms agree with the expression-model transforms
        em = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
        emx = to_logspace_xy_model(em)
        cmx = to_logspace_xy_model(Model{Float64}(code(em), 2))
        @test interpret_vec(emx, X, p) ≈ interpret_vec(cmx, X, p)
    end

    @testset "value" begin
        m = Model(Float64, Expr(:->, :(x, p), :(x[1] * p[1] + x[1])), 1)
        X = reshape([1.0, 2.0, 3.0], 3, 1)
        p = [3.0]
        v = interpret_vec(m, X, p)
        @test v ≈ [4.0, 8.0, 12.0]
        @test interpret(m, X, p) ≈ 24.0
    end

    @testset "a nested :block returns its child's index" begin
        # `(x[1] + p[1]) * begin x[1] end`.  The block emits no instruction of
        # its own, and its child `x[1]` is a hashcons hit, so the recursion
        # pushes nothing — the block used to report `length(code)`, i.e. the
        # `+` instruction just emitted, and the model silently computed
        # `(x[1] + p[1])^2` instead.
        body = Expr(:call, :*, :(x[1] + p[1]), Expr(:block, :(x[1])))
        m = Model(Float64, Expr(:->, :(x, p), body), 1)
        X = reshape([1.0, 2.0, 3.0], 3, 1)
        p = [3.0]
        @test interpret_vec(m, X, p) ≈ (X[:, 1] .+ p[1]) .* X[:, 1]
    end

    @testset "operator fusion emits no dead instructions" begin
        # `sqrt(abs(u))`, `log(abs(u))` and `abs(u)^v` fuse to SQRTABS, LOGABS
        # and POWABS.  The argument used to be converted before the fusion was
        # decided, so the `ABS` the fused opcode absorbs was emitted and then
        # orphaned: still evaluated for every row of every batch, and still
        # counted in `length(code)`, which sizes the tape buffers.
        ops(body, np) = [i.opcode for i in
                         code(Model(Float64, Expr(:->, :(x, p), body), np))]

        @test ops(:(sqrt(abs(x[1]))), 0) == [VAR, SQRTABS]
        @test ops(:(log(abs(x[1]))), 0) == [VAR, LOGABS]
        @test ops(:(abs(x[1])^p[1]), 1) == [VAR, PARAM, POWABS]

        # POWCONST keeps precedence over the POWABS fusion and does read the
        # ABS, so that one must stay.
        @test ops(:(abs(x[1])^2), 0) == [VAR, ABS, POWCONST]

        # An `abs` that is also used on its own is a genuine node: it is emitted
        # once for that use, and the fused opcode reads the VAR directly.
        @test ops(:(abs(x[1]) + sqrt(abs(x[1]))), 0) == [VAR, ABS, SQRTABS, ADD]

        # Fusing earlier must not change any value.  Inputs span zero so the
        # `abs` branches matter.
        X = (rand(MersenneTwister(23), 64, 2) .- 0.5) .* 4
        for (body, np) in ((:(sqrt(abs(x[1]))), 0), (:(log(abs(x[1]))), 0),
                           (:(abs(x[1])^2), 0), (:(abs(x[1]) + sqrt(abs(x[1]))), 0),
                           (:(sqrt(abs(x[1] * x[2]))), 0))
            m = Model(Float64, Expr(:->, :(x, p), body), np)
            f = eval(Expr(:->, :(x, p), body))
            want = [Base.invokelatest(f, view(X, i, :), Float64[]) for i in 1:size(X, 1)]
            @test interpret_vec(m, X, Float64[]) ≈ want
        end
    end

    @testset "parameter gradient/jacobian" begin
        # f = x1*p1 + sin(p2) + p1*p2 ; shared p1
        expr = Expr(:->, :(x, p),
            :(x[1] * p[1] + sin(p[2]) + p[1] * p[2]))
        m = Model(Float64, expr, 2)
        X = reshape([1.2, 3.4, 5.6], 3, 1)
        p = [0.5, -0.3]
        grad = zeros(2)
        nll = interpret_grad!(grad, m, X, p)
        # d/dp1 = sum(x) + sum(p2) ; d/dp2 = sum(cos(p2)) + sum(p1)
        @test grad[1] ≈ (sum(X[:, 1]) + 3 * p[2])
        @test grad[2] ≈ 3 * (cos(p[2]) + p[1])
        @test nll ≈ sum(X[:, 1] * p[1] .+ sin(p[2]) .+ p[1] * p[2])

        # jacobian
        v, jac = interpret_jac(m, X, p)
        @test v == X[:, 1] * p[1] .+ sin(p[2]) .+ p[1] * p[2]
        for i in 1:3
            @test jac[i, 1] ≈ X[i, 1] + p[2]
            @test jac[i, 2] ≈ cos(p[2]) + p[1]
        end

        # matches ForwardDiff
        fd = ForwardDiff.gradient(pp -> interpret(m, X, pp), p)
        @test grad ≈ fd
    end

    @testset "ForwardDiff equivalence" begin
        expr = Expr(:->, :(x, p), :(x[1] * p[1] + p[2]^3 + x[2]))
        m = Model(Float64, expr, 2)
        X = rand(Xoshiro(1), 1000, 2) .+ one(Float64)
        p = ones(2)
        grad = zeros(2)
        interpret_grad!(grad, m, X, p)
        fd = ForwardDiff.gradient(pp -> interpret(m, X, pp), p)
        @test grad ≈ fd atol=1e-10

        # reverse rules of POW (parameter in the exponent and in the base) and SQR
        m2 = Model(Float64, Expr(:->, :(x, p), :(x[1]^p[1] + p[2]^x[2] + sqr(x[1] * p[2]))), 2)
        ops2 = [instr.opcode for instr in code(m2)]
        @test POW in ops2 && SQR in ops2
        p2 = [0.7, 1.3]
        grad2 = zeros(2)
        interpret_grad!(grad2, m2, X, p2)
        @test grad2 ≈ ForwardDiff.gradient(pp -> interpret(m2, X, pp), p2)

        # reverse rule of POWABS with negative and positive bases and a parameter
        # in the base and in the (non-integer) exponent; the base derivative
        # reuses the forward value
        m3 = Model(Float64, Expr(:->, :(x, p), :(abs(x[1] - p[1])^p[2] + abs(p[3] * x[2])^(-0.3 * x[1]))), 3)
        @test POWABS in [instr.opcode for instr in code(m3)]
        p3 = [1.5, 1.7, -0.8]
        grad3 = zeros(3)
        interpret_grad!(grad3, m3, X, p3)
        @test grad3 ≈ ForwardDiff.gradient(pp -> interpret(m3, X, pp), p3)
        _, jac3 = interpret_jac(m3, X, p3)
        @test vec(sum(jac3; dims = 1)) ≈ grad3
    end

    # libm (`_pow`, `_exp`, ...) and Base agree exactly on zero, infinite and NaN
    # results; finite results may differ in the last bits (glibc differs from Base
    # where Apple's libm does not), within 2 ulps
    agree(a, b) = isequal(a, b) || (isfinite(a) && isfinite(b) && signbit(a) == signbit(b) &&
                                    abs(a - b) <= 2 * eps(max(abs(a), abs(b))))
    # up to 10 argument tuples where `f` and `g` disagree, with both results, so
    # that a failing test shows them
    disagreements(f, g, args) = first([(v, f(v...), g(v...)) for v in args if !agree(f(v...), g(v...))], 10)

    @testset "POW: libm pow behind the NaN guard of NaNMath.pow" begin
        rng = Xoshiro(8)
        pw = SymRegInterpreter._pow
        nanpow = SymRegInterpreter.NaNMath.pow
        for T in (Float32, Float64)
            x = T.(exp.(2 .* randn(rng, 10_000)) .* rand(rng, (-1, 1), 10_000))
            # non-integer and integer exponents (a negative base gives NaN only for the former)
            y = T.([isodd(i) ? 2 * randn(rng) : rand(rng, -3:3) for i in 1:10_000])
            # two ulps: for y == -2 Base computes inv(x)^2, up to ≈ 1.8 ulp from
            # the exact value, while libm stays within half an ulp
            @test isempty(disagreements(pw, nanpow, zip(x, y)))
            sx = T[0, -0.0, 1, -1, -2, Inf, -Inf, NaN, nextfloat(T(0)), floatmin(T), floatmax(T), 2, 0.5, -0.5]
            sy = T[0, -0.0, 1, -1, 2, -2, 3, 0.5, -0.5, 2.5, -2.5, Inf, -Inf, NaN, 200, -200]
            @test isempty(disagreements(pw, nanpow, Iterators.product(sx, sy)))
        end
    end

    @testset "EXP / LOG / LOGABS: libm for plain floats, Base for duals" begin
        rng = Xoshiro(9)
        lexp = SymRegInterpreter._exp
        llog = SymRegInterpreter._log
        for T in (Float32, Float64)
            # two ulps: Base's Float32 exp alone is up to ≈ 0.84 ulp off
            x = T.(10 .* randn(rng, 10_000))
            @test isempty(disagreements(lexp, exp, zip(x)))
            z = T.(exp.(10 .* randn(rng, 10_000)))
            @test isempty(disagreements(llog, log, zip(z)))
            # special cases as in Base (log only for x >= 0, the kernels guard the rest)
            se = T[0, -0.0, 1, -1, Inf, -Inf, NaN, nextfloat(T(0)), floatmin(T), floatmax(T), 88.7, -103.9, 709.7, -745.1]
            @test isempty(disagreements(lexp, exp, zip(se)))
            sl = T[0, -0.0, 1, Inf, NaN, nextfloat(T(0)), floatmin(T), floatmax(T), 0.5, 2]
            @test isempty(disagreements(llog, log, zip(sl)))

            # the kernels' guards: LOG of a negative argument and LOGABS of zero are NaN
            X = reshape(T[-2, -0.0, 0, 0.5, 3], :, 1)
            mlog = Model(T, Expr(:->, :(x, p), :(log(x[1]))), 0)
            mlogabs = Model(T, Expr(:->, :(x, p), :(log(abs(x[1])))), 0)
            @test all(agree.(interpret_vec(mlog, X, T[]), T[NaN, -Inf, -Inf, log(T(0.5)), log(T(3))]))
            @test all(agree.(interpret_vec(mlogabs, X, T[]), T[log(T(2)), NaN, NaN, log(T(0.5)), log(T(3))]))
        end

        # ForwardDiff duals take Base's exp / log; derivatives match ForwardDiff on
        # the plain expression, and the reverse sweep agrees
        m = Model(Float32, Expr(:->, :(x, p), :(exp(p[1] * x[1]) + log(abs(x[1] - p[2])) + log(p[3] * x[1]))), 3)
        X = rand(Xoshiro(3), Float32, 200, 1) .+ 0.5f0
        p = Float32[0.4, 2.0, 1.3]
        g = ForwardDiff.gradient(pp -> interpret(m, X, pp), p)
        @test g ≈ ForwardDiff.gradient(pp -> sum(exp.(pp[1] .* X[:, 1]) .+ log.(abs.(X[:, 1] .- pp[2])) .+
                                                 log.(pp[3] .* X[:, 1])), p)
        grad = zeros(Float32, 3)
        interpret_grad!(grad, m, X, p)
        @test grad ≈ g
    end

    @testset "POWABS: libm pow for plain floats, Base ^ for duals" begin
        rng = Xoshiro(7)
        powabs = SymRegInterpreter._powabs
        for T in (Float32, Float64)
            x = T.(exp.(2 .* randn(rng, 10_000)) .* rand(rng, (-1, 1), 10_000))
            y = T.(2 .* randn(rng, 10_000))
            @test isempty(disagreements(powabs, (a, b) -> abs(a)^b, zip(x, y)))
            # special cases as in Base
            sx = T[0, -0.0, 1, -1, Inf, -Inf, NaN, nextfloat(T(0)), floatmin(T), floatmax(T), 2, 0.5]
            sy = T[0, -0.0, 1, -1, 0.5, -0.5, 2.5, -2.5, Inf, -Inf, NaN, 200, -200]
            @test isempty(disagreements(powabs, (a, b) -> abs(a)^b, Iterators.product(sx, sy)))
        end

        # ForwardDiff duals take Base ^ and ForwardDiff's rule: derivatives of the
        # interpreted model match ForwardDiff on the plain expression, and the
        # reverse sweep on the libm forward values agrees with both
        m = Model(Float32, Expr(:->, :(x, p), :(abs(x[1] - p[1])^p[2])), 2)
        X = rand(Xoshiro(2), Float32, 200, 1) .+ 0.5f0
        p = Float32[0.7, 1.6]
        g = ForwardDiff.gradient(pp -> interpret(m, X, pp), p)
        @test g ≈ ForwardDiff.gradient(pp -> sum(abs.(X[:, 1] .- pp[1]) .^ pp[2]), p)
        grad = zeros(Float32, 2)
        interpret_grad!(grad, m, X, p)
        @test grad ≈ g
    end

    @testset "mixed Float32 data / Float64 param" begin
        # the buffers are Float32, so Float64 parameters would lose precision
        m = Model(Float32, Expr(:->, :(x, p), :(x[1] * p[1] + p[2])), 2)
        X = reshape(Float32[1, 2, 3], 3, 1)
        p = Float64[2.0, 1.0]
        grad = zeros(Float64, 2)
        @test_throws EltypeMismatchError interpret_grad!(grad, m, X, p)
        @test_throws EltypeMismatchError interpret_vec(m, X, p)
        @test_throws EltypeMismatchError ForwardDiff.gradient(pp -> interpret(m, X, pp), p)
    end

    @testset "nested Hessian" begin
        expr = Expr(:->, :(x, p), :(x[1] * p[1] + p[1]^2 + p[2]))
        m = Model(Float64, expr, 2)
        X = reshape([1.0, 2.0, 3.0], 3, 1)
        p = [1.0, 2.0]
        f = pp -> interpret(m, X, pp)
        H = ForwardDiff.hessian(f, p)
        @test H[1, 1] ≈ 6.0  # d2/dp1^2 of 3*(x1*p1 + p1^2): sum(2)=6
    end

    @testset "SCALEDVAR" begin
        # p1*x1 + p2*x1
        codevec = SymRegInterpreter.Instruction{Float64}[
            create_scaledvar_instruction(Float64, 1, 1; val=2.0),
            create_scaledvar_instruction(Float64, 2, 1; val=3.0),
            Instruction{Float64}(ADD, 1, 2, 0.0),
        ]
        m = Model{Float64}(codevec, 2)
        X = reshape([1.0, 2.0], 2, 1)
        p = [10.0, 20.0]
        @test interpret_vec(m, X, p) ≈ (p[1] + p[2]) .* X[:, 1]
        grad = zeros(2)
        interpret_grad!(grad, m, X, p)
        @test grad[1] ≈ sum(X[:, 1])
        @test grad[2] ≈ sum(X[:, 1])

        numvar = maximum(i -> begin instr = code(m)[i]; (instr.opcode == VAR || instr.opcode == SCALEDVAR) ? Int(instr.arg1idx) : 0 end, eachindex(code(m)))
        jacx = zeros(eltype(p), size(X, 1), numvar)
        jacp = zeros(eltype(p), size(X, 1), n_param(m))
        y = zeros(eltype(p), size(X, 1))
        interpret_jac!(y, jacp, jacx, code(m), X, p; batchsize=128)
        @test jacx[:, 1] ≈ [p[1] + p[2], p[1] + p[2]]
    end

    @testset "jacp and jacx" begin
        codevec = SymRegInterpreter.Instruction{Float64}[
            create_scaledvar_instruction(Float64, 1, 1; val=1.0),  # p1*x1
            create_scaledvar_instruction(Float64, 2, 1; val=1.0),  # p2*x1
            Instruction{Float64}(VAR, 2, 0, 0.0),               # x2 (not in a param product)
        ]
        # root: (p1*x1 + p2*x1) * x2
        push!(codevec, Instruction{Float64}(ADD, 1, 2, 0.0))
        push!(codevec, Instruction{Float64}(MUL, 3, 4, 0.0))
        m = Model{Float64}(codevec, 2)
        X = Float64[1 10; 2 20; 3 30]
        p = Float64[2, 3]
        jacp = zeros(size(X, 1), 2)
        jacx = zeros(size(X, 1), 2)
        y = zeros(size(X, 1))
        interpret_jac!(y, jacp, jacx, code(m), X, p; batchsize=128)
        @test y ≈ (p[1] .+ p[2]) .* X[:, 1] .* X[:, 2]
        # d/dp1 = x1*x2 ; d/dp2 = x1*x2
        @test jacp[:, 1] ≈ X[:, 1] .* X[:, 2]
        @test jacp[:, 2] ≈ X[:, 1] .* X[:, 2]
        # d/dx1 = (p1+p2)*x2 ; d/dx2 = (p1+p2)*x1
        @test jacx[:, 1] ≈ (p[1] + p[2]) .* X[:, 2]
        @test jacx[:, 2] ≈ (p[1] + p[2]) .* X[:, 1]
    end

    @testset "batched input Jacobian (regression)" begin
        # >128 rows forces multiple interpreter batches (128 + partial), which
        # regressed the shared input Jacobian: each batch wrote into rows 1:128
        # instead of its own global rows.  Model: p1*x1^2 + x2.
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1]^2 + x[2])), 1)
        n = 300
        X = rand(MersenneTwister(42), n, 2) .+ 0.5
        p = [1.7]
        jacx = zeros(n, 2)
        jacp = zeros(n, 1)
        y = zeros(n)
        interpret_jac!(y, jacp, jacx, code(m), X, p; batchsize=128)
        @test y ≈ p[1] .* X[:, 1] .^ 2 .+ X[:, 2]
        # param Jacobian across batches
        @test jacp[:, 1] ≈ X[:, 1] .^ 2
        # input Jacobian across batches
        @test jacx[:, 1] ≈ 2 * p[1] .* X[:, 1]
        @test jacx[:, 2] ≈ ones(n)

        # a nonlinear unary op under the root (exercises log10 in the adjoint)
        m2 = Model(Float64, Expr(:->, :(x, p), :(p[1] * log10(x[1]) + x[2])), 1)
        jacx2 = zeros(n, 2)
        interpret_jac!(y, nothing, jacx2, code(m2), X, p; batchsize=128)
        @test jacx2[:, 1] ≈ p[1] ./ (X[:, 1] * log(10))           # d/dx1
        @test jacx2[:, 2] ≈ ones(n)
    end

    @testset "full batches match a single batch" begin
        # batchsize = 7 runs the loop over full batches, with (40 rows) and
        # without (35 rows) a remaining partial batch
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + sin(x[2] * p[2]))), 2)
        p = [0.7, 1.3]
        for n in (40, 35)
            X = rand(MersenneTwister(5), n, 2)
            yref = p[1] .* X[:, 1] .+ sin.(X[:, 2] .* p[2])
            @test interpret(code(m), X, p; batchsize = 7) ≈ sum(yref)

            dcode, fidx, dfidx = differentiate(code(m), 2)
            dyref = p[2] .* cos.(X[:, 2] .* p[2])
            f = zeros(n); df = zeros(n)
            interpret_vec2!(f, df, dcode, X, p, fidx, dfidx; batchsize = 7)
            @test f ≈ yref
            @test df ≈ dyref
            outs = zeros(n, 2)
            interpret_vecmat!(outs, dcode, X, p, [fidx, dfidx]; batchsize = 7)
            @test outs ≈ [yref dyref]
        end
    end

    @testset "jacx has one column per column of X, used or not" begin
        # `m3` ignores the second column of `X`.  Both entry points must still
        # treat the input Jacobian as `size(X, 2)` wide: `interpret_jac!`
        # asserts that shape, and `interpret_jac` used to size its internal
        # input Jacobian by the highest variable index the code references,
        # which for this model is 1.
        X = [1.0 4.0; 2.0 5.0; 3.0 6.0]
        p = [2.0]
        m3 = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1]^2)), 1)

        y = zeros(3)
        jacx = zeros(3, size(X, 2))
        interpret_jac!(y, nothing, jacx, code(m3), X, p)
        @test jacx[:, 1] ≈ 2 * p[1] .* X[:, 1]
        @test all(iszero, jacx[:, 2])          # unused column, but present

        # a jacx sized by the number of variables the code uses is not the
        # convention and is rejected rather than silently accepted
        @test_throws AssertionError interpret_jac!(y, nothing, zeros(3, 1), code(m3), X, p)

        v, jp = interpret_jac(m3, X, p)
        @test v ≈ p[1] .* X[:, 1] .^ 2
        @test jp[:, 1] ≈ X[:, 1] .^ 2
    end

    @testset "buffer reuse is allocation-free + task-local" begin
        model = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
        X = rand(MersenneTwister(7), 512, 2)
        p = [1.5, -0.7]
        nrows = size(X, 1)

        # warm path reuses the cached task-local tape buffers.  Retrieving them
        # from the buffer cache performs a small per-call lookup allocation (a
        # key tuple/closure); the large tape workspaces are reused, never
        # re-allocated.  Bound the cost so a tape reallocation (hundreds of KB)
        # would still fail.
        with_fit_buffer_cache() do
            v = zeros(nrows)
            interpret_vec!(v, code(model), X, p; batchsize=128)   # warm (JIT + buffer creation)
            a = @allocated begin
                for _ in 1:1000
                    interpret_vec!(v, code(model), X, p; batchsize=128)
                end
                nothing
            end
            @test a / 1000 < 1024
        end

        # distinct per-task caches and correct concurrent results
        ntasks = max(Threads.nthreads(), 2)
        caches = Vector{Any}(undef, ntasks)
        results = zeros(ntasks)
        @sync for t in 1:ntasks
            Threads.@spawn with_fit_buffer_cache() do
                v = interpret_vec(code(model), X, p; batchsize=128)
                caches[t] = v
                results[t] = sum(v)
            end
        end
        expected = sum(p[1] .* X[:, 1] .^ 2 .+ p[2] .* X[:, 2])
        @test all(isapprox.(results, expected))
        @test length(unique(objectid.(caches))) == ntasks   # buffers distinct per task
    end

    @testset "grow-to-fit buffer reuse" begin
        # The buffer cache reuses a larger cached workspace when a smaller or
        # equal size is requested, instead of allocating a fresh buffer per
        # size.  Evaluating a smaller model after a larger one (once separate
        # pools see the larger allocation first) must stay correct AND must not
        # reallocate the tape workspace on repeat calls.
        mbig = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^3 + sin(p[2]*x[2]) + p[1]*x[1]*x[2] + x[2]^2 + exp(x[1]))), 2)
        msmall = Model(Float64, Expr(:->, :(x, p), :(p[1]*x[1]^2 + p[2]*x[2])), 2)
        X = rand(MersenneTwister(3), 256, 2)
        p = [1.3, -0.5]
        expected_big = p[1] .* X[:,1].^3 .+ sin.(p[2].*X[:,2]) .+ p[1].*X[:,1].*X[:,2] .+ X[:,2].^2 .+ exp.(X[:,1])

        with_fit_buffer_cache() do
            @test interpret_vec(code(mbig), X, p) ≈ expected_big
            v = zeros(size(X, 1))
            interpret_vec!(v, code(mbig), X, p)
            # warm the smaller model (it reuses the larger tape allocation), then
            # a repeated smaller-model call must be allocation-light.
            interpret_vec!(v, code(msmall), X, p; batchsize=128)
            a = @allocated begin
                for _ in 1:1000
                    interpret_vec!(v, code(msmall), X, p; batchsize=128)
                end
                nothing
            end
            @test a / 1000 < 512
            @test v ≈ p[1] .* X[:,1].^2 .+ p[2] .* X[:,2]

            # gradient reuses the larger tape too and stays correct
            g = zeros(2)
            interpret_grad!(g, code(msmall), X, p; batchsize=128)
            fd = ForwardDiff.gradient(pp -> interpret(code(msmall), X, pp), p)
            @test g ≈ fd
        end
    end

    @testset "the reverse sweep zeroes only the columns it uses" begin
        # `tapediffs` is a grow-to-fit workspace, so once a large model has
        # been evaluated it keeps that model's column count for the rest of
        # the task.  Zeroing the whole buffer therefore charged every later
        # gradient for the largest program the task had ever seen — measured
        # at 19x on a 3-instruction model after a 203-instruction one, which
        # in a model search is paid by the majority of candidates.
        #
        # The sweep reads only the `nactive x length(code)` corner, so the
        # sentinel written outside it below must survive the small model's
        # gradient.  That is only true while the `fill!` stays restricted.
        terms = [:(p[1] * x[1]^$k + p[2] * sin($k * x[2])) for k in 1:20]
        mbig = Model(Float64, Expr(:->, :(x, p), foldl((a, b) -> :($a + $b), terms)), 2)
        msmall = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2])), 2)
        X = rand(MersenneTwister(5), 64, 2)
        p = [1.3, -0.5]

        with_fit_buffer_cache() do
            interpret_grad!(zeros(2), code(mbig), X, p; batchsize = 64)

            nsmall = length(code(msmall))
            buf = SymRegInterpreter.get_matrix_scratch_diffcache(
                Float64, :tape_diffs, 64, nsmall).du
            @test size(buf, 2) > nsmall          # the big model grew it
            fill!(buf, 7.0)

            g = zeros(2)
            interpret_grad!(g, code(msmall), X, p; batchsize = 64)
            @test g ≈ ForwardDiff.gradient(pp -> interpret(code(msmall), X, pp), p)
            @test all(==(7.0), @view buf[:, nsmall + 1:end])
        end
    end

    @testset "the reverse sweep skips parameter-independent instructions" begin
        # `_step_backwards!` at an instruction propagates its adjoint to its
        # children, and only PARAM / SCALEDVAR seed a parameter gradient.  A
        # parameter-free subtree therefore contributes nothing to `grad`, yet
        # its reverse kernels used to run over every row regardless — 1.8x on
        # the whole gradient for the model below.
        #
        # `p[1]*x[1] + sin(cos(x[2]))`: the ADD is parameter-dependent, so it is
        # visited and writes an adjoint into the SIN column.  The SIN itself is
        # not, so it is skipped, and the COS and VAR(2) columns below it must
        # stay at the zero the sweep's `fill!` left there.  An unmasked sweep
        # writes cos'/sin' into them, so this pins the skip rather than a time.
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + sin(cos(x[2])))), 1)
        c = code(m)
        X = 0.2 .+ 0.6 .* rand(MersenneTwister(29), 48, 2)
        p = [1.4]

        dep = SymRegInterpreter._reverse_mask(c, false)
        @test dep == [i.opcode in (PARAM, MUL, ADD) for i in c]

        icos = findfirst(i -> c[i].opcode == COS, eachindex(c))
        ivar2 = Int(c[icos].arg1idx)
        isin = findfirst(i -> c[i].opcode == SIN, eachindex(c))

        with_fit_buffer_cache() do
            g = zeros(1)
            interpret_grad!(g, c, X, p; batchsize = 48)
            @test g ≈ ForwardDiff.gradient(pp -> interpret(c, X, pp), p)

            td = SymRegInterpreter.get_matrix_scratch_diffcache(
                Float64, :tape_diffs, 48, length(c)).du
            @test all(iszero, @view td[1:48, icos])    # skipped
            @test all(iszero, @view td[1:48, ivar2])   # skipped
            @test !all(iszero, @view td[1:48, isin])   # written by the ADD above it
        end

        # An input Jacobian is seeded by VAR too, so that sweep must keep the
        # subtrees the parameter-only mask drops, and still produce every column.
        with_fit_buffer_cache() do
            v = zeros(48); jx = zeros(48, 2)
            interpret_jac!(v, nothing, jx, c, X, p; batchsize = 48)
            @test !all(iszero, @view jx[:, 2])
            @test SymRegInterpreter._reverse_mask(c, true) ==
                  [i.opcode != CONSTANT for i in c]
        end
    end

@testset "replace_param_with_const" begin
        m = Model(Float64, Expr(:->, :(x, p), :(x[1] * p[1] + p[2] * x[2])), 2)
        m2 = replace_param_with_const(m, 1, 5.0)
        @test n_param(m2) == 1
        X = Float64[1 2; 3 4]
        p2 = [7.0]
        @test interpret_vec(m2, X, p2) ≈ (5.0 .* X[:, 1] .+ 7.0 .* X[:, 2])

        # regression: replacing a *non-first* parameter (a lower-index parameter
        # remains) used to throw "internal: param N not removed"
        m3 = Model(Float64, Expr(:->, :(x, p), :(x[1] * p[1] + p[2] * x[2])), 2)
        m4 = replace_param_with_const(m3, 2, 3.0)
        @test n_param(m4) == 1
        p1 = [5.0]
        @test interpret_vec(m4, X, p1) ≈ (5.0 .* X[:, 1] .+ 3.0 .* X[:, 2])

        # regression: a SCALEDVAR whose parameter is replaced must also be
        # turned into a constant (not silently kept).  SCALEDVAR is only built
        # from code (not from an expression), so construct the model directly.
        code5 = [create_scaledvar_instruction(Float64, 1, 1; val=2.0),
                 create_scaledvar_instruction(Float64, 2, 2; val=4.0),
                 Instruction{Float64}(ADD, UInt32(1), UInt32(2), 0.0)]
        m5 = Model(code5, 2)
        m6 = replace_param_with_const(m5, 1, 2.0)
        @test n_param(m6) == 1
        p2b = [4.0]
        @test interpret_vec(m6, X, p2b) ≈ (2.0 .* X[:, 1] .+ 4.0 .* X[:, 2])

        # a non-integer rational is kept exact as DIV(CONSTANT n, CONSTANT d),
        # an integer one is a single constant
        m7 = replace_param_with_const(m, 1, -1//3)
        divs = filter(instr -> instr.opcode == DIV, code(m7))
        @test length(divs) == 1
        num, den = code(m7)[divs[1].arg1idx], code(m7)[divs[1].arg2idx]
        @test num.opcode == den.opcode == CONSTANT
        @test (num.val, den.val) == (-1.0, 3.0)
        @test interpret_vec(m7, X, p2) ≈ (-1 / 3 .* X[:, 1] .+ 7.0 .* X[:, 2])
        @test count(instr -> instr.opcode == CONSTANT, code(replace_param_with_const(m, 1, 2//1))) == 1
        m8 = replace_param_with_const(m5, 2, 3//2)  # SCALEDVAR
        @test DIV in [instr.opcode for instr in code(m8)]
        @test interpret_vec(m8, X, [4.0]) ≈ (4.0 .* X[:, 1] .+ 1.5 .* X[:, 2])
        # a rational literal in an expression is converted the same way
        m9 = Model(Float64, Expr(:->, :(x, p), Expr(:call, :*, 1//3, :(x[1]))), 0)
        @test [instr.opcode for instr in code(m9)] == [CONSTANT, CONSTANT, DIV, VAR, MUL]
        @test interpret_vec(m9, X, Float64[]) ≈ X[:, 1] ./ 3
    end
    @testset "differentiate" begin
        # f and df/dx[1] evaluated in one pass; both read at the indices
        # `differentiate` returned (via a code prefix -- dcode is topologically
        # sorted, so dcode[1:i] is a valid program rooted at i).  Neither index
        # is guaranteed to be the last instruction: hashconsing can collapse the
        # root's derivative onto an earlier node.  Checked against the
        # interpreter's reverse-mode jacx.
        function check_diff(expr, nparam; varidx=1)
            m = Model(Float64, Expr(:->, :(x, p), expr), nparam)
            dcode, fidx, dfidx = differentiate(code(m), varidx)
            p = pdiff[1:nparam]
            f = interpret_vec(dcode[1:fidx], Xdiff, p)
            df = interpret_vec(dcode[1:dfidx], Xdiff, p)
            y = zeros(size(Xdiff, 1))
            jacp = zeros(size(Xdiff, 1), nparam)
            jacx = zeros(size(Xdiff, 1), size(Xdiff, 2))
            interpret_jac!(y, jacp, jacx, code(m), Xdiff, p; batchsize=64)
            @test f ≈ y
            @test df ≈ jacx[:, varidx]
            dcode, fidx
        end

        Xdiff = 0.1 .+ 0.8 .* rand(MersenneTwister(11), 40, 2)  # safe for log/asin/sqrt
        pdiff = [0.7, 1.3]

        # hand-checked value and derivative
        dcode, fidx = check_diff(:(x[1] * p[1] + sin(x[1] * p[1])), 1)
        p1 = pdiff[1]
        @test interpret_vec(dcode[1:fidx], Xdiff, pdiff) ≈ p1 .* Xdiff[:, 1] .+ sin.(p1 .* Xdiff[:, 1])
        _, _, dfi = differentiate(code(Model(Float64, Expr(:->, :(x, p), :(x[1] * p[1] + sin(x[1] * p[1]))), 1)), 1)
        @test interpret_vec(dcode[1:dfi], Xdiff, pdiff) ≈ p1 .+ p1 .* cos.(p1 .* Xdiff[:, 1])

        # hashconsing: x[1]*x[2] exists exactly once, shared by f and df
        m = Model(Float64, Expr(:->, :(x, p), :(x[1] * x[2] + sin(x[1] * x[2]))), 0)
        dcode, fidx, dfidx = differentiate(code(m), 1)
        varnodes = [i for i in eachindex(dcode) if dcode[i].opcode == VAR]
        @test length(varnodes) == 2
        prodnodes = count(eachindex(dcode)) do i
            dcode[i].opcode == MUL &&
                ((dcode[i].arg1idx, dcode[i].arg2idx) in
                 ((varnodes[1], varnodes[2]), (varnodes[2], varnodes[1])))
        end
        check_diff(:(x[1] * p[1] + x[2] / p[2]), 2)
        check_diff(:(sin(x[1] * p[1]) * cos(x[2]) - exp(x[1]) * tanh(x[2])), 1)
        check_diff(:(log(x[1]) + log10(x[2]) + logabs(x[1]) + sqrt(x[2])), 0)
        check_diff(:(abs(x[1] - 0.5) + sign(x[2]) * x[1] + (x[1] - 0.5)^3), 0)
        check_diff(:(asin(x[1]) + sinh(x[1]) + cosh(x[2]) + tan(x[1])), 0)
        check_diff(:(inv(x[1]) - x[2]^2 + abs(x[1])^p[1]), 1)
        check_diff(:(-x[1] + x[2] * x[2] * x[2]), 0)
        check_diff(:(x[2] * p[1] + sin(x[2])), 1; varidx=2)

        # the rules of SIGN, SQR, SQRT, COS, COSH, TAN, DIV and POW, each on an
        # argument that depends on the differentiation variable
        check_diff(:(sign(x[1] - 0.5) * x[2] + sqr(x[1]) + sqrt(x[1]) + cos(x[1]) + cosh(x[1])), 0)
        check_diff(:(tan(x[1] * p[1])), 1)
        check_diff(:(x[1] / (p[1] + x[2]) + x[2] / x[1]), 1)
        check_diff(:(x[1]^(p[1] * x[2]) + x[2]^x[1]), 1)

        # LOG10 inside the derivative chain (not just as a zero derivative of
        # an x-independent node): d log10(x1) = 1/(x1 * log(10))
        check_diff(:(log10(x[1])), 0)

        # POWABS with a negative base: the inner derivative v*sign(u)*u'/|u|
        # must carry the sign of u (base always negative on the data range)
        check_diff(:(abs(-0.4 - x[1])^p[1] + log10(x[1]) * x[2]), 1)

        # SQRTABS: positive and negative arguments (sign of u must carry through)
        check_diff(:(sqrtabs(x[1] - 0.5) + sqrt(abs(-0.2 - x[2]))), 0)
        check_diff(:(sqrtabs(x[1] - 0.5) + sqrt(abs(-0.2 - x[2]))), 0; varidx=2)

        # SCALEDVAR: d(p1*x1 + p2*x2)/dx1 = p1
        svcode = [create_scaledvar_instruction(Float64, 1, 1; val=2.0),
                  create_scaledvar_instruction(Float64, 2, 2; val=4.0),
                  Instruction{Float64}(ADD, UInt32(1), UInt32(2), 0.0)]
        dcode, fidx, dfidx = differentiate(svcode, 1)
        f = interpret_vec(dcode[1:fidx], Xdiff, pdiff)
        df = interpret_vec(dcode[1:dfidx], Xdiff, pdiff)
        @test f ≈ pdiff[1] .* Xdiff[:, 1] .+ pdiff[2] .* Xdiff[:, 2]
        @test df ≈ fill(pdiff[1], size(Xdiff, 1))

        # root independent of x[1]: derivative is identically zero
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[2])), 1)
        dcode, fidx, dfidx = differentiate(code(m), 1)
        @test interpret_vec(dcode[1:dfidx], Xdiff, pdiff) ≈ zeros(size(Xdiff, 1))
        @test interpret_vec(dcode[1:fidx], Xdiff, pdiff) ≈ pdiff[1] .* Xdiff[:, 2]

        # hashconsing can collapse the root's derivative onto an already
        # emitted node, so `dfidx` is not in general the last instruction.
        # `prepare(::RARLikelihood, ...)` used to assert that it was and threw.
        mz = Model(Float64, Expr(:->, :(x, p), :(p[1] + 0.0)), 1)
        dcode, fidx, dfidx = differentiate(code(mz), 1)
        @test dfidx < length(dcode)
        @test interpret_vec(dcode[1:dfidx], Xdiff, pdiff) ≈ zeros(size(Xdiff, 1))
        @test interpret_vec(dcode[1:fidx], Xdiff, pdiff) ≈ fill(pdiff[1], size(Xdiff, 1))

        # pruning: dead nodes (unreachable from root) are dropped
        codevec = [create_var_instruction(Float64, 1),
                Instruction{Float64}(SIN, UInt32(1), UInt32(0), 0.0),  # dead
                create_var_instruction(Float64, 2),
                Instruction{Float64}(MUL, UInt32(1), UInt32(3), 0.0)]
        dcode, fidx, dfidx = differentiate(codevec, 1)
        @test !any(i -> dcode[i].opcode == SIN, eachindex(dcode))
        @test interpret_vec(dcode[1:dfidx], Xdiff, pdiff) ≈ Xdiff[:, 2]
    end
    @testset "seeded gradient sweep" begin
        Xd = 0.1 .+ 0.8 .* rand(MersenneTwister(13), 40, 2)
        pd = [0.7, 1.3]
        m = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + sin(x[1] * p[1]) + p[2] * x[2])), 2)
        n = size(Xd, 1)
        # single seed at the root with unit weights: equivalent to interpret_grad!
        g1 = zeros(2); g2 = zeros(2)
        interpret_grad!(g1, code(m), Xd, pd)
        interpret_grad_seeded!(g2, code(m), Xd, pd, (length(code(m)) => ones(n),))
        @test g1 ≈ g2

        # objective: Σ_i (λf_i * f_i + λg_i * g_i) over the differentiated code;
        # compare the seeded sweep against ForwardDiff through the same scalar
        dcode, fidx, dfidx = differentiate(code(m), 1)
        f = interpret_vec(dcode[1:fidx], Xd, pd)
        g = interpret_vec(dcode[1:dfidx], Xd, pd)
        λf = 0.3 .* (1:n) .+ 1.0
        λg = -0.7 .* (1:n) .+ 2.0
        obj = sum(λf .* f .+ λg .* g)
        gs = zeros(2)
        interpret_grad_seeded!(gs, dcode, Xd, pd,
            (fidx => λf, dfidx => λg); batchsize=16)   # forces multiple batches
        gfd = ForwardDiff.gradient(p -> sum(λf .* interpret_vec(dcode[1:fidx], Xd, p) .+
                                            λg .* interpret_vec(dcode[1:dfidx], Xd, p)), pd)
        @test gs ≈ gfd atol=1e-10

        # multi-variate differentiation: f and all input partials in one
        # vector; chains share the primal and hashcons cache
        mm = Model(Float64, Expr(:->, :(x, p), :(p[1] * x[1] + p[2] * x[1] * x[2] + p[1] * x[2]^2)), 2)
        dcode, fidx, dfidxs = differentiate(code(mm), [1, 2])
        outs = zeros(size(Xd, 1), 3)
        interpret_vecmat!(outs, dcode, Xd, pd, [fidx; dfidxs])
        # reverse-mode reference for both input Jacobians
        y = zeros(size(Xd, 1))
        jacx = zeros(size(Xd, 1), 2)
        interpret_jac!(y, nothing, jacx, code(mm), Xd, pd; batchsize=64)
        @test outs[:, 1] ≈ y          # primal
        @test outs[:, 2] ≈ jacx[:, 1]
        @test outs[:, 3] ≈ jacx[:, 2]
        # every chain is read at the index `differentiate` returned for it
        @test interpret_vec(dcode[1:dfidxs[2]], Xd, pd) ≈ jacx[:, 2]
        @test interpret_vec(dcode[1:dfidxs[1]], Xd, pd) ≈ jacx[:, 1]
        @test interpret_vec(dcode[1:fidx], Xd, pd) ≈ interpret_vec(code(mm), Xd, pd)
    end

    @testset "the fit buffer cache lives for the task, and can be reset" begin
        probe(n) = SymRegInterpreter.get_vector_scratch_diffcache(Float64, :lifetime_probe, n)

        # The cache is scoped to the task, not to the block: a second block in
        # the same task gets the same workspace back, still at the size the
        # first one grew it to.  This is deliberate -- a search that started
        # from an empty cache on every block would reallocate every tape -- and
        # is what makes the high-water mark permanent.
        d1 = with_fit_buffer_cache() do
            probe(64)
        end
        d2 = with_fit_buffer_cache() do
            probe(8)
        end
        @test d1 === d2
        @test length(d2.du) == 64

        # a reset drops it, and the next request allocates at the size it asks
        # for rather than at the high-water mark
        with_fit_buffer_cache() do
            @test reset_fit_buffer_cache!() isa SymRegInterpreter.FitBufferCache
            d3 = probe(8)
            @test d3 !== d1
            @test length(d3.du) == 8
        end

        # the reset applies to the cache a later block installs, so it is
        # equally valid outside one
        reset_fit_buffer_cache!()
        with_fit_buffer_cache() do
            @test length(probe(8).du) == 8
        end

        # and a new task starts from nothing whatever this one has grown
        with_fit_buffer_cache() do
            probe(256)
        end
        t = Threads.@spawn with_fit_buffer_cache() do
            length(probe(8).du)
        end
        @test fetch(t) == 8
        @test length(probe(8).du) == 8   # no block active here: nothing cached
    end
end



@testset "every public name has a docstring" begin
    @test isempty(Docs.undocumented_names(SymRegInterpreter))
end

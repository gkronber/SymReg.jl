using BenchmarkTools
using SymRegInterpreter
using SymRegInterpreter: code, with_fit_buffer_cache

const SUITE = BenchmarkGroup()

# Benchmark the linear-code interpreter: forward interpretation, parameter-
# gradient, and parameter-Jacobian evaluation.  Buffers are drawn from the
# buffer cache, so a warm call reuses them and is allocation-free.
# Two representative models:
#   * a 0-parameter pure expression (throughput of the evaluator loop)
#   * a 3-parameter model (gradient / Jacobian path)
for nrow in (1024, 4096, 16384), T in [Float32, Float64]
    p = Array{T}(undef, 0)
    X = rand(T, nrow, 2) .+ one(T)
    grad = Array{T}(undef, length(p))

    # 0-parameter model: f = x1*x1 + x2*x2
    m0 = Model(T, Expr(:->, :(x, p), :(x[1]*x[1] + x[2]*x[2])), 0)
    codevec = code(m0)
    vec = Array{T}(undef, nrow)
    jac = Array{T}(undef, nrow, length(p))

    SUITE["interpret"]["x²+y² - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_vec!($vec, $codevec, $X, $p) end
    SUITE["jacobian"]["x²+y² - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_jac!($vec, $jac, nothing, $codevec, $X, $p) end

    # 3-parameter model: f = p1*x1 + p2*x2 + p3
    m3 = Model(T, Expr(:->, :(x, p), :(p[1]*x[1] + p[2]*x[2] + p[3])), 3)
    p = ones(T, 3)
    code3 = code(m3)
    jac = Array{T}(undef, nrow, length(p))
    grad = Array{T}(undef, length(p))

    SUITE["interpret"]["3-param - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_vec!($vec, $code3, $X, $p) end
    SUITE["gradient"]["3-param - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_grad!($grad, $code3, $X, $p) end
    SUITE["jacobian"]["3-param - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_jac!($vec, $jac, nothing, $code3, $X, $p) end

    # 5-parameter model with the transcendental kernels that dominate the cost
    # of typical GP models: f = p1*|x1|^p2 + p3*log(x2) + p4*exp(p5*x1)
    # (POWABS, LOG, EXP).  The exponent p2 is not an integer, so `^` cannot
    # take Base's power-by-squaring shortcut.
    mt = Model(T, Expr(:->, :(x, p), :(p[1] * abs(x[1])^p[2] + p[3] * log(x[2]) + p[4] * exp(p[5] * x[1]))), 5)
    p = T[1.5, 1.7, 0.8, 0.3, -0.6]
    codet = code(mt)
    jac = Array{T}(undef, nrow, length(p))
    grad = Array{T}(undef, length(p))

    SUITE["interpret"]["pow-log-exp - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_vec!($vec, $codet, $X, $p) end
    SUITE["gradient"]["pow-log-exp - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_grad!($grad, $codet, $X, $p) end
    SUITE["jacobian"]["pow-log-exp - $T - $nrow"] = @benchmarkable with_fit_buffer_cache() do; SymRegInterpreter.interpret_jac!($vec, $jac, nothing, $codet, $X, $p) end
end

SUITE
# Precompilation workload: fitting and evaluating models under the Gaussian
# likelihood with profiled noise, in Float32 and Float64, so that a model
# search starts without compiling the fit (GeodesicLM) and the information
# matrix first.  Other likelihoods still compile on first use.

# p1 * x1 + p2 * x2 + ... + pk * xk, the variable x1 for k = 0
function _precompile_model(::Type{T}, k::Int, nvar::Int) where {T}
    terms = Any[:(p[$i] * x[$(mod1(i, nvar))]) for i in 1:k]
    body = k == 0 ? :(x[1]) : foldl((a, b) -> :($a + $b), terms)
    m = Model(T, Expr(:->, :(x, p), body), k)
    CodeModel(copy(code(m)), k)
end

@setup_workload begin
    n, nvar = 32, 3
    @compile_workload begin
        for T in (Float32, Float64)
            X = T[sin(i + j) for i in 1:n, j in 1:nvar]
            y = T(2) .* X[:, 1] .- X[:, 2] .* X[:, 3]
            l = GaussianProfiledLikelihood(X, y)
            with_fit_buffer_cache() do
                # The information matrix uses a ForwardDiff chunk of
                # min(#parameters, MAX_FORWARD_DIFF_CHUNK), and every chunk size
                # is its own specialization: cover them all.
                for k in 0:MAX_FORWARD_DIFF_CHUNK
                    model = _precompile_model(T, k, nvar)
                    θ = ones(T, n_param(l, model))
                    r = optimize(l, model, θ)
                    θ = T.(r.x)
                    evaluate_nll(l, model, θ)
                    information_matrix(l, model, θ)
                end
            end
        end
    end
end

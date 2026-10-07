# Precompilation workload: every model-selection criterion for models under the
# Gaussian likelihood with profiled noise, in Float32 and Float64.  Other
# likelihoods still compile on first use.

@setup_workload begin
    n, nvar = 32, 3
    @compile_workload begin
        for T in (Float32, Float64)
            X = T[sin(i + j) for i in 1:n, j in 1:nvar]
            y = T(2) .* X[:, 1] .- X[:, 2] .* X[:, 3]
            l = GaussianProfiledLikelihood(X, y)
            with_fit_buffer_cache() do
                # one model per ForwardDiff chunk size of the information matrix
                for k in 0:SymRegInterpreter.MAX_FORWARD_DIFF_CHUNK
                    model = SymRegLikelihoods._precompile_model(T, k, nvar)
                    θ = T.(SymRegLikelihoods.optimize(l, model, ones(T, n_param(l, model))).x)
                    for paramCompFunc in (param_complexity, param_complexity_rot,
                                          param_complexity_rot_scaled, param_complexity_det)
                        description_length_terms(l, model, θ; paramCompFunc)
                        description_length_complexity_penalty(l, model, θ; paramCompFunc)
                    end
                    AIC(l, model, θ); BIC(l, model, θ)
                    BIC_funccompl(l, model, θ); BIC_funccompl_penalty(l, model, θ)
                    fractional_bayes_factor(l, model, θ)
                    fractional_bayes_factor_complexity_penalty(l, model, θ)
                    func_complexity(model)
                end
            end
        end
    end
end

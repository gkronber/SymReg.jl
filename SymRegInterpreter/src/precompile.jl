# Precompilation workload: the interpreter's entry points for a `CodeModel` in
# Float32 and Float64, so that packages built on the interpreter start without
# compiling it first.

@setup_workload begin
    body = :(p[1] * x[1] + exp(p[2] * x[2]) / (p[3] + x[1] * x[1]))
    @compile_workload begin
        for T in (Float32, Float64)
            X = T[0.5 1.0; 1.5 -0.5; -1.0 2.0]
            m = Model(T, Expr(:->, :(x, p), body), 3)
            cm = CodeModel(copy(code(m)), n_param(m))
            p = T[1.0, 0.5, 2.0]
            for model in (m, cm)
                interpret_vec(model, X, p)
                interpret_grad!(similar(p), model, X, p)
                interpret_jac(model, X, p)
                with_fit_buffer_cache() do
                    interpret_vec(model, X, p)
                    interpret_grad!(similar(p), model, X, p)
                    interpret_jac(model, X, p)
                end
            end
        end
    end
end

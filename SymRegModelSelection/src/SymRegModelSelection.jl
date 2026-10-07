"""
    SymRegModelSelection

Model-selection criteria for symbolic regression models fitted with
`SymRegLikelihoods`: the [`description_length`](@ref), the
[`fractional_bayes_factor`](@ref), [`AIC`](@ref), [`BIC`](@ref) and
[`BIC_funccompl`](@ref), the function and parameter complexities they are built
from, and [`rational_constant_snap`](@ref).
"""
module SymRegModelSelection

# Model-selection criteria (description length, fractional Bayes factor, AIC,
# BIC) shared by the symbolic-regression hosts, operating on the unified
# `(likelihood, model, p)` interface of `SymRegLikelihoods` /
# `SymRegInterpreter`.
#
# Conventions:
#   * `p` is the natural parameter vector `[model params; likelihood params]`
#     of `SymRegLikelihoods` (length `n_param(l, model)`), with variances such
#     as an estimated Gaussian `sigma²` among the likelihood parameters.
#     The parameter complexity is charged on `p` itself and every
#     parameter-count term (FBF, AIC, BIC) uses its length.
#   * Function complexity exists in two representations: the `Expr` AST path
#     (`Model`/`core_expr`) and the bytecode path (`Vector{<:Instruction}`).
#     Both count non-parameter terminals (variables, scaled variables,
#     opcodes) as symbols in a `N * log(#symbols)` code length, integer
#     constants and rational numerators as `log(|c| + constant_offset)` and
#     rational denominators as `log(d)` per occurrence (parameters are
#     accounted for by the parameter complexity). Only integer and rational
#     constants are allowed; a rational is `DIV(CONSTANT n, CONSTANT d)` in the
#     bytecode. Remaining difference: the bytecode path counts shared
#     subexpressions (DAG) once while the Expr path counts occurrences.
#   * The information matrix lives in `SymRegLikelihoods`.
#   * Rational constant snapping tries to replace model parameters by
#     rational constants if it improves the description length.

using PrecompileTools: @setup_workload, @compile_workload
using SymRegInterpreter
using SymRegInterpreter: Model, ExprModel, CodeModel, Instruction, code, expr, core_expr,
    replace_param_with_const, with_fit_buffer_cache, CONSTANT, DIV, PARAM, POWCONST, SCALEDVAR, VAR
import SymRegInterpreter: n_param
using SymRegLikelihoods
using SymRegLikelihoods: AbstractLikelihood, information_matrix, n_likelihood_params,
    n_observations, evaluate_nll, optimize
using LinearAlgebra

export description_length, fractional_bayes_factor, AIC, BIC, BIC_funccompl,
    func_complexity, param_complexity, rational_constant_snap

public description_length_terms, description_length_complexity_penalty,
    fractional_bayes_factor_terms, fractional_bayes_factor_complexity_penalty,
    BIC_funccompl_penalty, param_complexity_det, param_complexity_rot,
    param_complexity_rot_scaled, find_best_snap, rational_approx

include("funccomplexity.jl")
include("symeigen.jl")
include("modelselection.jl")
include("rational_constant_snap.jl")

include("precompile.jl")

end

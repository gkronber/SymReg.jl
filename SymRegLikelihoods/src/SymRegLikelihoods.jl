"""
    SymRegLikelihoods

Likelihoods for fitting symbolic regression models of `SymRegInterpreter`: each
[`AbstractLikelihood`](@ref) holds the data and evaluates the negative
log-likelihood of a model with [`evaluate_nll`](@ref); [`optimize`](@ref) fits the
parameters, and [`information_matrix`](@ref), [`covariance`](@ref),
[`confidence_intervals`](@ref) and [`prediction_intervals`](@ref) quantify their
uncertainty.
"""
module SymRegLikelihoods

using PrecompileTools: @setup_workload, @compile_workload
using SymRegInterpreter
using SymRegInterpreter: Model, Instruction, code, interpret_vec, interpret_vec!, interpret_jac!,
    interpret_vec2!, interpret_vecmat!, interpret_grad!, interpret_grad_seeded!, differentiate,
    with_fit_buffer_cache, EltypeMismatchError, MAX_FORWARD_DIFF_CHUNK, get_buffer,
    get_cached_object, get_matrix_cache_view, get_matrix_diffcache, get_vector_cache,
    get_vector_cache_view, get_vector_diffcache, get_vector_scratch_view
import SymRegInterpreter: n_param
import NaNMath: NaNMath
using ForwardDiff
using LinearAlgebra
using PreallocationTools: get_tmp, DiffCache

export AbstractLikelihood,
    GaussianLikelihood, GaussianProfiledLikelihood, GaussianScatterLikelihood,
    LaplaceLikelihood, LaplaceProfiledLikelihood, LaplaceScatterLikelihood,
    XUniformDiagonalLikelihood, MNRDiagonalLikelihood, MNRDenseLikelihood, MNRBandedLikelihood,
    MNRGeneralBandedLikelihood, XUniformBandedLikelihood, BandedCovariance,
    XProfileDenseLikelihood, CosmicChronometerLikelihood,
    n_param, n_likelihood_params, n_observations, evaluate_nll, evaluate_nll_grad!,
    optimize, OptResult, NLoptOptimizer, LevenbergMarquardtOptimizer,
    information_matrix, covariance, confidence_intervals, prediction_intervals

public AbstractLossFunction, AbstractProfiledLikelihood, AbstractLeastSquaresLikelihood,
    PreparedLikelihood, prepare, parametertype, positive_params, n_opt_params,
    evaluate_nll_grad_fd!, evaluate_profiled_nll, evaluate_profiled_nll_grad!, profile!,
    has_intrinsic_scatter, intrinsic_scatter,
    residual!, residual_jacobian!, residual_and_jacobian!, least_squares_complete!,
    AbstractOptimizer, optimizer_parameters, natural_parameters,
    observation_scatter, pairwise_confidence_regions_laplace

include("util.jl")
include("interfaces.jl")
include("gaussian.jl")
include("laplace.jl")
include("xuniform_diagonal.jl")
include("mnr_diagonal.jl")
include("cosmicchronometer.jl")
include("mnr_dense.jl")
include("mnr_banded.jl")
include("banded_covariance.jl")
include("mnr_general_banded.jl")
include("xuniform_banded.jl")
include("xprofile_dense.jl")
include("structuredcache.jl")
include("optim.jl")
include("statistics.jl")

include("precompile.jl")

end

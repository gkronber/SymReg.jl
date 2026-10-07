# Uniform-prior marginal likelihood ("unif", arXiv:2309.00948, Sec. 2.2) for a
# single input with independent x and y errors (diagonal error covariance).  The
# true input is marginalized under an infinitely wide uniform prior, which to
# first order in the model propagates the x error through the slope:
#   s2  = yerr² + (∂f/∂x)² xerr² [+ σ_int²]
#   nll = 1/2 Σ ( (f(x) - y)² / s2 + log(2π s2) )
# with f and ∂f/∂x evaluated at the observed x.  Up to terms that depend on
# neither the model nor σ_int², it is the w² → ∞ limit of `MNRDiagonalLikelihood`
# with d == 1, and it equals ROXY's `unif` method.
#
# This is the likelihood used for the radial acceleration relation (RAR) in
# "On the functional form of the radial acceleration relation",
# arXiv:2301.04368, with x = log10(gbar) and y = log10(gobs).
#
# With `intrinsic_scatter = true`, σ_int² is the single likelihood parameter.

"""
    XUniformDiagonalLikelihood(x, xerr, y, yerr; intrinsic_scatter = false)

Gaussian likelihood for a single input with independent errors `xerr` and `yerr`
(standard deviations), which marginalizes the true input under a uniform prior
("unif" of arXiv:2309.00948): the x error enters through the slope of the model,
with the variance `yerr² + (∂f/∂x)² xerr²` per observation.  With
`intrinsic_scatter = true` the intrinsic scatter `σ_int²` is added to the
variance as the single likelihood parameter.
"""
struct XUniformDiagonalLikelihood{T} <: AbstractLikelihood{T}
    x::Vector{T}
    xerr2::Vector{T}
    y::Vector{T}
    yerr2::Vector{T}
    intrinsic_scatter::Bool

    function XUniformDiagonalLikelihood(x::AbstractVector{T}, xerr::AbstractVector{T},
            y::AbstractVector{T}, yerr::AbstractVector{T};
            intrinsic_scatter::Bool = false) where {T}
        new{T}(x, xerr.^2, y, yerr.^2, intrinsic_scatter)
    end
end

n_observations(l::XUniformDiagonalLikelihood) = length(l.x)
has_intrinsic_scatter(l::XUniformDiagonalLikelihood) = l.intrinsic_scatter
n_likelihood_params(l::XUniformDiagonalLikelihood) = Int(l.intrinsic_scatter)
positive_params(l::XUniformDiagonalLikelihood) = 1:Int(l.intrinsic_scatter)

struct PreparedXUniformDiagonal{T} <: PreparedLikelihood{T}
    likelihood::XUniformDiagonalLikelihood{T}
    model::Model{T}
    dcode::Vector{Instruction{T}}  # differentiate(model.code, 1); f at fidx, df/dx at dfidx
    fidx::Int
    dfidx::Int
    x::Matrix{T}                   # model input column
end

function prepare(l::XUniformDiagonalLikelihood{T}, model::Model{T}) where {T}
    dcode, fidx, dfidx = differentiate(code(model), 1)
    PreparedXUniformDiagonal(l, model, dcode, fidx, dfidx, reshape(l.x, :, 1))
end

# Uses the same differentiated code as the gradient, so that both agree.
function evaluate_nll(pl::PreparedXUniformDiagonal{T}, p::AbstractVector{TD}) where {T,TD}
    _check_nll_dims(pl.likelihood, pl.model, p)
    l = pl.likelihood
    n = length(l.x)
    f = get_buffer(get_vector_diffcache(T, :xunif_f, n), TD)
    dfdx = get_buffer(get_vector_diffcache(T, :xunif_g, n), TD)
    interpret_vec2!(f, dfdx, pl.dcode, pl.x, p, pl.fidx, pl.dfidx)
    s2int = _scatter2(l, p, n_param(pl.model))
    s2int >= 0 || return _worst_loss(TD, T)
    nll = zero(TD)
    for i in 1:n
        s2 = l.yerr2[i] + dfdx[i]^2 * l.xerr2[i] + s2int
        nll += TD(1/2) * (((f[i] - l.y[i])^2) / s2 + log(2 * TD(pi) * s2))
    end
    isnan(nll) || isinf(nll) ? _worst_loss(TD, T) : nll
end

# The adjoints λf = r/s2 and λg = (1 - r²/s2) g xerr2/s2 seed the reverse sweep
# at `fidx` and `dfidx`; dNLL/dσ_int² = ½ Σ (1 - r²/s2) / s2.
function evaluate_nll_grad!(grad::AbstractVector{TD}, pl::PreparedXUniformDiagonal{T},
        p::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, pl.likelihood, pl.model, p)
    l = pl.likelihood
    n = length(l.x)
    f = get_buffer(get_vector_diffcache(T, :xunif_f, n), TD)
    g = get_buffer(get_vector_diffcache(T, :xunif_g, n), TD)
    λf = get_buffer(get_vector_diffcache(T, :xunif_lf, n), TD)
    λg = get_buffer(get_vector_diffcache(T, :xunif_lg, n), TD)
    interpret_vec2!(f, g, pl.dcode, pl.x, p, pl.fidx, pl.dfidx)
    nmod = n_param(pl.model)
    s2int = _scatter2(l, p, nmod)

    fill!(grad, zero(TD))
    s2int >= 0 || return _worst_loss(TD, T)
    nll = zero(TD)
    dls = zero(TD)
    for i in 1:n
        r = f[i] - l.y[i]
        s2 = l.yerr2[i] + g[i]^2 * l.xerr2[i] + s2int
        invs2 = inv(s2)
        nll += T(1/2) * (r^2 * invs2 + log(2 * T(pi) * s2))
        λf[i] = r * invs2
        # dNLL_i/ds2 = (1 - r^2/s2) / (2 s2); ds2/dg = 2 g xerr2
        ds2 = (one(T) - r * r * invs2) * invs2
        λg[i] = ds2 * g[i] * l.xerr2[i]
        dls += ds2
    end
    if isnan(nll) || isinf(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    l.intrinsic_scatter && (grad[nmod + 1] = dls / 2)
    interpret_grad_seeded!(grad, pl.dcode, pl.x, p, (pl.fidx => λf, pl.dfidx => λg))
    nll
end

# Differentiate the model on every call; use `prepare` in hot loops.
function evaluate_nll(l::XUniformDiagonalLikelihood{T}, model::Model, p::AbstractVector{TD}) where {T,TD}
    _check_nll_dims(l, model, p)
    evaluate_nll(prepare(l, model), p)
end

function evaluate_nll_grad!(grad::AbstractVector{TD}, l::XUniformDiagonalLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, p)
    evaluate_nll_grad!(grad, prepare(l, model), p)
end

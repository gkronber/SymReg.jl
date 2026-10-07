# Cosmic-chronometer likelihood.  The model value is H², compared as
#   nll = 1/2 Σ ( (sqrt(f) - age) / sig )²
# without the constant 1/2 log(2π sig²), as in ESR.
#
# With `intrinsic_scatter = true`, s² = sig² + σ_int² and σ_int² is the single
# likelihood parameter.  The normalization is kept relative to the omitted one,
#   nll = 1/2 Σ [ (sqrt(f) - age)² / s² + log(s² / sig²) ],
# so that the NLL is continuous at σ_int = 0.

"""
    CosmicChronometerLikelihood(x, age, sig; intrinsic_scatter = false)

Likelihood of cosmic-chronometer data as in ESR: the model value is `H²`, and
`sqrt(f(x))` is compared with the observations `age` with the uncertainties
`sig`, without the constant normalization of the Gaussian.  With
`intrinsic_scatter = true` the intrinsic scatter `σ_int²` is added to `sig²` as
the single likelihood parameter.
"""
struct CosmicChronometerLikelihood{T} <: AbstractLikelihood{T}
    x::Vector{T}     # model variable
    age::Vector{T}   # observed age
    sig::Vector{T}   # per-observation uncertainty of the age
    intrinsic_scatter::Bool
end
function CosmicChronometerLikelihood(x::AbstractVector{T}, age::AbstractVector{T},
        sig::AbstractVector{T}; intrinsic_scatter::Bool = false) where {T}
    n = length(x)
    (length(age) == n && length(sig) == n) ||
        throw(DimensionMismatch("cosmic-chronometer data vectors must have equal length"))
    CosmicChronometerLikelihood{T}(Vector{T}(x), Vector{T}(age), Vector{T}(sig), intrinsic_scatter)
end

n_observations(l::CosmicChronometerLikelihood) = length(l.x)
has_intrinsic_scatter(l::CosmicChronometerLikelihood) = l.intrinsic_scatter
n_likelihood_params(l::CosmicChronometerLikelihood) = Int(l.intrinsic_scatter)
positive_params(l::CosmicChronometerLikelihood) = 1:Int(l.intrinsic_scatter)

function evaluate_nll(l::CosmicChronometerLikelihood{T}, model::Model, p::AbstractVector) where {T}
    _check_nll_dims(l, model, p)
    n = length(l.x)
    TD = eltype(p)
    v = get_buffer(get_vector_diffcache(T, :cosmic_pred, n), TD)
    interpret_vec!(v, code(model), reshape(l.x, :, 1), p)
    nll = zero(TD)
    if l.intrinsic_scatter
        s2int = _scatter2(l, p, n_param(model))
        s2int >= 0 || return _worst_loss(TD, T)
        for i in 1:n
            H = NaNMath.sqrt(v[i])
            sig2 = l.sig[i]^2
            s2 = sig2 + s2int
            nll += T(1/2) * ((H - l.age[i])^2 / s2 + log(s2 / sig2))
        end
    else
        for i in 1:n
            H = NaNMath.sqrt(v[i])
            nll += T(1/2) * (((H - l.age[i]) / l.sig[i])^2)
        end
    end
    isnan(nll) || isinf(nll) ? _worst_loss(TD, T) : nll
end

#   ∂nll/∂v_i   = (H_i - age_i) / (2 s_i² H_i),   H_i = sqrt(v_i)
#   ∂nll/∂σ_int² = 1/2 Σ (1 - (H_i - age_i)²/s_i²) / s_i²
function evaluate_nll_grad!(grad::AbstractVector{TD}, l::CosmicChronometerLikelihood{T},
        model::Model, p::AbstractVector{TD}) where {T,TD}
    _check_grad_dims(grad, l, model, p)
    n = length(l.x)
    nmod = n_param(model)
    xmat = reshape(l.x, :, 1)
    v = get_buffer(get_vector_diffcache(T, :cosmic_pred, n), TD)
    interpret_vec!(v, code(model), xmat, p)
    λ = get_buffer(get_vector_diffcache(T, :cosmic_lam, n), TD)
    fill!(grad, zero(TD))
    total = zero(TD)
    if l.intrinsic_scatter
        s2int = _scatter2(l, p, nmod)
        s2int >= 0 || return _worst_loss(TD, T)
        dls = zero(TD)
        for i in 1:n
            H = NaNMath.sqrt(v[i])
            e = H - l.age[i]
            sig2 = l.sig[i]^2
            s2 = sig2 + s2int
            invs2 = inv(s2)
            total += T(1/2) * (e * e * invs2 + log(s2 / sig2))
            λ[i] = e * invs2 / (2 * H)
            dls += (one(T) - e * e * invs2) * invs2
        end
        grad[nmod + 1] = dls / 2
    else
        for i in 1:n
            H = NaNMath.sqrt(v[i])
            r = (H - l.age[i]) / l.sig[i]
            total += T(1/2) * (r * r)
            λ[i] = r / (2 * l.sig[i] * H)
        end
    end
    if isnan(total) || isinf(total)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    interpret_grad_seeded!(grad, code(model), xmat, p, (length(code(model)) => λ,))
    total
end



# NaN or infinite NLLs are clamped to `floatmax(T)`, converted to the evaluation
# type `TD` so that it stays a Dual under ForwardDiff.
_worst_loss(::Type{TD}, ::Type{T}) where {TD,T} = TD(floatmax(T))

@inline _finite_loss(nll::TD, ::Type{T}) where {TD,T} = isfinite(nll) ? nll : _worst_loss(TD, T)

# Add the model-parameter entries to `grad` by a reverse sweep seeded with `λ`,
# the adjoints of the NLL w.r.t. the predictions.
function _prediction_sweep!(grad::AbstractVector{TD}, model::Model, X, θ, λ, nll,
        ::Type{T}) where {TD,T}
    if !isfinite(nll)
        fill!(grad, zero(TD))
        return _worst_loss(TD, T)
    end
    interpret_grad_seeded!(grad, code(model), X, θ, (length(code(model)) => λ,))
    nll
end

# In-place LU of `M`, or `nothing` when `M` is non-finite or singular.
function _try_lu!(M::AbstractMatrix)
    all(isfinite, M) || return nothing
    F = lu!(M; check = false)
    issuccess(F) ? F : nothing
end

# Reverse sweep seeded with `cols[1] => λf` and `cols[k + 1] => λg[:, k]`.
# Small column counts go through `Val` so that the seed tuple has a static length.
@inline function _grad_seeded!(grad, code, X, p, cols, λf, λg)
    d = size(λg, 2)
    d == 1 && return interpret_grad_seeded!(grad, code, X, p, _seed_pairs(cols, λf, λg, Val(1)))
    d == 2 && return interpret_grad_seeded!(grad, code, X, p, _seed_pairs(cols, λf, λg, Val(2)))
    d == 3 && return interpret_grad_seeded!(grad, code, X, p, _seed_pairs(cols, λf, λg, Val(3)))
    interpret_grad_seeded!(grad, code, X, p,
        (cols[1] => λf, (cols[k + 1] => view(λg, :, k) for k in 1:d)...))
end

@inline _seed_pairs(cols, λf, λg, ::Val{D}) where {D} =
    (cols[1] => λf, ntuple(k -> cols[k + 1] => view(λg, :, k), Val(D))...)

# Smallest noise scale (standard deviation for the Gaussian, scale b for the
# Laplace likelihood) that a profiled likelihood estimates.
# At an exact fit (RMS=0) the estimate would make the profile NLL -Inf.
function _min_scale(y::AbstractVector{T}) where {T}
    rms = isempty(y) ? zero(T) : norm(y) / sqrt(T(length(y)))
    max(eps(T) * rms, sqrt(floatmin(T)))
end

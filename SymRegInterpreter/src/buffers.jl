# A cache used while fitting a sequence of expressions so the hot path stays allocation-free.
# `workspaces` holds the grow-to-fit DiffCache workspaces used by the interpreter,
# resolved by a cheap linear scan.
#
# This is a `mutable` struct even though field is ever reassigned
# (`reset_fit_buffer_cache!` mutates the `workspaces` vector, not the struct).
# `with_fit_buffer_cache` stores the cache in task-local storage, an
# `IdDict{Any,Any}`: a mutable struct is already a heap reference and stores for
# free.
mutable struct FitBufferCache
    workspaces::Vector{Any}
end

FitBufferCache() = FitBufferCache(Any[])

# One cache per task, handed out for as long as the task lives.  See
# `with_fit_buffer_cache` for why the lifetime is the task and not the block.
const _fit_buffer_caches = Base.OncePerTask{FitBufferCache}(FitBufferCache)
const _fit_buffer_cache_key = :SymRegInterpreter_fit_buffer_cache

"""
    with_fit_buffer_cache(f)

Run `f` with the interpreter's workspace cache active and return its result.
Outside such a block there is no cache at all: `_resolve_workspace` hands back a
throwaway `Workspace` and every buffer request allocates.  The block is what
switches the caching on; it is not what creates the storage.

## Lifetime: the task, not the block

`Base.OncePerTask` creates one `FitBufferCache` per task and then keeps handing
out that same object.  `with_fit_buffer_cache` only installs it in task-local
storage for the duration of `f`, so **two sequential blocks in the same task
share their workspaces**, and a buffer grown inside the first is still there, at
its grown size, inside the second.  Only a new task starts from nothing.

This is deliberate.  A model search fits millions of expressions, and a cache
that started empty on every block would allocate every tape and every scratch
vector again for each one; making the cache survive the block is what keeps
fitting allocation-free.

## Growth: monotone, with no automatic release

Workspaces requested through the `*_scratch_*` entry points grow to fit and
never shrink — `_workspace_get_diffcache` keeps every dimension that is large
enough and at least doubles one that is too small (`_shape_grow`), because a
request that is larger in one dimension and smaller in another must not make the
two shapes alternate and reallocate on every call, and a slowly growing request
must not reallocate on every step.

The consequence is: within one task the cache converges on
the largest shape any model has ever needed and holds that memory until the
task ends.  That is the intended trade.  Nothing releases it on its own;
[`reset_fit_buffer_cache!`](@ref) does, for a caller that wants to reset explicitly.
"""
function with_fit_buffer_cache(f::F) where {F<:Function}
    cache = _fit_buffer_caches()
    task_local_storage(f, _fit_buffer_cache_key, cache)
end

"""
    reset_fit_buffer_cache!() -> FitBufferCache

Drop every workspace held by the current task's fit buffer cache, releasing the
memory it has grown to, and return the now-empty cache.

The next request through any `get_*` entry point allocates again at exactly the
shape that request asks for, so a reset costs one round of allocation and
leaves a cache sized to whatever comes next rather than to the largest model
seen so far.  Nothing else changes: the cache object is the task's and stays in
place, so a `with_fit_buffer_cache` block that is already running keeps working,
and the call is equally valid outside any block — it then clears the cache a
later block would install.

A caller holding a buffer, or a view handed out
by `get_matrix_scratch_view` and friends, keeps a valid array — the old memory
is released only once nothing references it.

Nothing in this package needs this today; it exists so that a long-running host
can explicitly reset the cache.
"""
function reset_fit_buffer_cache!()
    cache = get(task_local_storage(), _fit_buffer_cache_key, nothing)::Union{Nothing,FitBufferCache}
    reset_fit_buffer_cache!(cache === nothing ? _fit_buffer_caches() : cache)
end

function reset_fit_buffer_cache!(cache::FitBufferCache)
    empty!(cache.workspaces)
    cache
end


# ---------------------------------------------------------------------------
# Interpreter workspace caches
# ---------------------------------------------------------------------------
# One growable DiffCache slot per (`desc`, eltype) family.  It remembers the
# single largest buffer seen so far together with its primal shape, and reuses
# that buffer when its existing shape already covers the request (internal,
# reusable workspaces) or matches it exactly (workspaces returned to the
# caller).
#
# The dual storage of a DiffCache is created empty (chunk size 0) and
# `get_tmp` enlarges it on the first request for ForwardDiff duals, to the
# chunk size the requested scalar type encodes.  Most workspaces are never used
# with duals (residuals, Jacobians, gradients), and an eagerly sized dual buffer
# is `(chunk + 1)^levels = 25` times the primal buffer.  Grow-to-fit workspaces
# grow geometrically (`_shape_grow`), so a sequence of slightly larger requests,
# e.g. ever longer programs in a GP run, reallocates only O(log) times.
mutable struct Workspace
    desc::Symbol
    T::Type
    shape::Any # current primal shape (tuple), or `nothing`
    cache::Any # DiffCache, or `nothing` before first use
end



# Cap of the ForwardDiff chunk size used with these workspaces (see
# `forward_diff_chunk` in SymRegLikelihoods).  ForwardDiff's default chunk grows
# with the parameter-vector length, and each distinct chunk size produces a
# distinct `Dual{...,N}` type, a larger dual buffer and new compiled
# specializations; the parameter count varies between models.  The dual
# buffers are sized on demand by `get_tmp`, for whatever chunk is requested.
const MAX_FORWARD_DIFF_CHUNK = 4

# Internal (reusable, grow-to-fit) diffcaches
get_matrix_scratch_diffcache(::Type{T}, desc::Symbol, nrows::Int, ncols::Int) where {T} = _workspace_get_diffcache(true, T, desc, (nrows, ncols))::DiffCache{Matrix{T},Vector{T}}
get_vector_scratch_diffcache(::Type{T}, desc::Symbol, n::Int) where {T} = _workspace_get_diffcache(true, T, desc, (n,))::DiffCache{Vector{T},Vector{T}}

# Grow-to-fit workspaces that hand the *caller* a buffer of exactly the
# requested shape, as a contiguous view into a larger shared buffer.
# Same scratch contract as everywhere else: the caller must fully overwrite
# the returned view before reading it (the memory behind it is shared with
# earlier, differently-shaped uses).
function get_matrix_scratch_view(::Type{T}, desc::Symbol, nrows::Int, ncols::Int,
                                 ::Type{S}) where {T,S<:Number}
    dc = _workspace_get_diffcache(true, T, desc, (nrows, ncols))::DiffCache{Matrix{T},Vector{T}}
    view(get_buffer(dc, S), 1:nrows, 1:ncols)
end

function get_vector_scratch_view(::Type{T}, desc::Symbol, n::Int, ::Type{S}) where {T,S<:Number}
    dc = _workspace_get_diffcache(true, T, desc, (n,))::DiffCache{Vector{T},Vector{T}}
    view(get_buffer(dc, S), 1:n)
end

# Plain (non-dual) variants of `get_matrix_scratch_view` / `get_vector_scratch_view`.
function get_matrix_cache_view(::Type{T}, desc::Symbol, nrows::Int, ncols::Int) where {T}
    buf = _workspace_get_cache(true, T, desc, (nrows, ncols))::Matrix{T}
    view(buf, 1:nrows, 1:ncols)
end

function get_vector_cache_view(::Type{T}, desc::Symbol, n::Int) where {T}
    buf = _workspace_get_cache(true, T, desc, (n,))::Vector{T}
    view(buf, 1:n)
end

# Task-local slot for a reusable non-array object, e.g. an optimizer workspace
# that owns its own scratch.  Resolved by (`desc`, `T`) like the array
# workspaces above; `init` runs only on the first request within the active
# `with_fit_buffer_cache` scope, and on every call when no cache is active.
function get_cached_object(init::F, desc::Symbol, ::Type{T}) where {F<:Function,T}
    ws = _resolve_workspace(desc, T)
    obj = ws.cache
    obj === nothing || return obj
    obj = init()
    ws.cache = obj
    obj
end

# The exact-size variants below keep a single slot per (`desc`, eltype), so a
# request whose shape differs from the previous one throws the cached buffer
# away and allocates a new one. 
#
# Exact-size diffcaches
get_matrix_diffcache(::Type{T}, desc::Symbol, nrows::Int, ncols::Int) where {T} = _workspace_get_diffcache(false, T, desc, (nrows, ncols))::DiffCache{Matrix{T},Vector{T}}
get_vector_diffcache(::Type{T}, desc::Symbol, n::Int) where {T} = _workspace_get_diffcache(false, T, desc, (n,))::DiffCache{Vector{T},Vector{T}}

# Exact-size buffers
get_matrix_cache(::Type{T}, desc::Symbol, nrows::Int, ncols::Int) where {T} = _workspace_get_cache(false, T, desc, (nrows, ncols))::Matrix{T}
get_vector_cache(::Type{T}, desc::Symbol, n::Int) where {T} = _workspace_get_cache(false, T, desc, (n,))::Vector{T}



"""
    get_buffer(cache::DiffCache, S) -> AbstractArray{S}

The buffer of `cache` for values of scalar type `S`: the primal buffer when `S`
is the cache's element type `T`, or the (transparently resized) dual buffer for
ForwardDiff duals over `T`, also nested ones.  Any other `S` would be stored at
the precision of `T` and throws an [`EltypeMismatchError`](@ref).

Unlike `PreallocationTools.get_tmp`, which falls back to the primal buffer for
any type that promotes to `T`, the element type must match exactly.  The checks
are on types only and fold away at compile time.
"""
function get_buffer(cache::DiffCache{<:AbstractArray{T}}, ::Type{S}) where {T,S<:Number}
    _primal_type(S) === T || _throw_eltype_mismatch(T, S)
    S === T ? cache.du : get_tmp(cache, S)
end

_primal_type(::Type{S}) where {S} = S
_primal_type(::Type{<:ForwardDiff.Dual{<:Any,V}}) where {V} = _primal_type(V)

"""
    EltypeMismatchError(buffer, value)

Values of type `value` were evaluated with workspace buffers of element type
`buffer`.  The parameters must have the model's element type, or be ForwardDiff
duals over it.
"""
struct EltypeMismatchError <: Exception
    buffer::Type
    value::Type
end

Base.showerror(io::IO, e::EltypeMismatchError) =
    print(io, "EltypeMismatchError: values of type ", e.value,
          " cannot use a buffer with element type ", e.buffer, "; convert the parameters to ",
          e.buffer, " or ForwardDiff duals over ", e.buffer)

@noinline _throw_eltype_mismatch(T, S) = throw(EltypeMismatchError(T, S))

# Reuse the cached DiffCache whose shape already fits `shape` (grow=true) or
# matches it exactly (grow=false); otherwise allocate exactly `shape` and
# remember it.
function _workspace_get_diffcache(grow::Bool, ::Type{T}, desc::Symbol, shape::NTuple{N,Int})::DiffCache where {T,N}
    ws = _resolve_workspace(desc, T)
    alloc = shape
    if ws.cache !== nothing
        s = ws.shape::NTuple{N,Int}
        fits = grow ? _shape_geq(s, shape) : (s == shape)
        fits && return ws.cache
        # Grow-to-fit must be monotone: a request that is larger in one
        # dimension and smaller in another must not shrink the buffer, or the
        # two shapes alternate and every call reallocates.
        grow && (alloc = _shape_grow(s, shape))
    end
    # chunk size 0: no dual storage until `get_tmp` is asked for duals
    dc = DiffCache(zeros(T, alloc...), 0; warn_on_resize=false)
    ws.shape = alloc
    ws.cache = dc
    dc
end

# plain vector/matrix version of above
function _workspace_get_cache(grow::Bool, ::Type{T}, desc::Symbol, shape::NTuple{N,Int})::Array{T,N} where {T,N}
    ws = _resolve_workspace(desc, T)
    alloc = shape
    if ws.cache !== nothing
        s = ws.shape::NTuple{N,Int}
        fits = grow ? _shape_geq(s, shape) : (s == shape)
        fits && return ws.cache
        grow && (alloc = _shape_grow(s, shape))
    end
    buf = zeros(T, alloc...)
    ws.shape = alloc
    ws.cache = buf
    buf
end

function _resolve_workspace(desc::Symbol, ::Type{T})::Workspace where {T}
    cache = get(task_local_storage(), _fit_buffer_cache_key, nothing)::Union{Nothing,FitBufferCache}
    _resolve_workspace(cache, desc, T)
end

_resolve_workspace(::Nothing, desc::Symbol, ::Type{T}) where {T} = Workspace(desc, T, nothing, nothing)
function _resolve_workspace(cache::FitBufferCache, desc::Symbol, ::Type{T}) where {T}
    # linear scan
    for e in cache.workspaces
        ws = e::Workspace
        ws.desc === desc && ws.T === T && return ws
    end
    ws = Workspace(desc, T, nothing, nothing)
    push!(cache.workspaces, ws)
    ws
end

# does `avail` have at least as many cells as `need` in every dimension?
# Static `Val(N)` indexing keeps the small-tuple access allocation-free.
_shape_geq(avail::Tuple, need::Tuple) = _shape_geq(avail, need, Val(length(need)))
_shape_geq(avail::Tuple, need::Tuple, ::Val{1}) = length(avail) >= 1 && avail[1] >= need[1]
_shape_geq(avail::Tuple, need::Tuple, ::Val{2}) = length(avail) >= 2 && avail[1] >= need[1] && avail[2] >= need[2]

# New shape for a grow-to-fit workspace of shape `a` that must cover `b`: a
# dimension that is too small at least doubles, the others are kept.
_shape_grow(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} =
    ntuple(i -> b[i] > a[i] ? max(b[i], 2 * a[i]) : a[i], Val(N))

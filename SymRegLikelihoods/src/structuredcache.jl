# DiffCaches for structured matrices, one `DiffCache` per band.

using PreallocationTools: DiffCache, PreallocationTools
import PreallocationTools: get_tmp

abstract type AbstractDiffCache end

struct DiagonalDiffCache <: AbstractDiffCache
    diag::DiffCache
    function DiagonalDiffCache(m::Diagonal, n)
        new(DiffCache(m.diag, n))
    end
end
get_tmp(cache::DiagonalDiffCache, ::Type{T}) where {T} =
    Diagonal(get_tmp(cache.diag, T))

struct BidiagonalDiffCache <: AbstractDiffCache
    dv::DiffCache
    ev::DiffCache
    uplo::Char
    function BidiagonalDiffCache(m::Bidiagonal, n)
        new(DiffCache(m.dv, n), DiffCache(m.ev, n), m.uplo)
    end
end
get_tmp(cache::BidiagonalDiffCache, ::Type{T}) where {T} =
    Bidiagonal(get_tmp(cache.dv, T), get_tmp(cache.ev, T), cache.uplo)

struct SymTridiagonalDiffCache <: AbstractDiffCache
    dv::DiffCache
    ev::DiffCache
    function SymTridiagonalDiffCache(m::SymTridiagonal, n)
        new(DiffCache(m.dv, n), DiffCache(m.ev, n))
    end
end
get_tmp(cache::SymTridiagonalDiffCache, ::Type{T}) where {T} =
    SymTridiagonal(get_tmp(cache.dv, T), get_tmp(cache.ev, T))

struct TridiagonalDiffCache <: AbstractDiffCache
    dl::DiffCache
    d::DiffCache
    du::DiffCache
    function TridiagonalDiffCache(m::Tridiagonal, n)
        new(DiffCache(m.dl, n), DiffCache(m.d, n), DiffCache(m.du, n))
    end
end
get_tmp(cache::TridiagonalDiffCache, ::Type{T}) where {T} =
    Tridiagonal(get_tmp(cache.dl, T), get_tmp(cache.d, T), get_tmp(cache.du, T))

create_diffcache(a::AbstractArray, n_param) = DiffCache(a, n_param)
create_diffcache(a::Diagonal, n_param) = DiagonalDiffCache(a, n_param)
create_diffcache(a::Bidiagonal, n_param) = BidiagonalDiffCache(a, n_param)
create_diffcache(a::SymTridiagonal, n_param) = SymTridiagonalDiffCache(a, n_param)
create_diffcache(a::Tridiagonal, n_param) = TridiagonalDiffCache(a, n_param)

@Zygote.adjoint function Iterators.product(xs...)
    back(::AbstractArray{Nothing}) = nothing
    back(dy::NamedTuple{(:iterators,)}) = dy.iterators
    function back(dy::AbstractArray)
      d = 1
      ntuple(length(xs)) do n
        first(dy)[n] === nothing && return nothing
        nd = _ndims(xs[n])
        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
        d += nd
        init = zero.(first(dy)[n]) # allows for tuples, which accum can add:
        red = mapreduce(StaticGetter{n}(), accum, dy; dims=dims, init=init)
        return reshape(red, axes(xs[n]))
      end
    end
    Iterators.product(xs...), back
  end
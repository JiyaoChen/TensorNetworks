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


# improves performance compared to default implementation, also avoids errors with some complex arrays
@Zygote.adjoint function norm(A::AbstractArray, p::Real = 2)
  n = norm(A, p)
  back(Δ) = let n = n
              (Δ .* A ./ (n + eps(0f0)), )
          end
  return n, back
end

"""
		svd(A) -> Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}
"""
function svd(A)
	U, S, V = LinearAlgebra.svd(A);
	return U, S, Matrix(V)
end

function safe_inverse(x, epsilon = 1e-12)
    return x ./ (x.^2 .+ epsilon)
end

"""
    svd_back(U, S, V, dU, dS, dV) adjoint for SVD decomposition.
    References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V::AbstractArray, dU, dS, dV; η::Real = 1e-40) where T

    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η);
    NS = length(S);
    S2 = S.^2;
    Sinv = S ./ (S2 .+ η);
    F = S2' .- S2;
    F ./= (F.^2 .+ η);

    # res = ZeroAdder()
    res = zeros(length(S), length(S));
    if !(dU isa Nothing)
        UdU = U' * dU;
        J = F .* UdU;
        res += (J + J') * Diagonal(S) + 1im * Diagonal(imag(diag(UdU)) .* Sinv);
    end

    if !(dV isa Nothing)
        VdV = V' * dV;
        K = F .* VdV;
        res += Diagonal(S) * (K + K');
    end

    if !(dS isa Nothing)
        res += Diagonal(dS);
    end

    res = U * res * V';

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U * (U' * dU)) * Diagonal(Sinv) * V';
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * Diagonal(Sinv) * (dV' - (dV' * V) * V');
    end

    return res

end

@Zygote.adjoint function svd(A)
  U, S, V = svd(A)
  return (U, S, V), dy -> (svd_back(U, S, V, dy...),)
end
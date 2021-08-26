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

# mpow2(a::AbstractArray) = a .^ 2

# """
# 		svd(A) -> Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}
# """
# function svd(A)
# 	U, S, V = LinearAlgebra.svd(A);
# 	return U, S, Matrix(V)
# end


# """
#     svd_back(U, S, V, dU, dS, dV) adjoint for SVD decomposition.
#     References:
#     https://j-towns.github.io/papers/svd-derivative.pdf
#     https://giggleliu.github.io/2019/04/02/einsumbp.html
# """
# function svd_back(U::AbstractArray, S::AbstractArray{T}, V::AbstractArray, dU, dS, dV; η::Real = 1e-40) where T
    
#     all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
#     η = T(η);
#     NS = length(S);
#     S2 = mpow2(S);
#     Sinv = @. S/(S2+η);
#     F = S2' .- S2;
#     F ./= (mpow2(F) .+ η);

#     # res = ZeroAdder()
#     res = zero(U);
#     if !(dU isa Nothing)
#         UdU = U' * dU;
#         J = F .* UdU;
#         res += (J + J') * LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv);
#     end

#     if !(dV isa Nothing)
#         VdV = V' * dV;
#         K = F .* VdV;
#         res += LinearAlgebra.Diagonal(S) * (K + K');
#     end

#     if !(dS isa Nothing)
#         res += LinearAlgebra.Diagonal(dS);
#     end

#     res = U * res * V';

#     if !(dU isa Nothing) && size(U, 1) != size(U, 2)
#         res += (dU - U * (U' * dU)) * LinearAlgebra.Diagonal(Sinv) * V';
#     end

#     if !(dV isa Nothing) && size(V, 1) != size(V, 2)
#         res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV' * V) * V');
#     end

#     return res

# end

# @Zygote.adjoint function svd(A)
#   U, S, V = svd(A)
#   return (U, S, V), dy -> (svd_back(U, S, V, dy...),)
# end
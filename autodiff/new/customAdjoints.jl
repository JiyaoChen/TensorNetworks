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


# # improves performance compared to default implementation, also avoids errors with some complex arrays
# @Zygote.adjoint function norm(A::AbstractArray, p::Real = 2)
#   n = norm(A, p)
#   back(Δ) = let n = n
#               (Δ .* A ./ (n + eps(0f0)), )
#           end
#   return n, back
# end

# """
# 		svd(A) -> Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}
# """
# function svd(A)
# 	U, S, V = LinearAlgebra.svd(A)
# 	return U, S, Matrix(V)
# end


# """
#     svd_back(U, S, V, dU, dS, dV)
# adjoint for SVD decomposition.
# References:
#     https://j-towns.github.io/papers/svd-derivative.pdf
#     https://giggleliu.github.io/2019/04/02/einsumbp.html
# """
# function svd_back(U, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    
#   all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
#   η = T(η);
#   S2 = S.^2;
#   Sinv = S ./ (S2 .+ η);
#   F = S2' .- S2;
#   F ./= ((F.^2) .+ η);

#   # res = ZeroAdder()
#   res = zeros(Complex{T}, size(S));
#   if !(dU isa Nothing)
#       UdU = U' * dU;
#       J = F .* UdU;
#       res += (J + J') * S + 1im * imag(UdU) .* Sinv;
#   end
#   if !(dV isa Nothing)
#       VdV = V' * dV;
#       K = F .* VdV;
#       res += S * (K + K');
#   end
#   if !(dS isa Nothing)
#       res += dS;
#   end

#   res = U * res * V';

#   if !(dU isa Nothing) && size(U, 1) != size(U, 2)
#       res += (dU - U * (U' * dU)) * Sinv * V';
#   end

#   if !(dV isa Nothing) && size(V, 1) != size(V, 2)
#       res += U * Sinv * (dV' - (dV' * V) * V');
#   end

#   return res

# end

# @Zygote.adjoint function svd(A)
#   U, S, V = svd(A);
#   S = diagm(S);
#   # V = V' ?
#   (U, S, V), dy -> (svd_back(U, S, V, dy...), );
# end